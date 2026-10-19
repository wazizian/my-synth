"""Final metadata/byte-hash recoverability audit; no numerical imports or loads.

Only hashes raw .eqx/.npz bytes. Verifies source against loose Git commit/tree/blob
objects using stdlib zlib/SHA1. Pricing's logical array hash is bound indirectly
through a terminal GPU worker inventory and its passed validation/smoke reports;
it is never confused with the NPZ file hash. Does not evaluate final-test metrics.
Run only after exports are quiescent. Complements, never replaces, GPU restore tests.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zlib

HEX40 = re.compile(r'^[0-9a-f]{40}$')
HEX64 = re.compile(r'^[0-9a-f]{64}$')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def safe(base, name, directory=False):
    relative = PurePosixPath(str(name))
    if relative.is_absolute() or '..' in relative.parts or '\\' in str(name):
        raise ValueError('unsafe relative path: ' + str(name))
    path = base.joinpath(*relative.parts)
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('symlink prohibited: ' + str(path))
    info = path.stat()
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise ValueError('unexpected filesystem type: ' + str(path))
    return path


def read(path):
    safe(path.parent, path.name)
    return json.loads(path.read_text())


def check(condition, message):
    if not condition: raise ValueError(message)


class Audit:
    def __init__(self, root):
        self.root = root
        self.hash_cache = {}
        self.hash_bytes = 0
        self.sources = {}
        self.failures = []
        self.limitations = []
        self.object_cache = {}

    def fail(self, scope, error):
        self.failures.append({'scope': str(scope), 'error': str(error)})

    def digest(self, path):
        safe(path.parent, path.name)
        info = path.stat()
        signature = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if signature not in self.hash_cache:
            value = hashlib.sha256()
            with path.open('rb') as stream:
                opened = os.fstat(stream.fileno())
                check((opened.st_dev, opened.st_ino) == signature[:2], 'file changed while opening')
                for block in iter(lambda: stream.read(1024 * 1024), b''): value.update(block)
            after = path.stat()
            check(signature == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                  'file changed while hashing: ' + str(path))
            self.hash_cache[signature] = value.hexdigest()
            self.hash_bytes += info.st_size
        return self.hash_cache[signature]

    def git_object(self, snapshot, oid, expected_type):
        check(bool(HEX40.fullmatch(oid)), 'invalid Git object SHA1')
        cache_key = (str(snapshot), oid)
        if cache_key not in self.object_cache:
            object_path = snapshot / '.git/objects' / oid[:2] / oid[2:]
            if not object_path.exists():
                limitation = 'Loose Git object unavailable (packed or missing): ' + str(object_path)
                self.limitations.append(limitation)
                raise ValueError(limitation)
            safe(snapshot, '.git/objects/' + oid[:2] + '/' + oid[2:])
            check(object_path.stat().st_size <= 64 * 1024 ** 2, 'Git object exceeds metadata audit bound')
            decoder = zlib.decompressobj()
            raw = decoder.decompress(object_path.read_bytes(), 64 * 1024 ** 2 + 1)
            check(len(raw) <= 64 * 1024 ** 2 and decoder.eof and not decoder.unused_data,
                  'Invalid or oversized loose Git object')
            check(hashlib.sha1(raw).hexdigest() == oid, 'Git object SHA1 mismatch: ' + oid)
            header, payload = raw.split(b'\0', 1)
            kind, length = header.split(b' ', 1)
            check(int(length) == len(payload), 'Git object declared size mismatch')
            self.object_cache[cache_key] = (kind.decode('ascii'), payload)
        kind, payload = self.object_cache[cache_key]
        check(kind == expected_type, 'unexpected Git object type: ' + kind)
        return payload

    def tree_files(self, snapshot, oid, prefix=''):
        payload = self.git_object(snapshot, oid, 'tree')
        entries, offset = {}, 0
        while offset < len(payload):
            space = payload.index(b' ', offset)
            end = payload.index(b'\0', space)
            mode = payload[offset:space].decode('ascii')
            name = payload[space + 1:end].decode('utf-8')
            check(name not in ('', '.', '..') and '/' not in name and '\\' not in name, 'unsafe Git tree path')
            child_oid = payload[end + 1:end + 21].hex()
            check(len(child_oid) == 40, 'truncated Git tree')
            offset = end + 21
            full_name = prefix + name
            if mode in ('40000', '040000'):
                entries.update(self.tree_files(snapshot, child_oid, full_name + '/'))
            else:
                check(mode in ('100644', '100755'), 'Git symlink/submodule/special entry prohibited')
                check(full_name not in entries, 'duplicate Git tree entry')
                entries[full_name] = child_oid
        return entries

    def source(self, commit):
        if commit in self.sources: return self.sources[commit]
        result = {'commit': commit, 'passed': False}
        self.sources[commit] = result
        try:
            check(isinstance(commit, str) and bool(HEX40.fullmatch(commit)), 'missing/invalid code_commit')
            snapshot = safe(self.root, 'source_snapshots/' + commit, directory=True)
            provenance = read(safe(snapshot, 'code_provenance.json'))
            check(provenance.get('commit') == commit, 'provenance commit mismatch')
            files = provenance['files']
            check(files and canonical_hash(files) == provenance['tracked_source_sha256'], 'provenance manifest hash mismatch')
            actual = {}
            for name, expected in files.items():
                actual[name] = self.digest(safe(snapshot, name))
                check(actual[name] == expected, 'source bytes mismatch: ' + name)
            runtime = {name: digest for name, digest in actual.items() if name.split('/')[0] in ('src', 'scripts', 'configs')}
            check(canonical_hash(runtime) == provenance['source_hash'], 'source_hash mismatch')
            commit_data = self.git_object(snapshot, commit, 'commit')
            tree = next((line[5:].decode('ascii') for line in commit_data.split(b'\n') if line.startswith(b'tree ')), None)
            check(tree == provenance['git_tree'], 'commit tree disagrees with provenance')
            committed = self.tree_files(snapshot, tree)
            check(set(committed) == set(files), 'Git tracked paths differ from provenance files')
            for name, oid in committed.items():
                blob = self.git_object(snapshot, oid, 'blob')
                check(hashlib.sha256(blob).hexdigest() == actual[name], 'Git blob differs from saved source: ' + name)
            result.update(passed=True, source_hash=provenance['source_hash'], git_tree=tree,
                          verified_files=len(files), git_verification='SHA1-verified loose commit/tree/blob graph and SHA256 source bytes')
        except Exception as error:
            result['error'] = str(error)
            self.fail('source:' + str(commit), error)
        return result

    def pricing_binding(self, inventory_path, logical_hash):
        result = {'passed': False, 'scope': 'indirect binding through terminal GPU worker inventory and passed reports'}
        try:
            check(inventory_path is not None, 'Supply --pricing-inventory from a terminal validated GPU worker')
            inventory_path = inventory_path.absolute()
            safe(self.root, str(inventory_path.relative_to(self.root)))
            worker = inventory_path.parents[2]
            check(worker.parent == self.root / 'workers', 'pricing inventory must be workers/WORKER/runs/RUN/artifacts.json')
            check(not (worker / 'active.json').exists(), 'pricing evidence worker is still marked active')
            rental = read(safe(worker, 'rental_history.json'))
            check(bool(rental.get('destroyed_at')), 'pricing evidence lacks terminal rental record')
            records = read(inventory_path)
            indexed = {record['path']: record for record in records}
            check(len(indexed) == len(records), 'duplicate inventory paths')
            project = inventory_path.parent / 'outputs/project'
            def evidence(name):
                record = indexed['project/' + name]
                path = safe(project, name)
                check(path.stat().st_size == record['bytes'] and self.digest(path) == record['sha256'],
                      'worker evidence disagrees with inventory: ' + name)
                return read(path)
            table_record = indexed['project/cache/pricing/heston.npz']
            root_table = safe(self.root, 'cache/pricing/heston.npz')
            check(root_table.stat().st_size == table_record['bytes'] and self.digest(root_table) == table_record['sha256'],
                  'ROOT pricing table bytes differ from terminal GPU inventory')
            validation = evidence('cache/pricing/heston.validation.json')
            fallback = evidence('cache/pricing/fallback.validation.json')
            smoke = evidence('results/integrated_smoke.json')
            manifest = evidence('data/manifest.json')
            check(validation.get('passed') and validation.get('sha256') == logical_hash, 'GPU pricing validation/logical hash mismatch')
            check(fallback.get('passed') and fallback.get('pricing_hash') == logical_hash, 'GPU fallback validation/logical hash mismatch')
            check(smoke.get('passed') and manifest.get('pricing_hash') == logical_hash, 'GPU smoke/data binding missing')
            profile_key = 'project/results/production_profile_gate.json'
            if profile_key in indexed:
                check(evidence('results/production_profile_gate.json').get('passed'), 'GPU production profile gate failed')
            resume_names = ('actual_resume_completed', 'actual_resume_adam_advances', 'actual_resume_key_advances',
                            'actual_resume_preserves_history', 'actual_resume_round_advances', 'actual_resume_learner_changes')
            check(all(smoke.get('checks', {}).get(name, {}).get('passed') for name in resume_names), 'GPU actual-resume evidence incomplete')
            result.update(passed=True, inventory=str(inventory_path), inventory_sha256=self.digest(inventory_path),
                          npz_file_sha256=table_record['sha256'], logical_pricing_hash=logical_hash,
                          terminal_worker=worker.name, gpu_resume_checks=list(resume_names),
                          semantic_scope='GPU smoke proves actual resume mechanism; this audit verifies bytes/references for each retained checkpoint without deserializing it')
        except Exception as error:
            result['error'] = str(error)
            self.fail('pricing_binding', error)
        return result

    def execute(self, inventory_path):
        manifest = read(safe(self.root, 'data/manifest.json'))
        scales = read(safe(self.root, 'objective_scales.json'))
        expected = dict(datasets=canonical_hash(manifest['datasets']), benchmark=manifest['benchmark_hash'],
                        scales=self.digest(self.root / 'objective_scales.json'), pricing=manifest['pricing_hash'])
        benchmark = {key: value for key, value in manifest.items() if key not in ('datasets', 'benchmark_hash', 'objective_scales_hash')}
        check(canonical_hash(benchmark) == expected['benchmark'], 'benchmark definition hash mismatch')
        check(manifest['objective_scales_hash'] == expected['scales'], 'dataset manifest scale hash mismatch')
        for name, record in manifest['datasets'].items():
            try: check(self.digest(safe(self.root, record['path'])) == record['sha256'], 'dataset file hash mismatch')
            except Exception as error: self.fail('dataset:' + name, error)
        root_validation = read(safe(self.root, 'cache/pricing/heston.validation.json'))
        check(root_validation.get('passed') and root_validation['sha256'] == expected['pricing'], 'ROOT pricing validation hash/status mismatch')
        pricing = self.pricing_binding(inventory_path, expected['pricing'])
        runs, checkpoint_count, history_references = [], 0, 0
        for directory in sorted((self.root / 'runs').iterdir()):
            if not directory.is_dir(): continue
            row = dict(run_id=directory.name, checkpoints=0, verified_checkpoints=0, history_references=0, pointers=0)
            runs.append(row)
            try:
                safe(self.root / 'runs', directory.name, directory=True)
                config = read(safe(directory, 'config.json'))
                config_hash = canonical_hash(config)
                checkpoint_directory = safe(directory, 'checkpoints', directory=True)
                bundles = {}
                for metadata_path in sorted(checkpoint_directory.glob('*/metadata.json')):
                    checkpoint_count += 1; row['checkpoints'] += 1
                    try:
                        safe(checkpoint_directory, str(metadata_path.relative_to(checkpoint_directory)))
                        metadata = read(metadata_path)
                        identifier = metadata_path.parent.name
                        check(metadata.get('checkpoint_id') == identifier, 'checkpoint identifier mismatch')
                        check(metadata.get('format_version') == 1, 'unsupported checkpoint format')
                        check(metadata.get('run_id') == directory.name, 'checkpoint run identifier mismatch')
                        check(metadata.get('config') == config and metadata.get('config_hash') == config_hash, 'immutable config binding mismatch')
                        check(metadata.get('hashes') == expected, 'checkpoint dataset/benchmark/pricing/scale binding mismatch')
                        check(metadata.get('objective_scales') == scales['scales'], 'checkpoint objective scale values mismatch')
                        check(metadata.get('benchmark_hash') == expected['benchmark'] and metadata.get('dataset_manifest_hash') == expected['datasets']
                              and metadata.get('pricing_table_hash') == expected['pricing'], 'duplicated artifact binding fields disagree')
                        check(self.digest(safe(metadata_path.parent, 'state.eqx')) == metadata['state_sha256'], 'checkpoint state SHA256 mismatch')
                        source = self.source(metadata.get('code_commit'))
                        check(source.get('passed') and metadata.get('source_hash') == source.get('source_hash'), 'checkpoint source provenance unverified/mismatched')
                        counters = metadata['counters']
                        rounds = counters['outer_rounds']
                        check(isinstance(rounds, int) and rounds >= 0, 'invalid outer-round counter')
                        check(counters['learner_steps'] == rounds * config['block_steps'], 'learner-step/block counter mismatch')
                        check(counters['synthetic_path_count'] == rounds * config['block_steps'] * config['batch_size'], 'synthetic path counter mismatch')
                        history = metadata.get('history', [])
                        if config['method'] == 'selfplay':
                            # One extra pre-update snapshot may have been saved before
                            # an interrupted/rejected block. It is safe for exact resume.
                            check(len(history) in (rounds, rounds + 1), 'selfplay history length does not cover exact outer rounds')
                            for index, record in enumerate(history):
                                check(record.get('index') == index, 'selfplay history index is missing/noncontiguous')
                                check(record['path'] == f'{index:08d}.eqx', 'selfplay snapshot filename/index mismatch')
                                snapshot = safe(directory / 'history', record['path'])
                                check(self.digest(snapshot) == record['sha256'], 'selfplay snapshot hash mismatch: ' + str(index))
                                if 'bytes' in record: check(snapshot.stat().st_size == record['bytes'], 'selfplay snapshot byte count mismatch')
                            row['history_references'] += len(history); history_references += len(history)
                        else: check(not history, 'non-selfplay checkpoint unexpectedly references history')
                        bundles[identifier] = {'state_sha256': metadata['state_sha256']}
                        row['verified_checkpoints'] += 1
                    except Exception as error: self.fail(metadata_path.relative_to(self.root), error)
                for tag in ('latest', 'previous', 'best', 'final'):
                    pointer_path = checkpoint_directory / (tag + '.json')
                    if not pointer_path.exists(): continue
                    try:
                        pointer = read(pointer_path)
                        identifier = pointer['checkpoint_id']
                        check(pointer['path'] == identifier and Path(identifier).name == identifier, 'unsafe/mismatched pointer path')
                        bundle = safe(checkpoint_directory, identifier, directory=True)
                        check(self.digest(safe(bundle, 'metadata.json')) == pointer['metadata_sha256'], 'pointer metadata hash mismatch')
                        check(self.digest(safe(bundle, 'state.eqx')) == pointer['state_sha256'], 'pointer state hash mismatch')
                        check(identifier in bundles and bundles[identifier]['state_sha256'] == pointer['state_sha256'], 'pointer references unverified checkpoint')
                        row['pointers'] += 1
                    except Exception as error: self.fail(pointer_path.relative_to(self.root), error)
                check(row['checkpoints'] > 0, 'no recoverable checkpoint bundle')
            except Exception as error: self.fail(directory.relative_to(self.root), error)
        check(bool(runs), 'no run directories found')
        return dict(passed=not self.failures and not self.limitations, scope='metadata/byte hashes and source-object integrity; no numerical deserialization',
                    checkpoint_count=checkpoint_count, history_reference_count=history_references, runs=runs,
                    unique_inode_payloads_hashed=len(self.hash_cache), bytes_hashed=self.hash_bytes,
                    source_snapshots=list(self.sources.values()), pricing_binding=pricing,
                    failures=self.failures, limitations=sorted(set(self.limitations)),
                    semantic_restore_scope='Passed GPU actual-resume tests support implementation; per-checkpoint semantic restore is not executed by this metadata-only auditor')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/jump/deep-hedging'))
    parser.add_argument('--pricing-inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.absolute()
    output = args.output.absolute()
    check(root not in output.parents and output != root, 'Audit output must be outside the frozen project')
    audit = Audit(root)
    try: result = audit.execute(args.pricing_inventory)
    except Exception as error:
        audit.fail('audit_setup', error)
        result = dict(passed=False, failures=audit.failures, limitations=audit.limitations,
                      source_snapshots=list(audit.sources.values()), bytes_hashed=audit.hash_bytes)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result.get(key) for key in ('passed', 'checkpoint_count', 'history_reference_count', 'unique_inode_payloads_hashed')},
                     sort_keys=True))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
