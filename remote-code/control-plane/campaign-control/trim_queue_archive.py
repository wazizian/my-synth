"""Thin a terminal queue archive, retaining metadata and verified immutable bytes.

Dry-run by default. --apply publishes and verifies a thin tar.gz plus recovery
inventory before deleting only the original regenerable queue tar.gz. Omitted
.eqx/.npz payloads must remain available at their byte-identical root paths.
No numerical imports, rentals, remote calls, or project source modifications.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import uuid

ROOT = Path('/home/jump/deep-hedging')
TERMINAL = {'complete', 'completed', 'failed', 'stopped', 'not_started'}


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def fingerprint(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or any(parent.is_symlink() for parent in path.parents):
        raise RuntimeError('Expected a regular non-symlink path: ' + str(path))
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def stream_hash(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''): value.update(block)
    return value.hexdigest()


def file_hash(path):
    before = fingerprint(path)
    with path.open('rb') as stream:
        info = os.fstat(stream.fileno())
        if (info.st_dev, info.st_ino) != before[:2]: raise RuntimeError('File changed while opening')
        result = stream_hash(stream)
    if fingerprint(path) != before: raise RuntimeError('File changed while hashing: ' + str(path))
    return result


def assert_terminal(archive):
    lease_name = archive.name.removesuffix('.tar.gz')
    state = read(ROOT / 'queue/state.json')
    matches = [phase['leases'][lease_name] for phase in state.get('phases', {}).values()
               if lease_name in phase.get('leases', {})]
    if len(matches) != 1 or matches[0].get('status') not in TERMINAL:
        raise RuntimeError('Archive requires one verified terminal lease: ' + lease_name)
    pid = matches[0].get('pid')
    if isinstance(pid, int) and pid > 0:
        try: os.kill(pid, 0)
        except ProcessLookupError: pass
        except PermissionError: raise RuntimeError('Lease process may still be running')
        else: raise RuntimeError('Lease PID still exists; retain archive')
    worker = ROOT / 'workers' / lease_name
    if (worker / 'active.json').exists(): raise RuntimeError('Worker active.json exists; retain archive')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid(): continue
        try: command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except OSError: continue
        if str(worker) in command or '--worker ' + lease_name + ' ' in command:
            raise RuntimeError('Live process references worker; retain archive')
    return lease_name


def safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name:
        raise RuntimeError('Unsafe archive member: ' + name)
    if path.as_posix() in ('.', ''): raise RuntimeError('Empty archive member path')
    return path.as_posix()


def may_omit(name):
    """Whitelist only immutable scientific payloads, never metadata/source/Git."""
    parts = PurePosixPath(name).parts
    suffix = PurePosixPath(name).suffix
    if suffix == '.eqx' and len(parts) >= 4 and parts[0] == 'runs':
        return (parts[2] == 'history' or
                (parts[2] == 'checkpoints' and len(parts) == 5 and parts[-1] == 'state.eqx'))
    if suffix == '.npz':
        return (parts[0] == 'data' or parts[:2] == ('cache', 'pricing') or
                (len(parts) >= 4 and parts[0] == 'runs' and parts[2] == 'episode_metrics'))
    # JAX compiled binaries are retained. They are not scientific immutable .eqx/.npz.
    return False


def sync(path):
    fd = os.open(path, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def atomic_json(path, value):
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync(path.parent)
    finally:
        if temporary.exists(): temporary.unlink()


def inspect(archive, output=None):
    records, names, omitted_bytes = [], set(), 0
    prior_members = {}
    archive_before = fingerprint(archive)
    with tarfile.open(archive, 'r:gz') as source:
        for member in source:
            name = safe_name(member.name)
            if name in names: raise RuntimeError('Duplicate tar member name: ' + name)
            names.add(name)
            if not member.isdir() and not member.isfile() and not member.islnk():
                raise RuntimeError('Symlinks/devices/special members prohibited: ' + name)
            metadata = dict(name=name, size=member.size, type='directory' if member.isdir() else 'file',
                            mode=member.mode, uid=member.uid, gid=member.gid, mtime=member.mtime,
                            uname=member.uname, gname=member.gname, pax_headers=member.pax_headers)
            if member.islnk():
                target = safe_name(member.linkname)
                if target == name:
                    raise RuntimeError('Self-referential hardlink prohibited: ' + name)
                # TarFile's own link resolver is deliberately not used. Only
                # already verified earlier members may be targets: this bounds
                # resolution, rules out cycles/forward ambiguity and matches
                # the hardlinks generated by Python tarfile during packaging.
                if target not in prior_members:
                    raise RuntimeError('Missing/forward hardlink target unsupported: ' + target)
                target_record = prior_members[target]
                if target_record['type'] not in ('file', 'hardlink'):
                    raise RuntimeError('Hardlink target is not a verified regular payload: ' + target)
                canonical = target_record.get('canonical_target', target)
                canonical_record = prior_members.get(canonical)
                if canonical_record is None or canonical_record['type'] != 'file':
                    raise RuntimeError('Hardlink chain does not resolve to an earlier regular member')
                if not (may_omit(name) and may_omit(target) and may_omit(canonical)
                        and target_record['omitted'] and canonical_record['omitted']):
                    raise RuntimeError('Non-allowlisted hardlink retained by aborting trim: ' + name)
                if member.size != 0:
                    raise RuntimeError('Hardlink with unexpected embedded payload prohibited: ' + name)
                payload_size, digest = canonical_record['size'], canonical_record['sha256']
                if target_record['size'] != payload_size or target_record['sha256'] != digest:
                    raise RuntimeError('Hardlink chain content metadata mismatch')
                for verified_name in {name, target, canonical}:
                    current = ROOT / verified_name
                    if not current.exists() or fingerprint(current)[2] != payload_size or file_hash(current) != digest:
                        raise RuntimeError('Hardlink payload lacks identical root copy: ' + verified_name)
                metadata.update(type='hardlink', size=payload_size, archive_size=member.size,
                                linkname=member.linkname, canonical_target=canonical,
                                sha256=digest, omitted=True, recovery_path=str(ROOT / name))
                # The link itself carried no archive payload; do not count the
                # referenced bytes twice in the removable uncompressed total.
            elif member.isdir():
                metadata['omitted'] = False
                if output is not None: output.addfile(member)
            else:
                with source.extractfile(member) as content: digest = stream_hash(content)
                omit = may_omit(name)
                metadata.update(sha256=digest, omitted=omit)
                if omit:
                    current = ROOT / name
                    if not current.exists() or fingerprint(current)[2] != member.size or file_hash(current) != digest:
                        raise RuntimeError('Immutable payload lacks identical root copy: ' + name)
                    metadata['recovery_path'] = str(current)
                    omitted_bytes += member.size
                elif output is not None:
                    # Reopen exact member bytes after the hash pass; TarInfo retains
                    # timestamps, permissions, owner names and PAX metadata.
                    with source.extractfile(member) as content: output.addfile(member, content)
            records.append(metadata)
            prior_members[name] = metadata
    if fingerprint(archive) != archive_before: raise RuntimeError('Original archive changed during scan')
    return records, omitted_bytes


def verify_thin(path, records):
    retained = [record for record in records if not record['omitted']]
    index = 0
    with tarfile.open(path, 'r:gz') as source:
        for member in source:
            if index >= len(retained): raise RuntimeError('Unexpected thin archive member')
            record = retained[index]; index += 1
            actual = dict(name=safe_name(member.name), size=member.size,
                          type='directory' if member.isdir() else 'file', mode=member.mode,
                          uid=member.uid, gid=member.gid, mtime=member.mtime,
                          uname=member.uname, gname=member.gname, pax_headers=member.pax_headers)
            if any(record[key] != value for key, value in actual.items()):
                raise RuntimeError('Thin archive metadata mismatch: ' + member.name)
            if member.isfile():
                with source.extractfile(member) as content:
                    if stream_hash(content) != record['sha256']:
                        raise RuntimeError('Thin archive content mismatch: ' + member.name)
    if index != len(retained): raise RuntimeError('Thin archive is missing retained members')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path, help='Existing /home/jump/deep-hedging/queue/LEASE.tar.gz')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    archive = args.archive.absolute()
    if archive.parent != ROOT / 'queue' or not archive.name.endswith('.tar.gz'):
        parser.error('Only direct queue/*.tar.gz archives are permitted')
    fingerprint(archive)
    helper_root = Path(__file__).resolve().parent
    with (helper_root / 'trim_queue_archive.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lease = assert_terminal(archive)
        original_fingerprint = fingerprint(archive)
        original_hash = file_hash(archive)
        records, omitted_bytes = inspect(archive)
        if not any(record['omitted'] for record in records):
            raise RuntimeError('No verified scientific payloads to omit; original retained')
        report = dict(mode='apply' if args.apply else 'dry_run', archive=str(archive), lease=lease,
                      original_sha256=original_hash, original_bytes=original_fingerprint[2],
                      omitted_uncompressed_bytes=omitted_bytes, omitted_members=sum(r['omitted'] for r in records),
                      retained_members=sum(not r['omitted'] for r in records))
        if args.apply:
            destination = helper_root / 'thin_queue_archives'
            destination.mkdir(exist_ok=True)
            stem = lease + '-' + original_hash[:16]
            thin = destination / (stem + '.metadata.tar.gz')
            inventory = destination / (stem + '.inventory.json')
            if thin.exists() or inventory.exists():
                raise RuntimeError('Recovery artifacts already exist; inspect previous attempt before retrying')
            temporary = destination / ('.' + stem + '.' + uuid.uuid4().hex + '.tmp')
            try:
                with tarfile.open(temporary, 'w:gz', format=tarfile.PAX_FORMAT) as output:
                    second_records, _ = inspect(archive, output)
                if second_records != records: raise RuntimeError('Archive changed between verification passes')
                sync(temporary)
                verify_thin(temporary, records)
                os.replace(temporary, thin); sync(destination)
                report.update(thin_archive=str(thin), thin_sha256=file_hash(thin), thin_bytes=thin.stat().st_size,
                              inventory=str(inventory), original_deleted=False,
                              recovery='Retained metadata/files in thin archive plus omitted payloads at recovery_path; verify every size/SHA256. Omitted hardlinks retain original linkname/archive_size and canonical_target; recreate only after the canonical regular member, or restore equivalent verified bytes.',
                              members=records)
                atomic_json(inventory, report)
                verify_thin(thin, records)
                saved = read(inventory)
                if saved != report or file_hash(thin) != saved['thin_sha256']:
                    raise RuntimeError('Recovery artifacts failed durable verification')
                for record in records:
                    if record['omitted']:
                        current = Path(record['recovery_path'])
                        if fingerprint(current)[2] != record['size'] or file_hash(current) != record['sha256']:
                            raise RuntimeError('Recovery payload changed; original retained')
                assert_terminal(archive)
                if fingerprint(archive) != original_fingerprint or file_hash(archive) != original_hash:
                    raise RuntimeError('Original archive changed; not deleting')
                archive.unlink(); sync(archive.parent)
                report['original_deleted'] = True
                atomic_json(inventory, report)
            finally:
                if temporary.exists(): temporary.unlink()
        print(json.dumps({key: value for key, value in report.items() if key != 'members'}, indent=2))


if __name__ == '__main__':
    main()
