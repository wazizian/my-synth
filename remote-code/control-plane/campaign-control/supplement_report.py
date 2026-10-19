"""Post-report stdlib-only checkpoint/evaluator provenance and cost-scope export.

Reads final CSVs and checkpoint metadata without numerical imports. Never alters
original reports or invents per-trial allocations of shared worker charges.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import stat

DOMAINS = ('base', 'easy', 'hard')


def safe(base, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe input path: ' + str(relative))
    path = base / relative
    if any(item.is_symlink() for item in (path, *path.parents)) or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('Input must be a regular non-symlink file: ' + str(path))
    return path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    fields = sorted({name for row in rows for name in row})
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/jump/deep-hedging'))
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory outside the frozen project')
    args = parser.parse_args()
    root, output = args.root.absolute(), args.output_dir.absolute()
    if root == output or root in output.parents:
        parser.error('Supplement output must be outside the frozen project')
    if output.exists(): parser.error('Output directory must be new; original supplements are never overwritten')
    rows, input_hashes, errors = [], {}, []
    for table_name in ('final_metrics.csv', 'all_trials.csv'):
        path = safe(root, 'results/' + table_name)
        input_hashes[str(path.relative_to(root))] = digest(path)
        with path.open(newline='') as stream:
            for source in csv.DictReader(stream):
                run_id, checkpoint_id = source.get('run_id'), source.get('checkpoint_id')
                row = dict(source_table=table_name, run_id=run_id, method=source.get('method'),
                           checkpoint_id=checkpoint_id, reported_status=source.get('status'),
                           lineage_active_training_seconds=source.get('lineage_active_training_seconds'),
                           checkpoint_active_training_seconds=source.get('checkpoint_active_training_seconds'),
                           metadata_status='unverified')
                for domain in DOMAINS:
                    for field in ('code_commit', 'source_hash', 'worker', 'instance', 'gpu_model', 'timestamp'):
                        row[domain + '_evaluator_' + field] = source.get(domain + '_' + field)
                try:
                    if not run_id or Path(run_id).name != run_id: raise ValueError('Missing/unsafe run ID')
                    if not checkpoint_id:
                        row.update(metadata_status='checkpoint_unavailable', reason=source.get('reason') or
                                   'No checkpoint ID in preserved report row; inspect original run failure status')
                    else:
                        if Path(checkpoint_id).name != checkpoint_id: raise ValueError('Unsafe checkpoint ID')
                        metadata_path = safe(root, Path('runs') / run_id / 'checkpoints' / checkpoint_id / 'metadata.json')
                        metadata = read(metadata_path)
                        if metadata.get('checkpoint_id') != checkpoint_id or metadata.get('run_id') != run_id:
                            raise ValueError('Checkpoint identity differs from report row')
                        if source.get('method') != metadata.get('method'):
                            raise ValueError('Checkpoint method differs from report row')
                        counters = metadata.get('counters', {})
                        row.update(metadata_status='verified_metadata_binding',
                                   metadata_path=str(metadata_path.relative_to(root)), metadata_sha256=digest(metadata_path),
                                   checkpoint_code_commit=metadata.get('code_commit'),
                                   checkpoint_source_hash=metadata.get('source_hash'),
                                   checkpoint_config_hash=metadata.get('config_hash'),
                                   checkpoint_state_sha256=metadata.get('state_sha256'),
                                   checkpoint_created_at=metadata.get('created_at'),
                                   checkpoint_worker=metadata.get('worker'), checkpoint_instance=metadata.get('instance'),
                                   checkpoint_gpu_model=metadata.get('gpu_model'),
                                   checkpoint_learner_steps=counters.get('learner_steps'),
                                   checkpoint_outer_rounds=counters.get('outer_rounds'),
                                   checkpoint_metadata_active_training_seconds=counters.get('active_training_seconds'))
                        for domain in DOMAINS:
                            evaluator_commit = row.get(domain + '_evaluator_code_commit')
                            evaluator_hash = row.get(domain + '_evaluator_source_hash')
                            row[domain + '_evaluator_differs_from_checkpoint'] = (
                                evaluator_commit != metadata.get('code_commit') or evaluator_hash != metadata.get('source_hash')) if evaluator_commit or evaluator_hash else None
                except Exception as error:
                    row.update(metadata_status='invalid', reason=str(error))
                    errors.append({'source_table': table_name, 'run_id': run_id, 'checkpoint_id': checkpoint_id, 'reason': str(error)})
                rows.append(row)
    final_billing = root / 'budget/provider_charges_final.json'
    billing_relative = ('budget/provider_charges_final.json' if final_billing.exists()
                        else 'budget/provider_charges_latest.json')
    billing_path = safe(root, billing_relative)
    billing = read(billing_path)
    input_hashes[billing_relative] = digest(billing_path)
    ledger = safe(root, 'budget/ledger.jsonl')
    input_hashes['budget/ledger.jsonl'] = digest(ledger)
    provider_workers = []
    for worker, amount in sorted(billing.get('per_worker_reported_usd', {}).items()):
        instances = sorted(instance for instance, name in billing.get('instance_workers', {}).items() if name == worker)
        provider_workers.append(dict(worker=worker, provider_reported_usd=amount,
                                     instance_ids=json.dumps(instances), provider_snapshot_status=billing.get('status'),
                                     provider_snapshot_queried_at=billing.get('queried_at'), canonical_source=billing_relative))
    cost_scope = dict(
        canonical_provider_per_worker_source=billing_relative,
        canonical_provider_source_sha256=input_hashes[billing_relative],
        provider_snapshot_status=billing.get('status'), provider_snapshot_queried_at=billing.get('queried_at'),
        provider_reported_total_usd=billing.get('total_reported_usd'),
        provider_scope=billing.get('scope'), active_campaign_instances=billing.get('active_campaign_instances'),
        unreported_instance_ids=billing.get('unreported_instance_ids'),
        control_plane_ledger_source='budget/ledger.jsonl', control_plane_ledger_sha256=input_hashes['budget/ledger.jsonl'],
        cost_interpretation='Provider per-worker charges are canonical observed billing. budget/ledger.jsonl records reservations, conservative estimates, reconciliation and lifecycle evidence. Billing may settle after shutdown.',
        trial_cost_scope='Domain instance-cost fields in original evaluation tables describe the evaluation worker cumulative rental estimate. They are not allocated trial training/tuning costs, must not be summed across domains/checkpoints, and do not replace the provider ledger.',
        allocation='No per-trial cost allocation is invented here. Shared-worker and setup/compilation/export costs remain at worker/campaign scope unless an explicit independently supported allocation is later supplied.',
        provenance_interpretation='checkpoint_code_commit/source_hash identify the code recorded when that checkpoint was saved. *_evaluator_code_commit/source_hash identify the code executing its final evaluation. A difference is disclosed, not silently treated as matching or automatically invalid.',
        limitation='This supplement verifies metadata identities and preserves source hashes. It does not deserialize checkpoints, re-evaluate metrics, verify provider state live, or establish per-checkpoint semantic restore; use the separate GPU and recovery audits.')
    # Verify original report files were stable during this read-only extraction.
    for name, expected in input_hashes.items():
        if digest(safe(root, name)) != expected: raise RuntimeError('Input changed while supplementing: ' + name)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'checkpoint_evaluation_provenance.csv', rows)
    write_csv(output / 'provider_worker_costs.csv', provider_workers)
    write_json(output / 'cost_and_provenance_scope.json', cost_scope)
    write_json(output / 'supplement_manifest.json', dict(passed=not errors, inputs=input_hashes,
        rows=len(rows), metadata_errors=errors, originals_modified=False,
        outputs=['checkpoint_evaluation_provenance.csv', 'provider_worker_costs.csv', 'cost_and_provenance_scope.json']))
    print(json.dumps({'passed': not errors, 'rows': len(rows), 'output_directory': str(output)}))
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
