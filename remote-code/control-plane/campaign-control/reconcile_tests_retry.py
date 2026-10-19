#!/usr/bin/env python3
"""Offline metadata reconciliation of the authorized tests-0 and tests-2 replacements.

Run only after the original controller and both replacement wrappers have exited:
  python3 /home/jump/campaign-control/reconcile_tests_retry.py
No provider, rental, numerical, training, or source/config mutation calls.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path('/home/jump/deep-hedging')
CONTROL = Path('/home/jump/campaign-control')
EXPECTED = ['baseline-lr001', 'hard_reference-lr001', 'profile_dpg_mixed', 'profile_soar_hard',
            'smoke_baseline', 'smoke_hard_reference', 'smoke_soar_mixed']
GROUPS = {
    'tests-0': ('tests-retry0', EXPECTED),
    'tests-2': ('tests-retry2', ['dpg_mixed-lr001', 'profile_dpg_hard', 'profile_selfplay',
                'selfplay-lr001', 'smoke_dpg_mixed', 'smoke_soar_hard', 'soar_mixed-lr001']),
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_run(project, run_id):
    directory = project / 'runs' / run_id
    canonical = ROOT / 'runs' / run_id
    require(read(directory / 'config.json') == read(canonical / 'config.json'),
            f'Immutable config changed: {run_id}')
    for pointer in ('best.json', 'latest.json', 'final.json'):
        require(read(directory / 'checkpoints' / pointer) == read(canonical / 'checkpoints' / pointer),
                f'Checkpoint pointer changed: {run_id}/{pointer}')
    expected = {p.parent.name for p in (canonical / 'checkpoints').glob('*/metadata.json')}
    actual = {p.parent.name for p in (directory / 'checkpoints').glob('*/metadata.json')}
    require(expected and actual == expected, f'Checkpoint set changed: {run_id}')
    evidence = {}
    for checkpoint_id in sorted(expected):
        bundle = directory / 'checkpoints' / checkpoint_id
        require(not bundle.is_symlink(), 'Symlinked checkpoint')
        for filename in ('metadata.json', 'state.eqx'):
            path = bundle / filename
            require(path.is_file() and not path.is_symlink()
                    and digest(path) == digest(canonical / 'checkpoints' / checkpoint_id / filename),
                    f'Immutable checkpoint changed: {path}')
        metadata = read(bundle / 'metadata.json')
        require(metadata['checkpoint_id'] == checkpoint_id and metadata['run_id'] == run_id
                and metadata['state_sha256'] == digest(bundle / 'state.eqx'), 'Checkpoint identity/hash mismatch')
        summary_path = directory / f'final_test_{checkpoint_id}.json'
        summary = read(summary_path)
        artifacts = {}
        for domain in ('base', 'easy', 'hard'):
            row = summary.get('domains', {}).get(domain, {})
            mse = row.get('raw_mse')
            require(row.get('status') == 'ok' and row.get('checkpoint_id') == checkpoint_id
                    and row.get('domain') == domain and row.get('split_role') == 'final_test'
                    and isinstance(mse, (float, int)) and math.isfinite(mse),
                    f'Final test incomplete: {run_id}/{checkpoint_id}/{domain}')
            relative = Path(row.get('episode_metrics_path', ''))
            require(not relative.is_absolute() and '..' not in relative.parts
                    and relative.parts[:3] == ('runs', run_id, 'episode_metrics')
                    and relative.suffix == '.npz', 'Unsafe final raw metrics path')
            artifact = project / relative
            require(artifact.is_file() and not artifact.is_symlink() and artifact.stat().st_size > 0,
                    f'Missing final raw metrics: {artifact}')
            artifacts[domain] = {'path': str(relative), 'sha256': digest(artifact)}
        evidence[checkpoint_id] = {'summary_sha256': digest(summary_path), 'artifacts': artifacts}
    return evidence


def validate_replacement(lease_name, worker, expected, leases):
    original = leases[lease_name]
    require(original['run_ids'] == expected, f'Original run IDs changed: {lease_name}')
    path = CONTROL / (worker + '.json')
    replacement = read(path)
    require(replacement.get('returncode') == 0, f'Replacement wrapper has not exited successfully: {worker}')
    require(replacement['original_lease'] == lease_name and replacement['replacement_worker'] == worker
            and replacement['run_ids'] == expected, 'Replacement provenance mismatch')
    command = list(original['command'])
    command[command.index('--worker') + 1] = worker
    require(replacement['command'] == command, 'Replacement command differs beyond worker name')
    rental_path = ROOT / 'workers' / worker / 'rental_history.json'
    rental = read(rental_path)
    require(rental.get('destroyed_at') and rental.get('instance_id'), 'Replacement rental not destroyed')
    if replacement.get('instance_id') is not None:
        require(str(replacement['instance_id']) == str(rental['instance_id']), 'Replacement instance mismatch')
    require(not (ROOT / 'workers' / lease_name / 'rental_history.json').exists(),
            'Original attempt created an instance; recovery assumptions invalid')
    projects = list((ROOT / 'workers' / worker / 'runs').glob('*/outputs/project'))
    require(len(projects) == 1, 'Expected exactly one replacement output directory')
    project = projects[0]
    require(read(project.parent.parent / 'sync-status.json').get('final_sync_complete') is True,
            'Replacement final sync incomplete')
    group = read(project / 'group_status.json')
    require(group.get('status') == 'complete' and not group.get('failures')
            and group.get('evaluate') is True and sorted(group.get('runs', [])) == sorted(expected)
            and sorted(group.get('completed_runs', [])) == sorted(expected), 'Replacement group incomplete')
    evidence = {run_id: validate_run(project, run_id) for run_id in expected}
    if original.get('status') == 'complete':
        require(original.get('returncode') == 0 and original.get('recovery', {}).get('replacement_worker') == worker,
                'Original lease already completed by another path')
    else:
        require(original.get('status') == 'failed' and original.get('returncode') != 0,
                'Original controller has not finalized failed attempt')
    record = {'reconciled_at': time.time(), 'replacement_worker': worker,
              'replacement_record': str(path), 'replacement_record_sha256': digest(path),
              'replacement': replacement, 'replacement_rental': rental,
              'replacement_rental_sha256': digest(rental_path), 'output_project': str(project),
              'validated_archive_checkpoint_count': 169,
              'group_status_sha256': digest(project / 'group_status.json'), 'validated_runs': evidence,
              'original_failed_lease': copy.deepcopy(original), 'helper_sha256': digest(Path(__file__))}
    return project, evidence, record


def main():
    with (ROOT / 'queue/controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not list((ROOT / 'workers').glob('*/active.json')), 'Active rental records remain')
        state = read(ROOT / 'queue/state.json')
        leases = state['phases']['tests']['leases']
        require(set(leases) == {'tests-0', 'tests-1', 'tests-2'}, 'Unexpected test leases')
        sibling = leases['tests-1']
        require(sibling.get('status') == 'complete' and sibling.get('returncode') == 0,
                'Sibling tests-1 lease incomplete')
        require(read(ROOT / 'workers/tests-1/rental_history.json').get('destroyed_at'),
                'Sibling tests-1 rental not destroyed')
        for relative, expected_hash in state['source_files'].items():
            require(digest(ROOT / relative) == expected_hash, f'Frozen source changed: {relative}')
        prepared = {name: validate_replacement(name, worker, expected, leases)
                    for name, (worker, expected) in GROUPS.items()}
        all_runs = sibling['run_ids'] + [r for _, runs in GROUPS.values() for r in runs]
        require(len(all_runs) == len(set(all_runs)) == 21, 'Tests must cover 21 distinct archived runs')
        sibling_evidence = {r: validate_run(ROOT, r) for r in sibling['run_ids']}
        count = sum(len(v) for v in sibling_evidence.values())
        count += sum(len(v) for _, evidence, _ in prepared.values() for v in evidence.values())
        require(count == 169, 'All 169 retained checkpoints must have complete final evaluations')
        completed = [leases[name]['status'] == 'complete' for name in GROUPS]
        if any(completed):
            require(all(completed) and state['phases']['tests'].get('status') == 'complete',
                    'Partial reconciliation state requires manual investigation')
            print('Already reconciled; no changes.'); return
        sys.dont_write_bytecode = True
        spec = importlib.util.spec_from_file_location('frozen_campaign_control', ROOT / 'scripts/run_campaign.py')
        campaign = importlib.util.module_from_spec(spec); spec.loader.exec_module(campaign)
        # Imports are retry-safe. Neither lease is marked recovered until both
        # imports and the complete canonical 169-checkpoint verification pass.
        for name, (project, _, _) in prepared.items():
            campaign.copy_result(project, GROUPS[name][1])
        for run_id in all_runs:
            validate_run(ROOT, run_id)
        for name, (_, _, record) in prepared.items():
            original = leases[name]
            before = copy.deepcopy(original)
            original.update(status='complete', returncode=0, finished=time.time(), recovery=record)
            original.setdefault('recovery_attempts', []).append(before)
        state['phases']['tests'].update(status='complete', reconciled_at=time.time())
        # One atomic queue publication covers both recovered leases and phase.
        campaign.atomic(ROOT / 'queue/state.json', state)
        print(json.dumps({'status': 'reconciled', 'replacement_workers': [v[0] for v in GROUPS.values()],
                          'validated_runs': 21, 'validated_checkpoints': 169}))


if __name__ == '__main__':
    main()
