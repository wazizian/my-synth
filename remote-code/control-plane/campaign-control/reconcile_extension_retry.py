#!/usr/bin/env python3
"""Offline metadata reconciliation of the authorized extensions-0 replacement.

Run only after the original controller and replacement wrapper have exited:
  python3 /home/jump/campaign-control/reconcile_extension_retry.py
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
EXPECTED = ['baseline-lr001', 'soar_hard-lr001', 'soar_mixed-lr001']
WORKER = 'extensions-retry0'


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
    status = read(directory / 'status.json')
    active = status.get('active_training_seconds', 0)
    require(status.get('status') == 'completed' and isinstance(active, (float, int))
            and math.isfinite(active) and active >= 2700, f'Incomplete terminal run: {run_id}')
    require(status.get('run_id') == run_id, f'Wrong run identity: {run_id}')
    require(read(directory / 'config.json') == read(ROOT / 'runs' / run_id / 'config.json'),
            f'Immutable config changed: {run_id}')
    checkpoints = directory / 'checkpoints'
    evidence = {}
    for name in ('best', 'latest', 'final'):
        pointer = read(checkpoints / (name + '.json'))
        relative = Path(pointer['path'])
        require(not relative.is_absolute() and len(relative.parts) == 1
                and relative.name not in ('.', '..'), 'Unsafe checkpoint path')
        bundle = checkpoints / relative
        require(not bundle.is_symlink(), 'Symlinked checkpoint')
        for filename, key in [('state.eqx', 'state_sha256'), ('metadata.json', 'metadata_sha256')]:
            path = bundle / filename
            require(path.is_file() and not path.is_symlink() and digest(path) == pointer[key],
                    f'Checkpoint missing or hash mismatch: {path}')
        metadata = read(bundle / 'metadata.json')
        require(metadata['checkpoint_id'] == pointer['checkpoint_id']
                and metadata['run_id'] == run_id, 'Checkpoint identity mismatch')
        if name in ('final', 'latest'):
            require(metadata['counters']['active_training_seconds'] >= 2700,
                    f'Terminal checkpoint is below cap: {run_id}/{name}')
        evidence[name] = pointer
    require(evidence['latest']['checkpoint_id'] == evidence['final']['checkpoint_id']
            == status['last_checkpoint_id'], 'Terminal pointer/status mismatch')
    return {'active_training_seconds': active, 'checkpoints': evidence}


def main():
    with (ROOT / 'queue/controller.lock').open('a') as lock:
        # Original controller holds this flock for its full lifetime. No waiting:
        # calling early must fail rather than race an active controller.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not list((ROOT / 'workers').glob('*/active.json')), 'Active rental records remain')
        state = read(ROOT / 'queue/state.json')
        leases = state['phases']['extensions']['leases']
        require(set(leases) == {'extensions-0', 'extensions-1', 'extensions-2'}, 'Unexpected extension leases')
        for name in ('extensions-1', 'extensions-2'):
            require(leases[name]['status'] == 'complete' and leases[name].get('returncode') == 0,
                    f'Other extension lease incomplete: {name}')
        original = leases['extensions-0']
        require(original['run_ids'] == EXPECTED, 'Original run IDs changed')
        replacement_path = CONTROL / 'extensions-retry0.json'
        replacement = read(replacement_path)
        require(replacement.get('returncode') == 0, 'Replacement wrapper has not exited successfully')
        require(replacement['original_lease'] == 'extensions-0'
                and replacement['replacement_worker'] == WORKER
                and replacement['run_ids'] == EXPECTED, 'Replacement provenance mismatch')
        expected_command = list(original['command'])
        expected_command[expected_command.index('--worker') + 1] = WORKER
        require(replacement['command'] == expected_command, 'Replacement command differs beyond worker name')
        for worker in ('extensions-1', 'extensions-2', WORKER):
            history = read(ROOT / 'workers' / worker / 'rental_history.json')
            require(history.get('destroyed_at'), f'Missing rental destruction record: {worker}')
            if worker == WORKER:
                require(str(history.get('instance_id')) == '56469209', 'Unexpected replacement instance')
        require(not (ROOT / 'workers/extensions-0/rental_history.json').exists(),
                'Original attempt created an instance; recovery assumptions invalid')
        projects = list((ROOT / 'workers' / WORKER / 'runs').glob('*/outputs/project'))
        require(len(projects) == 1, 'Expected exactly one replacement output directory')
        project = projects[0]
        sync = read(project.parent.parent / 'sync-status.json')
        require(sync.get('final_sync_complete') is True, 'Replacement final sync incomplete')
        group = read(project / 'group_status.json')
        require(group.get('status') == 'complete' and not group.get('failures')
                and group.get('evaluate') is False and sorted(group.get('runs', [])) == sorted(EXPECTED)
                and sorted(group.get('completed_runs', [])) == sorted(EXPECTED), 'Replacement group incomplete')
        for relative, expected_hash in state['source_files'].items():
            require(digest(ROOT / relative) == expected_hash, f'Frozen source changed: {relative}')
        evidence = {run_id: validate_run(project, run_id) for run_id in EXPECTED}
        if original.get('status') == 'complete':
            require(original.get('recovery', {}).get('replacement_worker') == WORKER,
                    'Original lease already completed by another path')
            require(state['phases']['extensions'].get('status') == 'complete', 'Partial reconciliation state')
            print('Already reconciled; no changes.'); return
        require(original.get('status') == 'failed' and original.get('returncode') != 0,
                'Original controller has not finalized failed attempt')
        # Import the frozen stdlib-only control module; do not initialize JAX.
        sys.dont_write_bytecode = True
        spec = importlib.util.spec_from_file_location('frozen_campaign_control', ROOT / 'scripts/run_campaign.py')
        campaign = importlib.util.module_from_spec(spec); spec.loader.exec_module(campaign)
        before = copy.deepcopy(original)
        record = {'reconciled_at': time.time(), 'replacement_worker': WORKER,
                  'replacement_record': str(replacement_path), 'replacement_record_sha256': digest(replacement_path),
                  'replacement': replacement, 'output_project': str(project),
                  'group_status_sha256': digest(project / 'group_status.json'), 'validated_runs': evidence,
                  'original_failed_lease': before, 'helper_sha256': digest(Path(__file__))}
        # Partial import is retry-safe; queue completion is published only last.
        campaign.copy_result(project, EXPECTED)
        for run_id in EXPECTED:
            validate_run(ROOT, run_id)
        original.update(status='complete', returncode=0, finished=time.time(), recovery=record)
        original.setdefault('recovery_attempts', []).append(before)
        state['phases']['extensions'].update(status='complete', reconciled_at=time.time())
        campaign.atomic(ROOT / 'queue/state.json', state)
        print(json.dumps({'status': 'reconciled', 'replacement_worker': WORKER, 'runs': EXPECTED}))


if __name__ == '__main__':
    main()
