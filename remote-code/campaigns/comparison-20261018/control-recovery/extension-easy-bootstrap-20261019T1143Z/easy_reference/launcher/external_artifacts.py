"""Control-plane orchestration of verified private persistent artifact exports.

No artifact bytes are processed on the control plane. Hashing and staging run
on the GPU worker; provider credentials remain here. Call only for immutable
terminal outputs. A single durable coordinator can call this for every run.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

from controller_handoff import atomic_json

sys.path.insert(0, '/home/jump/vast-tools')
import smoke

CAMPAIGN = Path('/home/jump/deep-hedging/campaigns/comparison-20261018')
VOLUME_ID = 56629013
MACHINE_ID = 149899
LABEL = 'deep_hedging_20261018_artifacts'
CAPACITY_BYTES = 110_000_000_000  # Leave ten GB for manifests and headroom.


def selected_rows(rows, minimum_bytes=64 * 1024 * 1024):
    return [r for r in rows if r['path'].startswith('histories/')
            and r['path'].endswith('.tar.gz') and r['bytes'] >= minimum_bytes]


def volume_owned():
    volumes = smoke.api('GET', 'volumes?owner=me&type=all_volume')['volumes']
    volume = next((v for v in volumes if v['id'] == VOLUME_ID), None)
    if (volume is None or volume.get('machine_id') != MACHINE_ID
            or volume.get('label') != LABEL or volume.get('disk_space', 0) < 120
            or volume.get('status') != 'created'
            or volume.get('end_date', 0) <= time.time() + 86400):
        raise RuntimeError('Private artifact volume ownership/availability check failed')
    return volume


def verified_external_rows(run, remote_rows):
    """Return only exact remote inventory rows backed by a verified readback."""
    path = Path(run) / 'external-artifacts.json'
    if not path.exists():
        return []
    record = json.loads(path.read_text())
    if record.get('status') != 'verified':
        raise RuntimeError('External artifact record is not verified')
    metadata = json.loads((Path(run) / 'metadata.json').read_text())
    if (record.get('volume_id') != VOLUME_ID or record.get('run_id') != Path(run).name
            or record.get('instance_id') != metadata['instance']['instance_id']):
        raise RuntimeError('External artifact run identity mismatch')
    proof_path = Path(run) / 'external-readback.json'
    proof_bytes = proof_path.read_bytes()
    if hashlib.sha256(proof_bytes).hexdigest() != record['readback_proof_sha256']:
        raise RuntimeError('External readback proof digest mismatch')
    proof = json.loads(proof_bytes)
    if not proof.get('verified') or proof.get('rows') != record['rows']:
        raise RuntimeError('External readback proof inventory mismatch')
    indexed = {r['path']: r for r in remote_rows}
    for row in record['rows']:
        actual = indexed.get(row['path'])
        if actual is None or any(actual[k] != row[k] for k in ('bytes', 'sha256')):
            raise RuntimeError('External artifact changed since verified upload')
    volume_owned()
    return record['rows']


def rsync_excludes(run):
    """Caller must verify this record against current remote inventory first."""
    path = Path(run) / 'external-artifacts.json'
    if not path.exists():
        return []
    record = json.loads(path.read_text())
    if record.get('status') != 'verified':
        return []
    return ['--exclude=/' + row['path'] for row in record['rows']]


def reserve_capacity(key, rows):
    path = CAMPAIGN / 'external-capacity.json'
    ledger = json.loads(path.read_text()) if path.exists() else {}
    size = sum(row['bytes'] for row in rows)
    if key in ledger and ledger[key]['bytes'] != size:
        raise RuntimeError('External capacity reservation changed')
    other = sum(value['bytes'] for name, value in ledger.items() if name != key)
    if other + size > CAPACITY_BYTES:
        raise RuntimeError('Private artifact volume capacity reservation exceeded')
    ledger[key] = dict(bytes=size, volume_id=VOLUME_ID, rows=rows)
    atomic_json(path, ledger)


def worker(ssh, request):
    source = Path(__file__).with_name('external_artifact_worker.py').read_bytes()
    command = 'python3 - ' + shlex.quote(json.dumps(request, sort_keys=True))
    result = subprocess.run(ssh + [command], input=source, capture_output=True,
                            timeout=300, check=True)
    return json.loads(result.stdout)


def provider_copy(source_id, source_path, destination_id, destination_path):
    return smoke.api('PUT', 'commands/copy_direct/', dict(
        client_id='me', src_id=source_id, src_path=source_path,
        dst_id=destination_id, dst_path=destination_path))


def request_readback(state, request, state_path, instance_id):
    """A crash always leaves explicit intent or a durably acknowledged call."""
    state['phase'] = 'readback_intent'
    atomic_json(state_path, state)
    state['readback_requests'] = state.get('readback_requests', 0) + 1
    if state['readback_requests'] > 2:
        raise RuntimeError('External readback request recovery limit reached')
    if state['readback_requests'] > 1:
        smoke.api('DELETE', 'commands/copy_direct/', dict(
            client_id='me', dst_id='C.' + str(instance_id)))
        state['readback'] = '/workspace/external-artifact-readback-' + uuid.uuid4().hex
        request['readback'] = state['readback']
    atomic_json(state_path, state)
    provider_copy('V.' + str(VOLUME_ID), state['volume_path'],
                  'C.' + str(instance_id), state['readback'] + '/')
    state['phase'] = 'readback_acked'
    atomic_json(state_path, state)


def ensure_external(run, ssh, rows, *, fixture=False, timeout_seconds=600):
    """Persist stage transitions and verify readback before permitting exclusion.

    Production rows must be terminal immutable histories. Fixture mode admits
    only tiny storage-proof paths, for the explicitly authorized worker audit.
    """
    run = Path(run)
    if fixture:
        if any(not r['path'].startswith('storage-proof/') or r['bytes'] > 4096 for r in rows):
            raise RuntimeError('Fixture mode only admits tiny storage-control files')
    elif rows != selected_rows(rows):
        raise RuntimeError('Only selected immutable history archives may be externalized')
    if not rows:
        return []
    with (CAMPAIGN / 'external-volume.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        volume_owned()
        if (run / 'external-artifacts.json').exists():
            return verified_external_rows(run, rows)
        metadata = json.loads((run / 'metadata.json').read_text())
        instance_id = metadata['instance']['instance_id']
        state_path = run / 'external-export-state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        if state is None:
            nonce = uuid.uuid4().hex
            objects = [dict(row, object_name=f'{index:04d}-{Path(row["path"]).name}')
                       for index, row in enumerate(rows)]
            state = dict(run_id=run.name, instance_id=instance_id, rows=objects,
                         stage='/workspace/external-artifact-stage-' + nonce,
                         readback='/workspace/external-artifact-readback-' + nonce,
                         volume_path='/run-' + str(instance_id) + '-' + nonce + '/',
                         phase='prepare', started=time.time(), attempt=1,
                         attempt_deadline=time.time() + timeout_seconds)
            atomic_json(state_path, state)
        if [{k: row[k] for k in ('path', 'bytes', 'sha256')} for row in state['rows']] != rows:
            raise RuntimeError('Immutable external export inventory changed')
        reserve_capacity(str(instance_id) + ':' + run.name, state['rows'])
        if 'transfer_cost_quote' not in state:
            instance = next((v for v in smoke.list_instances() if v['id'] == instance_id), {})
            quoted = {key: instance.get(key) for key in ('inet_up_cost', 'inet_down_cost')}
            total_bytes = sum(row['bytes'] for row in state['rows'])
            known = all(isinstance(quoted[key], (int, float)) for key in quoted)
            state['transfer_cost_quote'] = dict(
                bytes_each_direction=total_bytes, source_instance_fields=quoted,
                assumed_units='USD per GB; one outbound upload plus one inbound full readback',
                source_cost_estimate_usd=(total_bytes / 1e9 * sum(quoted.values()) if known else None),
                volume_host_bandwidth_cost='Not returned by owned-volume API; additional provider charges may apply')
            atomic_json(state_path, state)
        request = dict(rows=state['rows'], stage=state['stage'], readback=state['readback'])
        # Two bounded transport attempts share an overall budget. This resets
        # neither the scientific run nor its runtime accounting.
        overall_deadline = state['started'] + 2 * timeout_seconds
        if time.time() >= overall_deadline:
            raise TimeoutError('External export exhausted its overall recovery budget')
        if time.time() >= state['attempt_deadline']:
            if state['attempt'] >= 2:
                raise TimeoutError('External export exhausted two bounded attempts')
            state['attempt'] += 1
            state['attempt_deadline'] = min(time.time() + timeout_seconds, overall_deadline)
            state['phase'] = ('prepare' if state['phase'] == 'prepare' else
                              'readback_intent' if state['phase'].startswith('readback')
                              else 'upload_intent')
            atomic_json(state_path, state)
        deadline = min(state['attempt_deadline'], overall_deadline)
        if state['phase'] == 'prepare':
            prepared = worker(ssh, dict(request, action='prepare'))
            if not prepared.get('prepared'):
                raise RuntimeError('Worker did not verify immutable hardlink staging')
            state['phase'] = 'upload_intent'
            atomic_json(state_path, state)
        if state['phase'] == 'upload_intent':
            # An intent may mean a crash before OR after the provider call.
            # Wait for any prior upload, then reissue the same immutable object
            # set idempotently; no API acceptance is treated as completion.
            if state.get('upload_requests', 0):
                time.sleep(max(0, 10 - (time.time() - state['upload_request_time'])))
            while time.time() < deadline:
                item = next((v for v in smoke.list_instances() if v['id'] == instance_id), {})
                if item.get('status_msg') != 'Copying container data...':
                    break
                time.sleep(5)
            else:
                raise TimeoutError('Prior provider upload remains active; source preserved')
            state['upload_requests'] = state.get('upload_requests', 0) + 1
            if state['upload_requests'] > 2:
                raise RuntimeError('External upload request recovery limit reached')
            state['upload_request_time'] = time.time()
            atomic_json(state_path, state)
            provider_copy('C.' + str(instance_id), state['stage'] + '/',
                          'V.' + str(VOLUME_ID), state['volume_path'])
            state['phase'] = 'upload_acked'
            atomic_json(state_path, state)
        if state['phase'] == 'upload_acked':
            while time.time() < deadline:
                item = next((v for v in smoke.list_instances() if v['id'] == instance_id), {})
                message = item.get('status_msg', '')
                if message == 'Done copying' and time.time() - state['upload_request_time'] >= 10:
                    break
                if 'rsync error' in message or 'Exception:' in message:
                    raise RuntimeError('Provider upload failed: ' + message[:500])
                time.sleep(5)
            else:
                raise TimeoutError('Bounded external upload did not complete')
            state['phase'] = 'readback_intent'
            atomic_json(state_path, state)
        if state['phase'] == 'readback_intent':
            request_readback(state, request, state_path, instance_id)
        while time.time() < deadline:
            proof = worker(ssh, dict(request, action='verify'))
            if proof.get('stale_readback'):
                request_readback(state, request, state_path, instance_id)
                time.sleep(5)
                continue
            if proof.get('verified'):
                atomic_json(run / 'external-readback.json', proof)
                record = dict(status='verified', volume_id=VOLUME_ID, machine_id=MACHINE_ID,
                              run_id=run.name, instance_id=instance_id, rows=state['rows'],
                              volume_path=state['volume_path'],
                              transfer_cost_quote=state['transfer_cost_quote'],
                              verified_utc=datetime.now(timezone.utc).isoformat(),
                              readback_proof_sha256=hashlib.sha256((run / 'external-readback.json').read_bytes()).hexdigest())
                atomic_json(run / 'external-artifacts.json', record)
                state['phase'] = 'verified'
                atomic_json(state_path, state)
                return verified_external_rows(run, rows)
            time.sleep(5)
        raise TimeoutError('Bounded external readback did not verify; preserve source worker disk')
