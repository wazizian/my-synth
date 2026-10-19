"""Bounded transfer of a live remote run to the durable recovery service."""
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import time


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def request_handoff(run_dir, ssh, args):
    try:
        return _request_handoff(run_dir, ssh, args)
    except Exception as error:
        atomic_json(run_dir / 'original-handoff-error.json', {
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'type': type(error).__name__, 'error': str(error),
            'context': 'Same-worker handoff failed; no relaunch attempted',
        })
        raise


def _request_handoff(run_dir, ssh, args):
    metadata = json.loads((run_dir / 'metadata.json').read_text())
    # Reconnect only to this worker. A terminal marker always takes precedence.
    probe_deadline = time.monotonic() + 60
    observations = []
    supervisor = None
    command = ("if test -f /workspace/job/outputs/workload.exit; then "
               "printf 'EXIT '; cat /workspace/job/outputs/workload.exit; "
               "elif pgrep -f '^bash supervise-workload.sh$' >/dev/null; then "
               "echo LIVE; else echo MISSING; fi")
    while time.monotonic() < probe_deadline:
        observation = {'created_utc': datetime.now(timezone.utc).isoformat()}
        try:
            result = subprocess.run(ssh + [command], capture_output=True, text=True,
                                    timeout=min(20, max(.01, probe_deadline - time.monotonic())))
            state = result.stdout.strip()
            observation.update(returncode=result.returncode, stdout=state[:2048],
                               stderr=result.stderr[-2048:])
        except (subprocess.TimeoutExpired, OSError) as error:
            observation.update(error_type=type(error).__name__, error=str(error)[:2048])
            state = None
        observations.append(observation)
        atomic_json(run_dir / 'handoff.probes.json', {'observations': observations})
        if state is not None and result.returncode == 0:
            if state.startswith('EXIT '):
                raise RuntimeError('Remote workload is terminal: ' + state)
            if state == 'LIVE':
                supervisor = state
                break
            if state != 'MISSING':
                raise RuntimeError('Invalid remote handoff probe state: ' + repr(state))
        time.sleep(min(2, max(0, probe_deadline - time.monotonic())))
    if supervisor is None:
        raise RuntimeError('Detached supervisor not confirmed within 60 seconds')
    request = dict(run_id=run_dir.name,
                   instance_id=metadata['instance']['instance_id'],
                   label=metadata['instance']['label'], ssh=ssh,
                   required_artifacts=args.require_artifact,
                   deadline_utc=args.handoff_deadline_utc,
                   supervisor=supervisor,
                   source_sha256=hashlib.sha256((run_dir / 'submitted-job.py').read_bytes()).hexdigest(),
                   created_utc=datetime.now(timezone.utc).isoformat())
    request_path = run_dir / 'handoff.request.json'
    atomic_json(request_path, request)
    digest = hashlib.sha256(request_path.read_bytes()).hexdigest()
    deadline = time.monotonic() + 180
    ack_path = run_dir / 'handoff.ack.json'
    while time.monotonic() < deadline:
        if ack_path.exists():
            ack = json.loads(ack_path.read_text())
            if (ack.get('request_sha256') == digest and ack.get('owner') == 'dh-credit-recovery'
                    and ack.get('instance_id') == request['instance_id']
                    and ack.get('run_id') == request['run_id']
                    and ack.get('deadline_utc') == request['deadline_utc']
                    and ack.get('required_artifacts') == request['required_artifacts']):
                atomic_json(run_dir / 'handoff.accepted.json', ack)
                return
            raise RuntimeError('Recovery ownership ACK does not match this run')
        time.sleep(2)
    raise RuntimeError('Durable recovery ownership ACK timed out')
