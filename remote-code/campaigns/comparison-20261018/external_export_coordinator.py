"""Single durable external-export owner; independent of receiver ACK scans."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from controller_handoff import atomic_json
from external_artifacts import ensure_external

sys.path.insert(0, '/home/jump/vast-tools')
import smoke

ROOT = Path(__file__).resolve().parent


def retryable(error):
    if isinstance(error, (TimeoutError, subprocess.TimeoutExpired, OSError)):
        return True
    if isinstance(error, subprocess.CalledProcessError):
        return error.returncode == 255
    text = str(error)
    return any(fragment in text for fragment in (
        'network/DNS unavailable', 'HTTP 429', 'HTTP 500', 'HTTP 502', 'HTTP 503', 'HTTP 504'))


def stop_and_preserve(run, metadata, error):
    saved = metadata['instance']
    matching = next((v for v in smoke.list_instances()
                     if v['id'] == saved['instance_id'] and v.get('label') == saved['label']), None)
    stop_requested = matching is not None and matching.get('actual_status') != 'stopped'
    if stop_requested:
        smoke.api('PUT', f"instances/{saved['instance_id']}/", dict(state='stopped'))
    deadline = time.monotonic() + 60
    while matching is not None and matching.get('actual_status') != 'stopped' and time.monotonic() < deadline:
        time.sleep(5)
        matching = next((v for v in smoke.list_instances()
                         if v['id'] == saved['instance_id'] and v.get('label') == saved['label']), None)
    atomic_json(run / 'external-export-failure.json', dict(
        instance_id=saved['instance_id'], label=saved['label'], error=repr(error),
        stopped_compute_requested=stop_requested, deletion_not_requested=True,
        provider_instance_present=matching is not None,
        source_disk_preserved_confirmed=matching is not None and matching.get('actual_status') == 'stopped',
        compute_stop_confirmed=matching is None or matching.get('actual_status') == 'stopped',
        storage_charges_continue=True, utc=datetime.now(timezone.utc).isoformat()))


def main():
    lock = (ROOT / 'external-export-coordinator.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        for request_path in ROOT.rglob('external-export-request.json'):
            run = request_path.parent
            if (run / 'external-artifacts.json').exists() or (run / 'external-export-failure.json').exists():
                continue
            metadata = None
            terminal_verified = False
            control_path = run / 'external-coordinator-state.json'
            control = dict(attempts_started=0)
            try:
                if control_path.exists():
                    control = json.loads(control_path.read_text())
                if time.time() < control.get('next_retry_time', 0):
                    continue
                metadata = json.loads((run / 'metadata.json').read_text())
                state_path = run.parent.parent / 'active.json'
                active = json.loads(state_path.read_text())
                saved = metadata['instance']
                if any(active.get(key) != saved.get(key) for key in ('instance_id', 'label')):
                    raise RuntimeError('External export active-run identity mismatch')
                request = json.loads(request_path.read_text())
                if request['instance_id'] != saved['instance_id']:
                    raise RuntimeError('External export request identity mismatch')
                terminal = subprocess.run(metadata['ssh'] + ['cat /workspace/job/outputs/workload.exit'],
                                          capture_output=True, text=True, timeout=30, check=True)
                if terminal.stdout.strip() != '0':
                    raise RuntimeError('External export requires successful immutable terminal outputs')
                terminal_verified = True
                if control['attempts_started'] >= 2:
                    raise RuntimeError('Coordinator external transport recovery exhausted')
                control['attempts_started'] += 1
                control['status'] = 'exporting'
                atomic_json(control_path, control)
                ensure_external(run, metadata['ssh'], request['rows'])
                control['status'] = 'verified'
                atomic_json(control_path, control)
                print('External export verified:', run, flush=True)
            except BlockingIOError:
                if terminal_verified:
                    control['attempts_started'] -= 1
                    control['status'] = 'waiting_for_volume_lock'
                    atomic_json(control_path, control)
                continue
            except Exception as exc:
                print('External export failed:', run, repr(exc), flush=True)
                if metadata is not None and terminal_verified:
                    export_state_path = run / 'external-export-state.json'
                    export_state = json.loads(export_state_path.read_text()) if export_state_path.exists() else {}
                    started = export_state.get('started', time.time())
                    if retryable(exc) and control['attempts_started'] < 2 and time.time() < started + 1200:
                        control.update(status='bounded_transport_retry_pending', last_error=repr(exc),
                                       next_retry_time=time.time() + 10)
                        atomic_json(control_path, control)
                        continue
                    control.update(status='failed_preserve_source', last_error=repr(exc))
                    atomic_json(control_path, control)
                    try:
                        stop_and_preserve(run, metadata, exc)
                    except Exception as stop_error:
                        atomic_json(run / 'external-export-stop-error.json', dict(
                            error=repr(exc), stop_error=repr(stop_error), time=time.time()))
                else:
                    atomic_json(run / 'external-export-refused.json', dict(
                        error=repr(exc), productive_worker_not_stopped=True, time=time.time()))
        time.sleep(10)


if __name__ == '__main__':
    main()
