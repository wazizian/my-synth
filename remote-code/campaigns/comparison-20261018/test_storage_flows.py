"""GPU-worker-only execution of actual receiver/watchdog command flows."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading as real_threading
import time as real_time
from types import ModuleType, SimpleNamespace


class StopLoop(BaseException):
    pass


def run_flow(source, root, mode, verified=False):
    root.mkdir()
    launch = root / 'final' / 'selfplay' / 'launcher'
    run = launch / 'runs' / '20261018T000000Z-fixture'
    out = run / 'outputs'
    out.mkdir(parents=True)
    instance = dict(instance_id=123, label='fixture')
    def save(path, obj):
        path.write_text(json.dumps(obj))
    save(launch / 'active.json', instance)
    save(run / 'metadata.json', dict(instance=instance, ssh=['SSH'], timeout_seconds=1,
                                   required_artifacts=['complete.json', 'histories/lr-0.001.tar.gz']))
    save(run / 'handoff.request.json', {})
    save(run / 'sync-status.json', dict(successes=0, failures=0, final_sync_complete=False))
    save(root / 'credit-instance-ledger.json', {})
    (out / 'complete.json').write_bytes(b'done')
    archive = dict(path='histories/lr-0.001.tar.gz', bytes=100000000, sha256='a' * 64)
    rows = [dict(path='complete.json', bytes=4, sha256=hashlib.sha256(b'done').hexdigest()), archive]
    if verified:
        save(run / 'external-artifacts.json', {})
    calls = []
    external = ModuleType('external_artifacts')
    external.selected_rows = lambda rows: [archive]
    external.verified_external_rows = lambda run, rows: [archive] if verified else []
    external.rsync_excludes = lambda run: ['--exclude=/histories/lr-0.001.tar.gz'] if verified else []
    fake_time = ModuleType('time')
    fake_time.__dict__.update(vars(real_time))
    fake_time.time = real_time.time
    fake_time.sleep = lambda seconds: (_ for _ in ()).throw(StopLoop())
    fake_threading = ModuleType('threading')
    fake_threading.__dict__.update(vars(real_threading))
    fake_threading.Thread = lambda *a, **k: SimpleNamespace(start=lambda: None)
    fake_subprocess = ModuleType('subprocess')
    def command(command, **kwargs):
        calls.append(command)
        if command[0] == 'SSH':
            if 'python3 -c ' in command[-1]:
                return SimpleNamespace(returncode=0, stdout=json.dumps(rows))
            return SimpleNamespace(returncode=0, stdout='RUNNING' if mode == 'RUNNING' else '0')
        if '--cleanup' in command:
            assert (run / 'fixture-verify-called.json').exists(), 'Cleanup preceded verifier'
            (launch / 'active.json').unlink()
        return SimpleNamespace(returncode=0)
    fake_subprocess.run = command
    fake_provider = ModuleType('smoke')
    actual = dict(id=123, label='fixture', actual_status='running')
    def api(method, path, body=None):
        calls.append([method, path, body])
        if method == 'DELETE':
            actual['actual_status'] = 'destroyed'
        if method == 'PUT':
            actual['actual_status'] = 'stopped'
    fake_provider.api = api
    fake_provider.list_instances = lambda: [] if actual['actual_status'] == 'destroyed' else [actual]
    (launch / 'export_safety.py').write_text(
        "import json\nfrom pathlib import Path\ndef verify_export(run,ssh,required):\n"
        " (run/'fixture-verify-called.json').write_text('{}')\n"
        " (run/'export-verified-manifest.json').write_text(" + repr(json.dumps(rows)) + ")\n")
    modules = dict(external_artifacts=external, time=fake_time, threading=fake_threading,
                   subprocess=fake_subprocess, smoke=fake_provider)
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        exec(compile(source.read_text(), str(source), 'exec'), dict(__file__=str(root / source.name), __name__='flow_fixture'))
    except StopLoop:
        pass
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    rsync = [command for command in calls if command[0] == 'rsync']
    deletes = [command for command in calls if command[0] == 'DELETE' or '--cleanup' in command]
    if source.name.startswith('recover'):
        if mode == 'RUNNING':
            assert rsync and '--exclude=/histories/*.tar.gz' in rsync[0]
            assert '--exclude=*.tmp*' in rsync[0] and not deletes
            assert json.loads((run / 'pending-deferred-artifacts.json').read_text())['final_export_complete'] is False
        elif not verified:
            assert (run / 'external-export-request.json').exists() and not rsync and not deletes
            assert json.loads((run / 'sync-status.json').read_text())['final_sync_complete'] is False
        else:
            assert rsync and '--exclude=/histories/lr-0.001.tar.gz' in rsync[0] and deletes
            assert json.loads((run / 'recovery-status.json').read_text())['complete_artifacts'] is True
    elif verified:
        assert deletes and rsync and '--exclude=/histories/lr-0.001.tar.gz' in rsync[0]
    else:
        assert not deletes and actual['actual_status'] == 'stopped'
        assert (run / 'external-export-request.json').exists()


def main():
    source = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_flow(source / 'recover_credit.storage_candidate.py', root / 'running', 'RUNNING')
        run_flow(source / 'recover_credit.storage_candidate.py', root / 'pending', '0')
        run_flow(source / 'recover_credit.storage_candidate.py', root / 'verified', '0', True)
        run_flow(source / 'watchdog.storage_candidate.py', root / 'deadline-pending', '0')
        run_flow(source / 'watchdog.storage_candidate.py', root / 'deadline-verified', '0', True)
    print('receiver/watchdog actual flow checks passed: live deferred small sync, '
          'terminal queue without cleanup, exact receipt final sync, deadline pending stop, verified delete', flush=True)


if __name__ == '__main__':
    main()
