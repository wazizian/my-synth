"""GPU-worker-only cleanup safety checks for externally retained artifacts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace

provider = ModuleType('smoke')
sys.modules['smoke'] = provider
import external_artifacts as external

spec = importlib.util.spec_from_file_location('candidate_safety', Path(__file__).with_name('export_safety.storage_candidate.py'))
safety = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safety)


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2))


def main():
    volume_available = True
    def api(method, path, body=None):
        return dict(volumes=[dict(id=external.VOLUME_ID, machine_id=external.MACHINE_ID,
                                 label=external.LABEL, disk_space=120, status='created',
                                 end_date=time.time() + 100000)] if volume_available else [])
    provider.api = api
    with tempfile.TemporaryDirectory() as temporary:
        launch = Path(temporary)
        run = launch / 'runs' / 'run'
        out = run / 'outputs'
        out.mkdir(parents=True)
        instance = dict(instance_id=123, label='fixture')
        save(launch / 'active.json', instance)
        save(run / 'metadata.json', dict(instance=instance))
        save(run / 'sync-status.json', dict(final_sync_complete=True))
        save(run / 'workload-launch-intent.json', {})
        (out / 'complete.json').write_bytes(b'done')
        archive = dict(path='histories/lr-0.001.tar.gz', bytes=1024, sha256='a' * 64)
        complete = dict(path='complete.json', bytes=4, sha256=hashlib.sha256(b'done').hexdigest())
        inventory = [complete, archive]
        original_run = safety.subprocess.run
        safety.subprocess.run = lambda *a, **k: SimpleNamespace(stdout=json.dumps(inventory))
        rows = [dict(archive, object_name='0000-lr-0.001.tar.gz')]
        save(run / 'external-readback.json', dict(verified=True, rows=rows, readback='/fresh/fixture'))
        record = dict(status='verified', volume_id=external.VOLUME_ID, run_id=run.name,
                      instance_id=123, rows=rows,
                      readback_proof_sha256=hashlib.sha256((run / 'external-readback.json').read_bytes()).hexdigest())
        save(run / 'external-artifacts.json', record)
        calls = []
        stopped = False
        def instances():
            return [dict(id=123, label='fixture', actual_status='stopped' if stopped else 'running')]
        def instance_api(method, path, body=None):
            nonlocal stopped
            calls.append(('provider', method, path))
            stopped = True
        smoke = SimpleNamespace(STATE=launch / 'active.json', ROOT=launch,
                                list_instances=instances, api=instance_api,
                                _unsafe_cleanup=lambda label: calls.append(('delete', label)))
        save(run / 'export-cleanup-permit.json', dict(verified=True, instance_id=123))
        save(run / 'pending-deferred-artifacts.json', dict(state='deferred_not_verified'))
        safety.guarded_cleanup(smoke, 'fixture')
        assert not any(call[0] == 'delete' for call in calls), 'Pending defer allowed stale-permit deletion'
        assert stopped
        (run / 'external-artifacts.json').unlink()
        try:
            safety.verify_export(run, [], ['complete.json', archive['path']])
        except RuntimeError:
            pass
        else:
            raise AssertionError('Missing local unverified archive accepted')
        assert not (run / 'export-cleanup-permit.json').exists()
        save(run / 'external-artifacts.json', record)
        safety.verify_export(run, [], ['complete.json', archive['path']])
        assert json.loads((run / 'pending-deferred-artifacts.json').read_text())['state'] == 'resolved_verified_exports'
        calls.clear()
        safety.guarded_cleanup(smoke, 'fixture')
        assert any(call[0] == 'delete' for call in calls), 'Exact external receipt did not allow cleanup'
        calls.clear()
        volume_available = False
        safety.guarded_cleanup(smoke, 'fixture')
        assert not any(call[0] == 'delete' for call in calls), 'Missing volume allowed source deletion'
        volume_available = True
        save(run / 'external-artifacts.json', dict(record, instance_id=456))
        calls.clear()
        safety.guarded_cleanup(smoke, 'fixture')
        assert not any(call[0] == 'delete' for call in calls), 'Changed receipt allowed stale-permit deletion'
        try:
            safety.verify_export(run, [], ['complete.json', archive['path']])
        except RuntimeError:
            pass
        else:
            raise AssertionError('Mismatched receipt accepted')
        safety.subprocess.run = original_run
    print('storage safety passed: pending defer blocks deletion; absent local archive '
          'requires exact verified receipt; missing volume and changed receipt block deletion', flush=True)


if __name__ == '__main__':
    main()
