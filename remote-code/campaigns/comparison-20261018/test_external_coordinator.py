"""GPU-worker-only retries and nonproductive-stop tests for actual coordinator."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace

provider = ModuleType('smoke')
sys.modules['smoke'] = provider
spec = importlib.util.spec_from_file_location('coordinator', Path(__file__).with_name('external_export_coordinator.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class StopLoop(BaseException):
    pass


def exercise(root, failures, productive=False, fatal=False):
    root.mkdir()
    module.ROOT = root
    run = root / 'launcher' / 'runs' / 'fixture'
    run.mkdir(parents=True)
    def save(path, value):
        path.write_text(json.dumps(value))
    instance = dict(instance_id=123, label='fixture')
    save(run / 'metadata.json', dict(instance=instance, ssh=['SSH']))
    save(run.parent.parent / 'active.json', instance)
    save(run / 'external-export-request.json', dict(instance_id=123, rows=[]))
    clock = SimpleNamespace(now=1792363000.0, sleeps=0)
    def sleep(seconds):
        clock.now += seconds
        clock.sleeps += 1
        if clock.sleeps >= 3:
            raise StopLoop()
    module.time = SimpleNamespace(time=lambda: clock.now, monotonic=lambda: clock.now, sleep=sleep)
    module.subprocess.run = lambda *a, **k: SimpleNamespace(stdout='RUNNING' if productive else '0')
    operations = []
    state = dict(id=123, label='fixture', actual_status='running')
    provider.list_instances = lambda: [state]
    def api(method, path, body=None):
        operations.append((method, path))
        assert method != 'DELETE', 'Coordinator must never delete artifacts or workers'
        state['actual_status'] = 'stopped'
    provider.api = api
    exports = 0
    def export(*args, **kwargs):
        nonlocal exports
        exports += 1
        save(run / 'external-export-state.json', dict(started=1792363000.0))
        if exports <= failures:
            if fatal:
                raise RuntimeError('Source SHA changed')
            raise TimeoutError('Transport timeout')
        save(run / 'external-artifacts.json', dict(status='verified'))
    module.ensure_external = export
    try:
        module.main()
    except StopLoop:
        pass
    if productive:
        assert not exports and not operations
        assert (run / 'external-export-refused.json').exists()
    elif fatal:
        assert exports == 1 and len(operations) == 1
    elif failures == 1:
        assert exports == 2 and not operations
        assert (run / 'external-artifacts.json').exists()
    else:
        assert exports == 2 and len(operations) == 1
        failure = json.loads((run / 'external-export-failure.json').read_text())
        assert failure['deletion_not_requested'] and failure['source_disk_preserved_confirmed']
    # Provider absence must not be described as confirmed disk preservation.
    provider.list_instances = lambda: []
    module.stop_and_preserve(run, dict(instance=instance), RuntimeError('fixture'))
    failure = json.loads((run / 'external-export-failure.json').read_text())
    assert failure['deletion_not_requested'] and not failure['source_disk_preserved_confirmed']


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    exercise(root / 'recovered', 1)
    exercise(root / 'exhausted', 2)
    exercise(root / 'fatal', 1, fatal=True)
    exercise(root / 'productive', 0, productive=True)
print('coordinator bounded retry, exhausted/fatal source-preserving stop, '
      'productive refusal and absent-provider reporting passed', flush=True)
