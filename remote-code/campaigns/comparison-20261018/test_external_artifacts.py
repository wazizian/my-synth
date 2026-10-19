"""GPU-worker-only mocked transport/crash/recovery checks; no live API access."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType

fake_smoke = ModuleType('smoke')
sys.modules['smoke'] = fake_smoke
spec = importlib.util.spec_from_file_location('external_artifacts', Path(__file__).with_name('external_artifacts.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Crash(BaseException):
    pass


def exercise(folder, crash_direction=None, after=False, timeout_resume=False, prepare_timeout=False, stale_crash=False):
    clock = SimpleNamespace(now=1792363000.0)
    module.time = SimpleNamespace(time=lambda: clock.now,
                                  sleep=lambda seconds: setattr(clock, 'now', clock.now + seconds))
    module.CAMPAIGN = folder
    folder.mkdir()
    run = folder / 'run'
    run.mkdir()
    (run / 'metadata.json').write_text(json.dumps(dict(instance=dict(instance_id=123))))
    rows = [dict(path='storage-proof/fixture.bin', bytes=1024, sha256='a' * 64)]
    cancelled = []

    def api(method, path, body=None):
        if method == 'DELETE':
            cancelled.append(body)
            return dict(success=True)
        return dict(volumes=[dict(id=module.VOLUME_ID, machine_id=module.MACHINE_ID,
                                 label=module.LABEL, disk_space=120, status='created',
                                 end_date=clock.now + 100000)])

    fake_smoke.api = api
    fake_smoke.list_instances = lambda: [dict(id=123, status_msg='Done copying')]
    operations = []
    crashed = False
    pending = timeout_resume
    prepare_calls = 0
    readback_calls = 0
    verify_calls = 0

    def provider(source_id, source_path, dest_id, dest_path):
        nonlocal crashed, readback_calls
        direction = 'upload' if source_id.startswith('C.') else 'readback'
        if direction == 'readback':
            readback_calls += 1
        if stale_crash and direction == 'readback' and readback_calls == 2:
            if after:
                operations.append((source_id, source_path, dest_id, dest_path))
            raise Crash()
        if direction == crash_direction and not crashed:
            crashed = True
            if after:
                operations.append((source_id, source_path, dest_id, dest_path))
            raise Crash()
        operations.append((source_id, source_path, dest_id, dest_path))
        return dict(success=True)

    def worker(ssh, request):
        nonlocal prepare_calls, verify_calls
        if request['action'] == 'prepare':
            prepare_calls += 1
            if prepare_timeout and prepare_calls == 1:
                clock.now += 31
                raise Crash()
            return dict(prepared=True)
        verify_calls += 1
        if stale_crash and verify_calls == 1:
            return dict(verified=False, pending=True, stale_readback=True)
        if pending:
            return dict(verified=False, pending=True)
        return dict(verified=True, rows=request['rows'], readback=request['readback'])

    module.provider_copy = provider
    module.worker = worker
    if stale_crash:
        try:
            module.ensure_external(run, [], rows, fixture=True, timeout_seconds=30)
        except Crash:
            pass
        else:
            raise AssertionError('ESTALE retry interruption was not exercised')
        interrupted = json.loads((run / 'external-export-state.json').read_text())
        assert interrupted['phase'] == 'readback_intent'
        assert interrupted['readback_requests'] == 2
        try:
            module.ensure_external(run, [], rows, fixture=True, timeout_seconds=30)
        except RuntimeError as exc:
            assert 'limit' in str(exc)
        else:
            raise AssertionError('Exhausted ambiguous request must fail closed')
        assert not (run / 'external-artifacts.json').exists()
        return
    elif prepare_timeout:
        try:
            module.ensure_external(run, [], rows, fixture=True, timeout_seconds=30)
        except Crash:
            pass
        else:
            raise AssertionError('Prepare interruption was not exercised')
        module.ensure_external(run, [], rows, fixture=True, timeout_seconds=30)
        assert prepare_calls == 2
        assert json.loads((run / 'external-export-state.json').read_text())['attempt'] == 2
    elif crash_direction:
        try:
            module.ensure_external(run, [], rows, fixture=True, timeout_seconds=30)
        except Crash:
            pass
        else:
            raise AssertionError('Injected request crash was not exercised')
        interrupted = json.loads((run / 'external-export-state.json').read_text())
        assert interrupted['phase'].endswith('_intent')
        module.ensure_external(run, [], rows, fixture=True, timeout_seconds=30)
        completed = json.loads((run / 'external-export-state.json').read_text())
        assert completed['phase'] == 'verified'
        if crash_direction == 'readback':
            assert completed['readback'] != interrupted['readback'] and cancelled
    elif timeout_resume:
        try:
            module.ensure_external(run, [], rows, fixture=True, timeout_seconds=30)
        except TimeoutError:
            pass
        else:
            raise AssertionError('Readback timeout was not exercised')
        pending = False
        module.ensure_external(run, [], rows, fixture=True, timeout_seconds=30)
        assert json.loads((run / 'external-export-state.json').read_text())['attempt'] == 2
        assert cancelled
    else:
        module.ensure_external(run, [], rows, fixture=True, timeout_seconds=30)
    assert module.verified_external_rows(run, rows)
    assert module.rsync_excludes(run) == ['--exclude=/storage-proof/fixture.bin']
    before = len(operations)
    module.ensure_external(run, [], rows, fixture=True, timeout_seconds=30)
    assert len(operations) == before, 'Verified resume must not retransmit'
    try:
        module.verified_external_rows(run, [dict(rows[0], sha256='b' * 64)])
    except RuntimeError:
        pass
    else:
        raise AssertionError('Changed immutable artifact accepted')


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        exercise(root / 'success')
        for direction in ('upload', 'readback'):
            for after in (False, True):
                exercise(root / (direction + str(after)), direction, after)
        exercise(root / 'timeout-resume', timeout_resume=True)
        exercise(root / 'prepare-timeout-resume', prepare_timeout=True)
        for after in (False, True):
            exercise(root / ('stale-crash' + str(after)), after=after, stale_crash=True)
    print('external artifact crash-before/after-request, timeout-resume, '
          'idempotence and changed-SHA checks passed', flush=True)


if __name__ == '__main__':
    main()
