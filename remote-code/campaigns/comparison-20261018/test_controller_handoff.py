"""Run only on a GPU worker: transfer success, refusal, and fresh patch checks."""
import hashlib
import json
from pathlib import Path
import py_compile
import shutil
from types import SimpleNamespace
import tempfile
from unittest.mock import patch

import controller_handoff as handoff
from enable_handoff import enable
from harden_export import harden


def main():
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        for name in ('job.py', 'smoke.py', 'USAGE.txt'):
            shutil.copy2(root / 'test-launcher-template' / name, folder / name)
        harden(folder)
        enable(folder)
        enable(folder)
        for name in ('job.py', 'smoke.py', 'controller_handoff.py'):
            py_compile.compile(str(folder / name), doraise=True)
        assert "if not args.handoff_confirmed:" in (folder / 'job.py').read_text()
        assert "if not getattr(args, 'handoff_confirmed', False):" in (folder / 'smoke.py').read_text()
        run = folder / 'run-1'
        run.mkdir()
        (run / 'metadata.json').write_text(json.dumps(dict(instance=dict(instance_id=123, label='unique'))))
        (run / 'submitted-job.py').write_text('source')
        args = SimpleNamespace(require_artifact=['complete.json'], handoff_deadline_utc='2026-10-19T06:05:00+00:00')
        original = handoff.atomic_json

        def receiver(path, value):
            original(path, value)
            if path.name == 'handoff.request.json':
                original(run / 'handoff.ack.json', dict(
                    request_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    owner='dh-credit-recovery', instance_id=123, run_id=run.name,
                    deadline_utc=args.handoff_deadline_utc, required_artifacts=args.require_artifact))

        with patch.object(handoff.subprocess, 'run', return_value=SimpleNamespace(stdout='123 bash supervise-workload.sh')):
            with patch.object(handoff, 'atomic_json', side_effect=receiver):
                handoff.request_handoff(run, ['ssh', 'worker'], args)
            assert (run / 'handoff.accepted.json').exists()
            (run / 'handoff.accepted.json').unlink()
            ack = json.loads((run / 'handoff.ack.json').read_text())
            ack['instance_id'] = 456
            original(run / 'handoff.ack.json', ack)
            try:
                handoff.request_handoff(run, ['ssh', 'worker'], args)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Mismatched ownership was accepted')
            assert not (run / 'handoff.accepted.json').exists()
            (run / 'handoff.ack.json').unlink()
            with patch.object(handoff.time, 'monotonic', side_effect=[0, 181]):
                try:
                    handoff.request_handoff(run, ['ssh', 'worker'], args)
                except RuntimeError as exc:
                    assert 'timed out' in str(exc)
                else:
                    raise AssertionError('Unacknowledged transfer was accepted')
        (folder / 'active.json').write_text('{}')
        try:
            enable(folder)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Active launcher modification was allowed')
    print('controller handoff checks passed', flush=True)


if __name__ == '__main__':
    main()
