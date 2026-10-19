"""GPU-worker-only stdlib control fixture."""
import hashlib
import ast
import json
from pathlib import Path
import py_compile
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import controller_handoff as h
from enable_handoff import enable
from harden_export import harden

class Clock:
    now = 0
    def monotonic(self): return self.now
    def sleep(self, seconds): self.now += seconds

def result(text, code=0):
    return SimpleNamespace(stdout=text, stderr='', returncode=code)

def main():
    source = Path(__file__).parent
    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp)
        fresh = base / 'fresh'; fresh.mkdir()
        for name in ('job.py', 'smoke.py', 'USAGE.txt'):
            shutil.copy2(source / 'template' / name, fresh / name)
        harden(fresh); enable(fresh); enable(fresh)
        for name in ('job.py','smoke.py','controller_handoff.py'):
            py_compile.compile(str(fresh/name), doraise=True)
        import traceback
        tree = ast.parse((fresh/'job.py').read_text())
        inner = next(node for node in ast.walk(tree) if isinstance(node, ast.Try)
                     and any(isinstance(child, ast.Constant) and child.value == 'original-launch-error.json'
                             for handler in node.handlers for child in ast.walk(handler)))
        inner.body = ast.parse("raise ValueError('original setup failure')").body
        inner.finalbody = ast.parse("raise RuntimeError('masked final verification failure')").body
        fixture = ast.fix_missing_locations(ast.Module(body=[inner],type_ignores=[]))
        try: exec(compile(fixture,'actual-inner-handler','exec'),{'atomic_json':h.atomic_json,'run_dir':fresh,'traceback':traceback})
        except RuntimeError as error: assert 'masked final' in str(error)
        receipt=json.loads((fresh/'original-launch-error.json').read_text())
        assert receipt['type']=='ValueError' and receipt['error']=='original setup failure'
        assert 'ValueError' in receipt['traceback']
        (fresh/'active.json').write_text('{}')
        try: enable(fresh)
        except RuntimeError: pass
        else: raise AssertionError('active patch allowed')
        args = SimpleNamespace(require_artifact=['complete.json'], handoff_deadline_utc='2026-10-19T06:05:00+00:00')
        original = h.atomic_json
        checks=[]
        for case in ('transient', 'terminal', 'deadline', 'invalid', 'badack', 'noack'):
            run = base/case; run.mkdir()
            (run/'metadata.json').write_text(json.dumps({'instance':{'instance_id':123,'label':'same-worker'}}))
            (run/'submitted-job.py').write_text('frozen source')
            clock=Clock(); calls=[]
            sequence = iter([subprocess.TimeoutExpired('ssh',20), result('',255), result('MISSING'), result('LIVE')])
            def remote(command, **kwargs):
                calls.append(command)
                assert command[:2] == ['ssh','same-worker']
                assert 'workload.exit' in command[-1]
                assert 'pgrep' in command[-1]
                if case=='transient':
                    value=next(sequence)
                    if isinstance(value,Exception): raise value
                    return value
                return result({'terminal':'EXIT 0','deadline':'MISSING','invalid':'ALIEN'}.get(case,'LIVE'))
            def receiver(path,value):
                original(path,value)
                if path.name=='handoff.request.json' and case!='noack':
                    original(run/'handoff.ack.json',dict(request_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),owner='dh-credit-recovery',instance_id=456 if case=='badack' else 123,run_id=run.name,deadline_utc=args.handoff_deadline_utc,required_artifacts=args.require_artifact))
            with patch.object(h.subprocess,'run',side_effect=remote),patch.object(h,'atomic_json',side_effect=receiver),patch.object(h.time,'monotonic',side_effect=clock.monotonic),patch.object(h.time,'sleep',side_effect=clock.sleep):
                try: h.request_handoff(run,['ssh','same-worker'],args)
                except RuntimeError as error:
                    assert case!='transient'
                    assert (run/'original-handoff-error.json').exists()
                    assert str(error)==json.loads((run/'original-handoff-error.json').read_text())['error']
                    assert not (run/'handoff.accepted.json').exists()
                else:
                    assert case=='transient'
                    assert len(calls)==4
                    assert (run/'handoff.accepted.json').exists()
            if case in ('terminal','deadline','invalid'):
                assert not (run/'handoff.request.json').exists()
            if case=='deadline': assert clock.now==60
            assert (run/'handoff.probes.json').exists()
            checks.append(case)
        print(json.dumps({'status':'PASS','cases':checks,'fresh_launcher_compile':True,'active_patch_refused':True,'same_worker_only':True,'original_errors_durable':True}),flush=True)

if __name__=='__main__': main()
