"""GPU-worker-only actual command/retry proof; no provider calls or allocations."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace

provider = ModuleType('smoke')
sys.modules['smoke'] = provider
import offer_exclusions
import rental_control as rental
import wallet_guard

ROOT = Path(__file__).resolve().parent


def case(root, behavior, expected_calls, expected_error=True):
    root.mkdir()
    launcher = root / 'launcher'
    launcher.mkdir()
    registry = root / 'registry.json'
    registry.write_text('[12345]')
    offer_exclusions.excluded_offer_ids.__defaults__ = (registry,)
    calls = []
    queries = 0
    if behavior.startswith('resume-'):
        (launcher/'preallocation-retries.json').write_text(json.dumps(dict(
            attempts_started=1,retries=0,excluded_offers=[],
            attempt_status='inflight' if behavior=='resume-interrupted' else 'returned_nonzero',
            child_returncode=(0 if behavior=='resume-zero' else True if behavior=='resume-bool' else 1),
            last_attempt_log_offset=0,
            last_attempt_before_provider_ids=[])))
        (root/'launcher.log').write_text('Offer 888: RTX 4090\nERROR: No qualifying RTX 4090 offer below $0.60/hour.\n')
        if behavior=='resume-missing':
            state=json.loads((launcher/'preallocation-retries.json').read_text())
            state.pop('child_returncode')
            (launcher/'preallocation-retries.json').write_text(json.dumps(state))
        if behavior=='resume-active':
            (launcher/'active.json').write_text('{}')
        if behavior=='resume-intent':
            run=launcher/'runs'/'fixture'
            run.mkdir(parents=True)
            (run/'workload-launch-intent.json').write_text('{}')
    def instances():
        nonlocal queries
        queries += 1
        if behavior=='absence-query-error' and queries>=2:
            raise RuntimeError('Provider absence API unavailable')
        if behavior == 'resume-new' or behavior == 'new' and queries >= 2:
            return [dict(id=999)]
        if behavior == 'old':
            return [dict(id=42)]
        return []
    provider.list_instances = instances
    def execute(command, **kwargs):
        assert json.loads((launcher/'preallocation-retries.json').read_text())['attempt_status']=='inflight'
        calls.append(command)
        kwargs['stdout'].write('Offer 777: RTX 4090\n')
        kwargs['stdout'].write('ERROR: No qualifying RTX 4090 offer below $0.60/hour.\n')
        if behavior == 'retry' and len(calls) == 1:
            registry.write_text('[12345,54321]')
        if behavior == 'active':
            (launcher/'active.json').write_text('{}')
        if behavior in ('intent', 'old'):
            run = launcher/'runs'/'fixture'
            run.mkdir(parents=True)
            if behavior == 'intent':
                (run/'workload-launch-intent.json').write_text('{}')
            else:
                (run/'metadata.json').write_text(json.dumps(dict(instance=dict(instance_id=42))))
        return SimpleNamespace(returncode=0 if behavior=='resume-absent' or behavior == 'retry' and len(calls) == 2 else 1)
    rental.subprocess.run = execute
    deadline = time.time() + (299 if behavior == 'deadline' else 10000)
    error = None
    try:
        rental.run_preallocation(['python','job.py','--timeout-seconds','7200','--max-hourly','0.60'],
                                 launcher, root/'launcher.log', deadline)
    except RuntimeError as exc:
        error = exc
    assert len(calls) == expected_calls, (behavior, len(calls), error)
    assert (error is not None) == expected_error, (behavior, error)
    if behavior == 'retry':
        got = {int(command[index+1]) for command in calls for index,value in enumerate(command) if value == '--exclude-offer'}
        assert {777,12345,54321,46384002,28780319} <= got
        assert calls[1][calls[1].index('--timeout-seconds')+1] == '7200'
    if behavior=='resume-absent':
        state=json.loads((launcher/'preallocation-retries.json').read_text())
        assert state['attempts_started']==2 and state['attempt_status']=='completed'
        assert '888' in calls[0]


def actual_commands(root):
    registry = root/'commands-registry.json'
    registry.write_text('[12345]')
    offer_exclusions.excluded_offer_ids.__defaults__ = (registry,)
    provider.list_instances = lambda: []
    commands = []
    rental.subprocess.run = lambda command,**kwargs: (commands.append(command) or SimpleNamespace(returncode=0))
    for filename in ('credit_pipeline.py','option_pipeline.py','launch_freeze.py'):
        tree = ast.parse((ROOT/filename).read_text())
        assignment = next(node for node in ast.walk(tree) if isinstance(node,ast.Assign)
                          and any(isinstance(target,ast.Name) and target.id=='command' for target in node.targets))
        call = next(node for node in ast.walk(tree) if isinstance(node,ast.Expr)
                    and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)
                    and node.value.func.id=='run_preallocation')
        folder = root/filename
        folder.mkdir()
        launch = folder/'launcher'
        launch.mkdir()
        log=folder/'launcher.log'
        log.write_text('Offer 888: RTX 4090\nERROR: No qualifying RTX 4090 offer below $0.60/hour.\n')
        saved=dict(attempts_started=1,retries=0,excluded_offers=[],attempt_status='returned_nonzero',child_returncode=1,
                   last_attempt_log_offset=0,
                   last_attempt_before_provider_ids=[])
        (launch/'preallocation-retries.json').write_text(json.dumps(saved))
        env = dict(launch=launch,worker=folder/'worker.py',source=root/'source',ROOT=root,
                   folder=folder,evaluation=folder,deadline=time.time()+10000,
                   DEADLINE=time.time()+10000,time=time,run_preallocation=rental.run_preallocation,
                   resumable_preallocation=rental.resumable_preallocation)
        guard=next(node for node in ast.walk(tree) if isinstance(node,ast.If)
                   and any(isinstance(child,ast.Call) and isinstance(child.func,ast.Name)
                           and child.func.id=='resumable_preallocation' for child in ast.walk(node.test)))
        guard_code=compile(ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[])),filename,'exec')
        exec(guard_code,env)
        (launch/'active.json').write_text('{}')
        try:
            exec(guard_code,env)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Actual caller restart guard accepted active instance')
        (launch/'active.json').unlink()
        (launch/'preallocation-retries.json').write_text(json.dumps(dict(saved,attempt_status='inflight')))
        try:
            exec(guard_code,env)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Actual caller restart guard accepted interrupted inflight request')
        (launch/'preallocation-retries.json').write_text(json.dumps(dict(saved,attempt_status='unknown')))
        try:
            exec(guard_code,env)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Actual caller restart guard accepted unknown prior state')
        (launch/'preallocation-retries.json').write_text(json.dumps(saved))
        block = ast.fix_missing_locations(ast.Module(body=[assignment,call],type_ignores=[]))
        exec(compile(block,filename,'exec'),env)
        command = commands[-1]
        assert env['command'][:4]==['uv','run','--no-project','python']
        canonical=[sys.executable]+env['command'][4:]
        assert command[:len(canonical)]==canonical, 'Canonicalization changed arguments beyond the launcher prefix'
        got = {int(command[index+1]) for index,value in enumerate(command) if value=='--exclude-offer'}
        assert offer_exclusions.KNOWN|{12345} <= got
        assert command[command.index('--max-hourly')+1]=='0.60'
        required = [command[index+1] for index,value in enumerate(command) if value=='--require-artifact']
        assert required == (['credit_freeze.json'] if filename=='launch_freeze.py'
                            else ['evaluation/complete.json','evaluation/results.json'])
    assert len(commands)==3


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        wallet_guard.ROOT=root
        case(root/'retry','retry',2,False)
        for behavior in ('active','intent','new','old','absence-query-error'):
            case(root/behavior,behavior,1)
        case(root/'deadline','deadline',0)
        case(root/'max','max',3)
        for behavior in ('resume-new','resume-active','resume-intent','resume-interrupted','resume-zero','resume-bool','resume-missing'):
            case(root/behavior,behavior,0)
        case(root/'resume-absent','resume-absent',1,False)
        actual_commands(root)
        launcher=root/'funding-blocked'
        launcher.mkdir()
        (root/'funding_halt.json').write_text('malformed marker')
        blocked=[]
        rental.subprocess.run=lambda command,**kwargs:blocked.append(command)
        try:
            rental.run_preallocation(['uv','run','--no-project','python','job.py','--timeout-seconds','600'],
                                     launcher,root/'funding.log',time.time()+3600)
        except RuntimeError as exc:
            assert 'funding halt' in str(exc)
        else:
            raise AssertionError('Prefix fix bypassed financial halt')
        assert not blocked
    print('rental proof passed: actual three final/freeze commands canonicalized to sys.executable '
          'with all remaining arguments unchanged; financial halt, exclusions, preallocation guards, '
          'deadline and max2 retries preserved',flush=True)


if __name__=='__main__':
    main()
