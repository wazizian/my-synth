"""GPU-worker-only classification proof; provider calls and rentals are mocked."""
import json
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace

provider = ModuleType('smoke')
provider.list_instances = lambda: []
sys.modules['smoke'] = provider
import rental_control as rental
import offer_exclusions


def exercise(root, failures, allowed, prior=''):
    root.mkdir()
    launcher=root/'launcher'
    launcher.mkdir()
    registry=root/'registry.json'
    registry.write_text('[]')
    offer_exclusions.excluded_offer_ids.__defaults__=(registry,)
    log=root/'launcher.log'
    log.write_text(prior)
    calls=[]
    def child(command, **kwargs):
        state=json.loads((launcher/'preallocation-retries.json').read_text())
        assert state['last_attempt_log_offset']==log.stat().st_size
        calls.append(command)
        if len(calls)<=len(failures):
            kwargs['stdout'].write('Offer 777: RTX 4090\nERROR: '+failures[len(calls)-1]+'\n')
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)
    rental.subprocess.run=child
    error=None
    try:
        rental.run_preallocation(['python','job.py','--timeout-seconds','600'],launcher,log,time.time()+3600)
    except RuntimeError as exc:
        error=exc
    assert (error is None)==allowed, (failures,error)
    assert len(calls)==(len(failures)+1 if allowed else len(failures))
    return json.loads((launcher/'preallocation-retries.json').read_text())


def main():
    accepted=['No qualifying RTX 4090 offer below $0.60/hour.']
    for code in (400,404,409):
        for message in ('not available','no_such_ask','not found','does not exist','already rented'):
            accepted.append(f'Vast API HTTP {code} on PUT asks/123/: {message}')
    rejected=[
        'Cannot reach Vast.ai API (network/DNS unavailable).',
        'TimeoutError: provider create request timed out',
        'Vast API HTTP 500 on PUT asks/123/: not available',
        'Vast API HTTP 502 on PUT asks/123/: not available',
        'Vast API HTTP 503 on PUT asks/123/: no_such_ask',
        'Vast API HTTP 409 on PUT asks/123/: insufficient balance',
        'Vast API HTTP 404 on GET asks/123/: not found',
        'Malformed create response: missing new_contract',
        'No such user',
        'Numerical workload exited 1']
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        for index,text in enumerate(accepted):
            state=exercise(root/f'accepted-{index}',[text],True)
            assert state['attempts_started']==2
        for index,text in enumerate(rejected):
            state=exercise(root/f'rejected-{index}',[text],False)
            assert state['attempts_started']==1 and state['attempt_status']=='returned_nonzero'
        old='ERROR: '+accepted[0]+'\n'
        state=exercise(root/'old-accepted-current-timeout',[rejected[1]],False,old)
        assert state['last_attempt_log_offset']==len(old.encode())
        state=exercise(root/'accepted-then-timeout',[accepted[0],rejected[1]],False)
        assert state['attempts_started']==2 and state['attempt_status']=='returned_nonzero'
    print('rental classification passed: explicit no-offer/unavailable PUT rejection retries; '
          'network/timeouts/5xx/malformed replies fail closed; per-attempt offsets reject stale accepted text',flush=True)


if __name__=='__main__':
    main()
