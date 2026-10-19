"""GPU-worker-only financial guard, independent thread and create barrier proof."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace

provider=ModuleType('smoke')
sys.modules['smoke']=provider
import wallet_guard as wallet
import offer_exclusions as exclusions
import budget_admission as budget
import harden_export

ROOT=Path(__file__).resolve().parent


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        wallet.ROOT=root
        marker=root/'funding_halt.json'
        clock=SimpleNamespace(now=1800000000.)
        wallet.time=SimpleNamespace(time=lambda:clock.now)
        launch=root/'owned'
        launch.mkdir()
        (launch/'active.json').write_text(json.dumps(dict(instance_id=1,label='owned')))
        wrong=root/'wrong'
        wrong.mkdir()
        (wrong/'active.json').write_text(json.dumps(dict(instance_id=3,label='wrong')))
        rows=[dict(id=1,label='owned',actual_status='running',dph_total=3),
              dict(id=2,label='unowned',actual_status='running',dph_total=3),
              dict(id=3,label='other',actual_status='running',dph_total=3)]
        credit=3.5
        failures=False
        operations=[]
        provider.list_instances=lambda:rows
        def api(method,path,body=None):
            if method=='GET':
                if failures:raise RuntimeError('wallet API unavailable')
                return dict(credit=credit)
            assert method=='PUT' and path=='instances/1/' and body==dict(state='stopped')
            halt=json.loads(marker.read_text())
            assert halt['compute_stop_required'] and halt['stop_targets'][0]['instance_id']==1
            operations.append((method,path))
            rows[0]['actual_status']='stopped'
        provider.api=api
        wallet.monitor_wallet(rows,[launch,wrong])
        halt=json.loads(marker.read_text())
        assert halt['stop_threshold_usd']==4 and operations==[('PUT','instances/1/')]
        assert halt['stop_targets'][0]['compute_stop_confirmed'] and halt['stop_targets'][0]['disk_retained_confirmed']
        assert not halt['stop_targets'][0]['deletion_requested']
        credit=100
        clock.now+=61
        wallet.monitor_wallet(rows,[launch,wrong])
        assert marker.exists(),'Low-credit halt must require manual review'
        marker.unlink()
        (root/'funding-wallet-state.json').unlink()
        operations.clear()
        failures=True
        for attempt in range(3):
            clock.now+=61
            wallet.monitor_wallet([],[])
            assert marker.exists()==(attempt==2)
        halt=json.loads(marker.read_text())
        assert not halt['compute_stop_required'] and not operations
        failures=False
        clock.now+=61
        wallet.monitor_wallet([],[])
        assert not marker.exists(),'Healthy wallet must clear only API-outage halt'
        registry=root/'registry.json'
        registry.write_text('[]')
        exclusions.excluded_offer_ids.__defaults__=(registry,)
        marker.write_text('malformed marker')
        try:exclusions.exclusion_arguments()
        except RuntimeError:pass
        else:raise AssertionError('Actual offer argument helper bypassed funding halt')
        budget.ROOT=root
        provider.api=lambda method,path,body=None:dict(credit=100) if path=='users/current' else dict(offers=[])
        provider.list_instances=lambda:[]
        assert not budget.snapshot('credit',root/'namespace')['admissible']
        marker.unlink()
        assert budget.snapshot('credit',root/'namespace')['admissible']
        launch=root/'fresh'
        launch.mkdir()
        for name in ('job.py','smoke.py','USAGE.txt'):
            shutil.copy2(ROOT/'test-launcher-template'/name,launch/name)
        harden_export.harden(launch)
        tree=ast.parse((launch/'smoke.py').read_text())
        guarded=next(node for node in ast.walk(tree) if isinstance(node,ast.Try)
                     and any(isinstance(item,ast.If) and 'funding_halt.json' in ast.unparse(item.test) for item in node.body))
        index=next(i for i,item in enumerate(guarded.body) if isinstance(item,ast.If) and 'funding_halt.json' in ast.unparse(item.test))
        assert isinstance(guarded.body[index+1],ast.Assign) and guarded.body[index+1].targets[0].id=='created'
        block=compile(ast.fix_missing_locations(ast.Module(body=guarded.body[index:index+2],type_ignores=[])),'create-barrier','exec')
        creates=[]
        env=dict(Path=lambda path:marker,api=lambda *args:creates.append(args),offer=dict(id=123),
                 args=SimpleNamespace(image='fixture',disk=40),label='fixture')
        marker.write_text('malformed marker')
        try:exec(block,env)
        except RuntimeError:pass
        else:raise AssertionError('Literal create barrier accepted malformed halt marker')
        assert not creates
        marker.unlink()
        exec(block,env)
        assert len(creates)==1 and creates[0][0]=='PUT'
        (launch/'active.json').write_text('{}')
        try:harden_export.harden(launch)
        except RuntimeError:pass
        else:raise AssertionError('Active launcher patch allowed')
        early=ast.parse((ROOT/'early_gate.py').read_text())
        pair=None
        for parent in ast.walk(early):
            body=getattr(parent,'body',None)
            if not isinstance(body,list):continue
            for i,item in enumerate(body[:-1]):
                if isinstance(item,ast.Expr) and isinstance(item.value,ast.Call) and isinstance(item.value.func,ast.Name) and item.value.func.id=='assert_funding_available':
                    pair=body[i:i+2]
        assert pair is not None and isinstance(pair[1].value.func,ast.Attribute) and pair[1].value.func.attr=='run'
        earlycalls=[]
        env=dict(assert_funding_available=wallet.assert_funding_available,args=[],
                 subprocess=SimpleNamespace(run=lambda *a,**k:earlycalls.append(a)))
        block=compile(ast.fix_missing_locations(ast.Module(body=pair,type_ignores=[])),'early-barrier','exec')
        marker.write_text('malformed marker')
        try:exec(block,env)
        except RuntimeError:pass
        else:raise AssertionError('Early gate bypassed funding halt')
        assert not earlycalls
        marker.unlink()
        exec(block,env)
        assert len(earlycalls)==1
        watchdog=ast.parse((ROOT/'watchdog.py').read_text())
        thread=next(node for node in ast.walk(watchdog) if isinstance(node,ast.Expr)
                    and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
                    and node.value.func.attr=='start' and 'wallet_loop' in ast.unparse(node))
        constructed=[]
        def make_thread(**kwargs):
            constructed.append(kwargs)
            return SimpleNamespace(start=lambda:constructed.append('started'))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[thread],type_ignores=[])),'wallet-thread','exec'),
             dict(threading=SimpleNamespace(Thread=make_thread),wallet_loop=wallet.wallet_loop,launches=lambda:[]))
        assert constructed[0]['daemon'] and constructed[0]['target'] is wallet.wallet_loop and constructed[-1]=='started'
        event=threading.Event()
        class EndLoop(BaseException):pass
        def monitor(*args):
            event.set()
            raise EndLoop()
        wallet.monitor_wallet=monitor
        def independent():
            try:wallet.wallet_loop(lambda:[])
            except EndLoop:pass
        real=threading.Thread(target=independent)
        real.start()
        assert event.wait(1),'Wallet polling blocked behind simulated export wait'
        real.join()
    print('wallet guard proof passed: reserve+burn STOP intent/identity/no-delete, API-error halt/heal, '
          'manual low halt, actual offer/budget/create/early barriers, independent watchdog thread',flush=True)


if __name__=='__main__':main()
