"""Complete sealed option evaluation after recovered three-arm training; no duplicate rentals."""
import base64,datetime,fcntl,json,subprocess,time,shutil,sys
from pathlib import Path
C=Path(__file__).resolve().parent
sys.path.insert(0,str(C.parent/'comparison-20261018'))
from wallet_guard import assert_funding_available
from offer_exclusions import exclusion_arguments
lock=(C/'finisher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def trained(arm):
 l=C/'launchers'/arm
 if (l/'active.json').exists():return
 for r in sorted((l/'runs').glob('*'),reverse=True):
  if (r/'artifacts.json').exists() and (r/'outputs/training/complete.json').exists():return r
while True:
 training={a:trained(a) for a in ['baseline','easy','hard']}
 if all(training.values()):break
 time.sleep(30)
# Preserve source used by evaluation; no training or selection holdout is reopened.
payload={}
for arm,r in training.items():
 for cost in [1,3,10]:payload[f'{arm}/cost-{cost}bp/selected.eqx']=base64.b64encode((r/f'outputs/training/cost-{cost}bp/selected.eqx').read_bytes()).decode()
diagnostic=json.loads((C/'diagnostic.json').read_text())
worker=(C/'worker.py').read_text().replace('520261017','820261017').replace('MODELS={}','MODELS='+repr(payload)).replace("evaluate.main(root/'evaluation',root/'models',{})","evaluate.main(root/'evaluation',root/'models',"+repr(diagnostic)+")")
(C/'evaluation_worker.py').write_text(worker)
l=C/'launchers/evaluation';l.mkdir(exist_ok=True)
for name in ['job.py','smoke.py','example_jax.py','requirements-jax.txt','AGENTS.md','USAGE.txt']:
 if not (l/name).exists():shutil.copy2(C/'launchers/baseline'/name,l/name)
if (l/'active.json').exists():raise RuntimeError('Existing evaluation instance; inspect and reattach')
subprocess.run([sys.executable,str(C.parent/'comparison-20261018/harden_export.py'),str(l)],check=True)
# An already exported evaluation is reusable after a controller restart.
complete=[r for r in (l/'runs').glob('*') if (r/'outputs/evaluation/results.json').exists() and (r/'artifacts.json').exists()] if (l/'runs').exists() else []
if not complete:
 args=[sys.executable,str(l/'job.py'),'--exclude-offer','48448052','--exclude-offer','55701183','--exclude-offer','51500671','--exclude-offer','56219704','--exclude-offer','55917102','--job',str(C/'evaluation_worker.py'),'--requirements',str(C/'requirements.txt'),'--timeout-seconds','7200','--max-hourly','.60','--sync-timeout-seconds','600','--require-artifact','evaluation/results.json','--require-artifact','evaluation/complete.json','--','evaluation']
 args[-2:-2]=exclusion_arguments()
 assert_funding_available()
 subprocess.run(args,check=True)
 complete=[r for r in (l/'runs').glob('*') if (r/'outputs/evaluation/results.json').exists() and (r/'artifacts.json').exists()]
r=max(complete);shutil.copy2(r/'outputs/evaluation/results.json',C/'results.json')
(C/'STATE.json').write_text(json.dumps(dict(phase='complete',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_runs={a:str(v) for a,v in training.items()},evaluation_run=str(r),all_workers_cleaned=True),indent=2))
print('VOLATILITY SCREEN COMPLETE',flush=True)
