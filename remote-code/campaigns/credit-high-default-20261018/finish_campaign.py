"""Control-plane supervision only; all numerical evaluation runs on GPU."""
from pathlib import Path
import time,json,hashlib,base64,subprocess,shutil,traceback
c=Path(__file__).resolve().parent
import fcntl
lock=(c/'finish.lock').open('w')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit('Another evaluator supervisor owns this campaign')
try:
 # A restarted service must adopt an already allocated evaluation, never duplicate it.
 evaluation=c/'launchers/evaluation'
 while (evaluation/'active.json').exists():
  (c/'STATE.md').write_text('Waiting for existing evaluation/export recovery; no duplicate allocation.\n')
  time.sleep(30)
 completed=[r for r in sorted((evaluation/'runs').glob('*')) if (r/'outputs/evaluation/complete.json').exists() and (r/'outputs/evaluation/results.json').exists() and (r/'artifacts.json').exists()]
 if completed:
  shutil.copy2(completed[-1]/'outputs/evaluation/results.json',c/'results.json')
  (c/'STATE.md').write_text('Reference-only diagnostic evaluation complete on stream820261017; final520261017 remains sealed. Export verified and instance cleanup confirmed.\n')
  raise SystemExit(0)
 while True:
  chosen={}
  for arm in ('baseline','easy','hard'):
   runs=sorted((c/'launchers'/arm/'runs').glob('*'))
   for run in runs:
    p=run/'outputs/complete.json'
    if p.exists() and json.loads(p.read_text()).get('status')=='complete' and (run/'artifacts.json').exists() and not (c/'launchers'/arm/'active.json').exists():chosen[arm]=run
  if len(chosen)==3:break
  (c/'STATE.md').write_text('TRAINING: ordinary-only baseline, all arms 1800 active seconds per LR at costs1/3/10bp.\nCompleted arms: '+', '.join(chosen)+'\nCredit extension: triple default intensity.\n')
  time.sleep(30)
 files={str(p.relative_to(c/'source')):p.read_text() for p in (c/'source').rglob('*') if p.is_file() and p.suffix in ('.py','.json')}
 payload={}
 for arm,run in chosen.items():
  for bp in (1,3,10):
   root=run/f'outputs/cost-{bp}bp/learned'
   for name in ('selection.json','selected.eqx'):
    payload[f'cost-{bp}bp/{arm}/{name}']=base64.b64encode((root/name).read_bytes()).decode()
 worker='''import os,pathlib,sys,base64,json
import jax
jax.config.update('jax_enable_x64',True)
assert jax.default_backend()=='gpu'
root=pathlib.Path(os.environ['JOB_OUTPUT_DIR']);src=root/'source';src.mkdir(exist_ok=True)
for n,v in FILES.items():
 p=src/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(v)
for n,v in PAYLOAD.items():
 p=root/'selected'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(v))
sys.path.insert(0,str(src))
from credit_hedging.followup_evaluation import main
main(src,root/'evaluation',root/'selected')
'''.replace('FILES',repr(files),1).replace('PAYLOAD',repr(payload),1)
 # Replacement must target identifiers in iteration expressions, not the literal itself.
 (c/'evaluation_worker.py').write_text(worker)
 compile(worker,'evaluation_worker.py','exec')
 launch=c/'launchers/evaluation';launch.mkdir(exist_ok=True)
 for name in ('job.py','smoke.py','requirements-jax.txt','example_jax.py','USAGE.txt','AGENTS.md'):
  shutil.copy2(c/'launchers/baseline'/name,launch/name)
 subprocess.run(['python',str(c.parent/'comparison-20261018/harden_export.py'),str(launch)],check=True)
 (c/'training_runs.json').write_text(json.dumps({a:str(r) for a,r in chosen.items()},indent=2))
 (c/'STATE.md').write_text('EVALUATING on GPU. All three training arms complete. Credit extension: triple default intensity.\n')
 cmd=['uv','run','--no-project','python',str(launch/'job.py'),'--exclude-offer','47076390','--job',str(c/'evaluation_worker.py'),'--requirements',str(c/'requirements.txt'),'--timeout-seconds','7200','--max-hourly','0.85','--require-artifact','evaluation/complete.json','--require-artifact','evaluation/results.json']
 with (c/'evaluation_launcher.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError('Evaluation launcher failed; inspect evaluation_launcher.log')
 run=next(r for r in sorted((launch/'runs').glob('*'),reverse=True) if (r/'outputs/evaluation/complete.json').exists())
 shutil.copy2(run/'outputs/evaluation/results.json',c/'results.json')
 (c/'STATE.md').write_text('TRAINING AND EVALUATION COMPLETE: ordinary-only baseline,1800s per LR,costs1/3/10bp. GPU launcher confirmed cleanup. Triple default intensity. Results in results.json.\n')
except Exception:
 (c/'supervisor_failure.txt').write_text(traceback.format_exc());raise
