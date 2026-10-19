"""Restart saved credit arms; all computation stays on managed GPU workers."""
from pathlib import Path
import subprocess,json,os,shutil,datetime
R=Path(__file__).resolve().parent
records=[]
processes=[]
for name in ('credit-ordinary-long-20261018','credit-high-default-20261018'):
 c=R/name
 # Evaluation must inherit the hardened launcher too.
 p=c/'finish_campaign.py';s=p.read_text().replace("shutil.copy2(Path('/home/jump/vast-tools')/name,launch/name)","shutil.copy2(c/'launchers/baseline'/name,launch/name)")
 s=s.replace('Credit extension: clarification pending.','Credit extension: original intensity.').replace('Credit extension pending clarification.','Credit extension: original intensity.').replace('Credit extension still pending clarification.','Credit extension: original intensity.')
 # Exclusive supervisor lock prevents duplicate evaluator rentals after a restart.
 if 'flock' not in s:
  s=s.replace('c=Path(__file__).resolve().parent',"c=Path(__file__).resolve().parent\nimport fcntl\nlock=(c/'finish.lock').open('w')\ntry:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError:raise SystemExit('Another evaluator supervisor owns this campaign')")
 p.write_text(s);compile(s,str(p),'exec')
 for arm in ('baseline','easy','hard'):
  launch=c/'launchers'/arm
  if (launch/'active.json').exists():raise RuntimeError('Existing active instance: '+str(launch))
  cap='.85' if name=='credit-high-default-20261018' else '.60'
  if name=='credit-high-default-20261018' and arm=='hard':
   # Measured hard-domain throughput was better on4090 than the previous5090.
   for f in ('smoke.py','example_jax.py'):shutil.copy2(Path('/home/jump/vast-tools')/f,launch/f)
   cap='.60'
  cmd=['uv','run','--no-project','python',str(launch/'job.py'),'--exclude-offer','53368570','--exclude-offer','55701183','--exclude-offer','56219704','--job',str(c/f'resume_{arm}_worker.py'),'--requirements',str(c/'requirements.txt'),'--timeout-seconds','24000','--max-hourly',cap,'--sync-timeout-seconds','600','--require-artifact','complete.json','--require-artifact','cost-10bp/learned/selected.eqx','--',arm]
  log=(c/f'restart-{arm}-launcher.log').open('a')
  proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
  processes.append(proc)
  records.append(dict(campaign=name,arm=arm,pid=proc.pid,started=datetime.datetime.now(datetime.timezone.utc).isoformat(),log=str(c/f'restart-{arm}-launcher.log')))
 log=(c/'restart-supervisor.log').open('a');proc=subprocess.Popen(['uv','run','--no-project','python',str(c/'finish_campaign.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 processes.append(proc)
 (c/'restart-supervisor.pid').write_text(str(proc.pid))
(R/'credit_restart.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))

import time
while any(p.poll() is None for p in processes):
 time.sleep(30)
print("All restart controllers finished",flush=True)
