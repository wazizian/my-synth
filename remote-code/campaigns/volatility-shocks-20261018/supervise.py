"""Bounded GPU pipeline; local work restricted to orchestration and artifact handling."""
import os,sys,json,time,subprocess,shutil,base64,datetime,csv,traceback
from pathlib import Path
C=Path(__file__).resolve().parent
TOOLS=Path('/home/jump/vast-tools')
CHILDREN={}
LAUNCHES={}
RETRIES={}
EXCLUDED={55701183,51500671,56219704,55917102}
PREVIOUS_RUNS={}
def write(name,value):
 p=C/name;tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,p)
def launcher(name):
 p=C/'launchers'/name;p.mkdir(parents=True,exist_ok=True)
 if not (p/'job.py').exists():
  for f in ('job.py','smoke.py','example_jax.py','requirements-jax.txt','AGENTS.md','USAGE.txt'):shutil.copy2((Path('/home/jump/deep-hedging/campaigns/credit-ordinary-long-20261018/launchers/baseline') if f=='job.py' else TOOLS)/f,p/f)
 return p
def launch(name,job,mode,timeout,artifacts):
 l=launcher(name)
 LAUNCHES[name]=(job,mode,timeout,artifacts)
 if (l/'active.json').exists():raise RuntimeError('Existing managed instance: '+name)
 args=['uv','run','--no-project','python',str(l/'job.py'),'--exclude-offer','55701183','--exclude-offer','51500671','--exclude-offer','56219704','--job',str(job),'--requirements',str(C/'requirements.txt'),'--timeout-seconds',str(timeout),'--max-hourly','0.60','--sync-timeout-seconds','600']
 for offer in sorted(EXCLUDED):args+=['--exclude-offer',str(offer)]
 for a in artifacts:args+=['--require-artifact',a]
 args+=['--',mode]
 PREVIOUS_RUNS[name]={r.name for r in (l/'runs').glob('*')}
 log=(C/(name+'-launcher.log')).open('a');p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);CHILDREN[name]=(p,log)
 write('STATE.json',dict(phase='launching '+name,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=p.pid))
 return p

def latest(name):
 runs=sorted((C/'launchers'/name/'runs').glob('*'))
 return runs[-1] if runs else None

def wait(name,required,timeout):
 start=time.monotonic()
 while True:
  monitor()
  run=latest(name)
  if run and run.name in PREVIOUS_RUNS.get(name,set()):run=None
  active=(C/'launchers'/name/'active.json').exists()
  if run and not active and (run/'artifacts.json').exists():
   path=run/'outputs'/required
   if path.exists():return run
   # Only current submitted pipeline child ending establishes terminal failure.
   if name in CHILDREN and CHILDREN[name][0].poll() is not None:raise RuntimeError(name+' failed: '+str(run))
  if name in CHILDREN and CHILDREN[name][0].poll() is not None:
   if run and (run/'outputs'/required).exists() and not active:return run
   raise RuntimeError(name+' launcher failed; see launcher log')
  if time.monotonic()-start>timeout:raise RuntimeError(name+' did not finish before supervisor deadline')
  time.sleep(30)

def monitor():
 # Retry infrastructure failures promptly across all arms, not just the arm being awaited.
 import re
 for name,(proc,_) in list(CHILDREN.items()):
  if proc.poll() is None:continue
  job,mode,timeout,artifacts=LAUNCHES[name];r=latest(name)
  current=r and r.name not in PREVIOUS_RUNS.get(name,set())
  if current and all(any((r/'outputs').glob(a)) for a in artifacts):continue
  if current and (r/'outputs/failure.json').exists():raise RuntimeError(name+' numerical worker failed; automatic retries disabled')
  if (C/'launchers'/name/'active.json').exists():continue
  if RETRIES.get(name,0)>=2:raise RuntimeError(name+' exhausted two infrastructure retries')
  log=C/(name+'-launcher.log')
  offers=re.findall(r'Offer (\d+):',log.read_text()) if log.exists() else []
  if offers:EXCLUDED.add(int(offers[-1]))
  RETRIES[name]=RETRIES.get(name,0)+1
  launch(name,job,mode,timeout,artifacts)
 rows=[];alerts=[]
 for name in ('audit','diagnostic','baseline','easy','hard','evaluation'):
  run=latest(name);l=C/'launchers'/name
  if not run:continue
  active_path=l/'active.json'
  if active_path.exists():
   active_instance=json.loads(active_path.read_text()).get('instance_id')
   matches=[]
   for candidate in (l/'runs').glob('*'):
    metadata=candidate/'metadata.json'
    if metadata.exists() and json.loads(metadata.read_text()).get('instance',{}).get('instance_id')==active_instance:matches.append(candidate)
   if not matches:
    rows.append(dict(arm=name,active=True,instance_id=active_instance,phase='booting; no matching run metadata'));continue
   run=max(matches)
  out=run/'outputs';row=dict(arm=name,run=run.name,active=active_path.exists())
  for f,key in (('sync-status.json','sync'),):
   if (run/f).exists():
    try:row[key]=json.loads((run/f).read_text())
    except ValueError:pass
  if (out/'failure.json').exists():alerts.append(name+': numerical worker failure')
  gpu=out/'gpu.csv'
  if gpu.exists():
   with gpu.open() as f:records=list(csv.DictReader(f))[-12:]
   util=[]
   for record in records:
    for k,v in record.items():
     if k.strip()=='utilization.gpu [%]':
      try:util.append(float(v.strip().split()[0]))
      except ValueError:pass
   if util:row['gpu_utilization_percent']=sum(util)/len(util)
   row['telemetry_age_seconds']=time.time()-gpu.stat().st_mtime
   if row['active'] and row['telemetry_age_seconds']>600:alerts.append(name+': stale telemetry')
  logs=list(out.glob('training/cost-*bp/lr-*/train.jsonl'))
  if logs:
   p=max(logs,key=lambda x:x.stat().st_mtime);row['trial']=str(p.parent.relative_to(out))
   for line in reversed(p.read_text().splitlines()):
    try:row['progress']=json.loads(line);break
    except ValueError:pass
  rows.append(row)
 write('monitor.latest.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),workers=rows,alerts=alerts))

def run():
 # Diagnostic and original demo can run concurrently. No training starts until both pass.
 d=latest('diagnostic')
 prior=d and (d/'outputs/diagnostic/diagnostic.json').exists() and json.loads((d/'outputs/diagnostic/diagnostic.json').read_text()).get('status')=='passed' and (d/'artifacts.json').exists() and not (C/'launchers/diagnostic/active.json').exists()
 if not prior:
  launch('diagnostic',C/'worker.py','diagnostic',3600,['diagnostic/diagnostic.json'])
  d=wait('diagnostic','diagnostic/diagnostic.json',7200)
 diagnostic=json.loads((d/'outputs/diagnostic/diagnostic.json').read_text())
 if diagnostic['status']!='passed':raise RuntimeError('Diagnostic gate failed')
 # The independently launched original reproduction must also have passed and been cleaned.
 audit_start=time.monotonic()
 while True:
  monitor();a=latest('audit')
  if a and (a/'outputs/audit.json').exists() and not (C/'launchers/audit/active.json').exists() and (a/'artifacts.json').exists():break
  if not (C/'launchers/audit/active.json').exists():raise RuntimeError('Original demo reproduction did not pass; training held')
  if time.monotonic()-audit_start>3600:raise RuntimeError('Original demo audit deadline')
  time.sleep(30)
 shutil.copy2(d/'outputs/diagnostic/diagnostic.json',C/'diagnostic.json')
 shutil.copy2(a/'outputs/audit.json',C/'audit.json')
 for arm in ('baseline','easy','hard'):launch(arm,C/'worker.py',arm,24000,['training/complete.json','training/cost-10bp/selected.eqx'])
 training={arm:wait(arm,'training/complete.json',30000) for arm in ('baseline','easy','hard')}
 payload={}
 for arm,run in training.items():
  for cost in (1,3,10):
   name=f'{arm}/cost-{cost}bp/selected.eqx';payload[name]=base64.b64encode((run/f'outputs/training/cost-{cost}bp/selected.eqx').read_bytes()).decode()
 worker=(C/'worker.py').read_text().replace('MODELS={}','MODELS='+repr(payload)).replace("evaluate.main(root/'evaluation',root/'models',{})","evaluate.main(root/'evaluation',root/'models',"+repr(diagnostic)+")")
 (C/'evaluation_worker.py').write_text(worker)
 launch('evaluation',C/'evaluation_worker.py','evaluation',7200,['evaluation/results.json','evaluation/complete.json'])
 final=wait('evaluation','evaluation/results.json',10800)
 shutil.copy2(final/'outputs/evaluation/results.json',C/'results.json')
 write('STATE.json',dict(phase='complete',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_workers_cleaned=True))
 print('VOLATILITY SCREEN COMPLETE',flush=True)

try:run()
except BaseException:
 (C/'supervisor_failure.txt').write_text(traceback.format_exc())
 # SIGTERM asks launchers to retrieve evidence and execute their finally cleanup.
 for name,(p,log) in CHILDREN.items():
  if p.poll() is None:p.terminate()
 for name,(p,log) in CHILDREN.items():
  if p.poll() is None:
   try:p.wait(timeout=180)
   except subprocess.TimeoutExpired:pass
 write('STATE.json',dict(phase='failed; inspect supervisor_failure.txt and active.json files',utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 raise
