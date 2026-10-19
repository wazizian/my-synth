"""Recover output syncing for existing detached jobs without launching or altering them."""
import datetime,fcntl,hashlib,json,shlex,subprocess,time
from pathlib import Path
C=Path(__file__).resolve().parent
lock=(C/'attach.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def write(p,obj):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def sync(arm):
 l=C/'launchers'/arm
 if not (l/'active.json').exists():return
 active=json.loads((l/'active.json').read_text())
 matching=[]
 for r in (l/'runs').glob('*'):
  m=r/'metadata.json'
  if m.exists() and json.loads(m.read_text()).get('instance',{}).get('instance_id')==active['instance_id']:matching.append(r)
 if not matching:return
 r=max(matching);m=json.loads((r/'metadata.json').read_text());ssh=m['ssh'];out=r/'outputs';out.mkdir(exist_ok=True)
 cmd=['rsync','-rlt','--safe-links','--partial','--delay-updates','--exclude=*.tmp*','--timeout=30','-e',shlex.join(ssh[:-1]),ssh[-1]+':/workspace/job/outputs/',str(out)+'/']
 with (r/'attach-sync.log').open('a') as log:result=subprocess.run(cmd,stdout=log,stderr=log,timeout=100)
 if result.returncode:return
 status=json.loads((r/'sync-status.json').read_text()) if (r/'sync-status.json').exists() else {'successes':0,'failures':0,'final_sync_complete':False}
 status['successes']+=1;status['last_success_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 if (out/'workload.exit').exists():
  status['final_sync_complete']=True
  inventory=[]
  for p in out.rglob('*'):
   if p.is_file() and '.tmp' not in p.name:inventory.append({'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()})
  write(r/'artifacts.json',inventory)
  # Output is durably retrieved before requesting provider shutdown.
  subprocess.run(['uv','run','--no-project','python',str(l/'smoke.py'),'--cleanup'],timeout=120,check=True)
 write(r/'sync-status.json',status)
def monitor():
 rows=[]
 for arm in ['baseline','easy','hard']:
  l=C/'launchers'/arm;active=l/'active.json'
  if not active.exists():continue
  identity=json.loads(active.read_text());row=dict(arm=arm,instance_id=identity['instance_id'])
  for r in sorted((l/'runs').glob('*'),reverse=True):
   meta=r/'metadata.json'
   if not meta.exists() or json.loads(meta.read_text()).get('instance',{}).get('instance_id')!=identity['instance_id']:continue
   row['run']=r.name;row['phase']='setup'
   logs=list((r/'outputs/training').glob('cost-*bp/lr-*/train.jsonl'))
   if logs:
    latest=max(logs,key=lambda f:f.stat().st_mtime);row['trial']=str(latest.parent.relative_to(r/'outputs'));row['phase']='training'
    with latest.open('rb') as stream:
     stream.seek(max(0,latest.stat().st_size-4096));lines=stream.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
     try:row['progress']=json.loads(line);break
     except ValueError:pass
   break
  rows.append(row)
 write(C/'monitor.latest.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),workers=rows,owner='option-attach.service'))
 write(C/'STATE.json',dict(phase='recovered training and development gates; durable services own jobs',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),services=['option-attach','option-finisher','option-early-gate','dh-option-pipeline','dh-comparison-watchdog']))

while True:
 for arm in ['baseline','easy','hard']:
  try:sync(arm)
  except Exception as exc:print(arm,type(exc).__name__,str(exc),flush=True)
 monitor()
 time.sleep(30)
