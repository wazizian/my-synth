"""Durable deadline watchdog: stop GPU spend, verify exports before destruction."""
from pathlib import Path
import datetime,fcntl,hashlib,json,shlex,subprocess,sys,time
C=Path(__file__).resolve().parent;BASE=C.parent
sys.path.insert(0,'/home/jump/vast-tools')
import smoke
from external_artifacts import selected_rows,verified_external_rows,rsync_excludes
DEADLINE=datetime.datetime(2026,10,19,6,5,tzinfo=datetime.timezone.utc).timestamp()
lock=(C/'watchdog.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
REMOTE_MANIFEST="""import pathlib,json,hashlib
root=pathlib.Path('/workspace/job/outputs');rows=[]
for p in sorted(root.rglob('*')):
 if p.is_file() and not p.is_symlink() and '.tmp' not in p.name:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1048576),b''):h.update(b)
  rows.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
print(json.dumps(rows))
"""
def atomic(path,value):
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def launches():
 roots=[BASE/'credit-ordinary-long-20261018',BASE/'credit-high-default-20261018',BASE/'volatility-shocks-20261018',C]
 return sorted({a.parent for root in roots for a in root.rglob('active.json') if 'runs' not in a.relative_to(root).parts})
def find_run(launch,saved):
 matches=[]
 for r in (launch/'runs').glob('*'):
  m=r/'metadata.json'
  if m.exists():
   v=json.loads(m.read_text())
   if v.get('instance',{}).get('instance_id')==saved.get('instance_id'):matches.append((r,v))
 return max(matches,key=lambda x:x[0].name) if matches else (None,None)
def identity(saved,instances):
 return next((i for i in instances if i.get('id')==saved.get('instance_id') and i.get('label')==saved.get('label')),None)
def external_required(output):
 marker=Path(output)/'recovery-requirements.json'
 if not marker.exists():return False
 obj=json.loads(marker.read_text())
 return bool(obj.get('external_history_required_until_complete',obj.get('external_history_required',False))) and not obj.get('external_history_recovery_resolved',False)
def export_verified(run,meta):
 ssh=meta['ssh'];out=run/'outputs';out.mkdir(exist_ok=True)
 # Freeze workload and its telemetry before taking a content manifest.
 subprocess.run(ssh+["pkill -TERM -f '^/opt/jax/bin/python' || true; pkill -TERM -f '^nvidia-smi --query-gpu=' || true"],capture_output=True,timeout=30)
 r=subprocess.run(ssh+['python3 -c '+shlex.quote(REMOTE_MANIFEST)],capture_output=True,text=True,timeout=90,check=True)
 rows=json.loads(r.stdout)
 if not rows:raise RuntimeError('Empty remote output manifest; preserve disk')
 external=verified_external_rows(run,rows)
 external_paths={row['path'] for row in external}
 selected=selected_rows(rows)
 if any(row['path'] not in external_paths for row in selected):
  terminal=subprocess.run(ssh+['cat /workspace/job/outputs/workload.exit'],capture_output=True,text=True,timeout=25)
  if terminal.returncode==0 and terminal.stdout.strip()=='0':
   atomic(run/'external-export-request.json',{'instance_id':meta['instance']['instance_id'],'rows':selected})
  raise RuntimeError('Large archives lack verified external export; preserve disk pending coordinator')
 cmd=['rsync','-rlt','--safe-links','--partial','--delay-updates','--exclude=*.tmp*',*rsync_excludes(run),'--timeout=30','-e',shlex.join(ssh[:-1]),ssh[-1]+':/workspace/job/outputs/',str(out)+'/']
 with (run/'deadline-export.log').open('a') as log:r=subprocess.run(cmd,stdout=log,stderr=log,timeout=120)
 if r.returncode:raise RuntimeError('Bounded export failed rsync='+str(r.returncode))
 if external_required(out):raise RuntimeError('External selfplay history required until recovery explicitly resolves; preserve disk')
 missing=[pattern for pattern in meta.get('required_artifacts',[]) if not any(p.is_file() and not p.is_symlink() for p in out.glob(pattern)) and not any(Path(name).match(pattern) for name in external_paths)]
 if missing:raise RuntimeError('Required final artifacts missing; preserve disk:'+str(missing))
 for row in rows:
  rel=Path(row['path'])
  if rel.is_absolute() or '..' in rel.parts:raise RuntimeError('Invalid remote manifest path')
  if row['path'] in external_paths:continue
  file=out/rel
  if not file.is_file() or file.stat().st_size!=row['bytes']:raise RuntimeError('Missing/size mismatch:'+str(rel))
  h=hashlib.sha256()
  with file.open('rb') as f:
   for b in iter(lambda:f.read(1048576),b''):h.update(b)
  if h.hexdigest()!=row['sha256']:raise RuntimeError('Hash mismatch:'+str(rel))
 atomic(run/'deadline-export-manifest.json',rows)
 atomic(run/'artifacts.json',rows)
 atomic(run/'deadline-export-verified.json',{'verified':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':len(rows),'bytes':sum(r['bytes'] for r in rows),'external_receipt_sha256':hashlib.sha256((run/'external-artifacts.json').read_bytes()).hexdigest() if external else None})
 return True
while True:
 try:
  instances=smoke.list_instances();now=time.time();states=[]
  for launch in launches():
   state=launch/'active.json'
   if not state.exists():continue
   saved=json.loads(state.read_text());run,meta=find_run(launch,saved)
   due=DEADLINE
   if run:
    start=datetime.datetime.strptime(run.name[:16],'%Y%m%dT%H%M%SZ').replace(tzinfo=datetime.timezone.utc).timestamp();due=min(due,start+meta['timeout_seconds']+1200)
   else:due=min(due,state.stat().st_mtime+1200)
   status={'launch':str(launch),'instance_id':saved.get('instance_id'),'label':saved.get('label'),'due_utc':datetime.datetime.fromtimestamp(due,datetime.timezone.utc).isoformat()}
   if now<due:states.append(status);continue
   ilock=(launch/'deadline-cleanup.lock').open('w')
   try:fcntl.flock(ilock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:continue
   try:
    item=identity(saved,instances)
    if item is None:
     same_id=next((i for i in instances if i.get('id')==saved.get('instance_id')),None)
     if same_id is None:
      status['state']='provider_absent_confirmed'
      if state.exists() and json.loads(state.read_text()).get('instance_id')==saved['instance_id']:state.unlink()
     else:status['state']='identity_mismatch_no_action'
     states.append(status);continue
    verified=False
    if run and (run/'deadline-export-verified.json').exists() and not external_required(run/'outputs'):
     proof=json.loads((run/'deadline-export-verified.json').read_text())
     receipt=proof.get('external_receipt_sha256')
     verified=not receipt or (run/'external-artifacts.json').exists() and hashlib.sha256((run/'external-artifacts.json').read_bytes()).hexdigest()==receipt
     if verified and receipt:
      try:verified_external_rows(run,json.loads((run/'deadline-export-manifest.json').read_text()))
      except Exception as exc:
       verified=False;status['external_revalidation_error']=str(exc)[:500]
    # Try at most once each five minutes if an earlier export was unavailable.
    alert=launch/'deadline-alert.json';previous=json.loads(alert.read_text()) if alert.exists() else {}
    if not verified and now-previous.get('last_attempt',0)>=300:
     try:
      if run and meta:verified=export_verified(run,meta)
      else:raise RuntimeError('Missing SSH/run metadata; preserve instance disk')
     except Exception as e:
      status['export_error']=str(e)[:500];status['last_attempt']=now
    if verified:
     # Verify identity again directly before irreversible destruction.
     live=identity(saved,smoke.list_instances())
     if live is not None:smoke.api('DELETE',f"instances/{saved['instance_id']}/")
     remaining=identity(saved,smoke.list_instances())
     if remaining is None or remaining.get('actual_status')=='destroyed':
      if state.exists() and json.loads(state.read_text()).get('instance_id')==saved['instance_id']:state.unlink()
      status['state']='export_verified_destroyed_confirmed'
     else:status['state']='destruction_pending'
    else:
     # Stopping preserves disk. NEVER destroy an instance with an unverified sole artifact copy.
     if item.get('actual_status')!='stopped':smoke.api('PUT',f"instances/{saved['instance_id']}/",{'state':'stopped'})
     stopped=identity(saved,smoke.list_instances());confirmed=stopped is not None and stopped.get('actual_status')=='stopped'
     status['state']='stopped_disk_preserved_export_unverified' if confirmed else 'stop_requested_disk_preserved_export_unverified';status['compute_stop_confirmed']=confirmed;status['last_attempt']=status.get('last_attempt',previous.get('last_attempt',now));status['storage_charges_continue']=True
    atomic(alert,status);states.append(status);print(json.dumps(status),flush=True)
   finally:ilock.close()
  atomic(C/'watchdog.latest.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_utc':'2026-10-19T06:05:00+00:00','workers':states})
  ledger=C/'credit-instance-ledger.json'
  if not ledger.exists() or time.time()-ledger.stat().st_mtime>900:
   from update_credit_ledger import update
   update()
 except Exception as e:
  atomic(C/'watchdog-error.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'error':str(e)[:1000]});print('Watchdog error:',str(e),flush=True)
 time.sleep(30)
