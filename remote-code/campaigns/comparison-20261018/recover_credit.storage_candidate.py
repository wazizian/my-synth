"""Reattach to existing detached credit jobs; never allocate replacements."""
from pathlib import Path
import json,subprocess,shlex,time,hashlib,fcntl,os,importlib.util,threading
from controller_handoff import atomic_json
from export_safety import REMOTE_MANIFEST
from external_artifacts import selected_rows,verified_external_rows,rsync_excludes
from datetime import datetime,timezone
root=Path(__file__).resolve().parent;base=root.parent
lock=(root/'credit-recovery.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
workers=[]
def discover():
 launches=[]
 for campaign in ('credit-ordinary-long-20261018','credit-high-default-20261018'):
  launches.extend((base/campaign/'launchers').glob('*'))
 launches.extend(a.parent for a in root.rglob('active.json') if 'runs' not in a.relative_to(root).parts)
 known={w['meta']['instance']['instance_id'] for w in workers}
 for launch in launches:
  state=launch/'active.json'
  if not state.exists():continue
  try:
   active=json.loads(state.read_text())
   if active['instance_id'] in known:continue
   matches=[r for r in sorted((launch/'runs').glob('*'),reverse=True) if (r/'metadata.json').exists() and json.loads((r/'metadata.json').read_text()).get('instance',{}).get('instance_id')==active['instance_id']]
   if not matches:continue
   run=matches[0]
   requested=(run/'handoff.request.json').exists()
   if not requested and root in launch.parents and (run/'sync-status.json').exists() and time.time()-(run/'sync-status.json').stat().st_mtime<180:continue
   if not requested and root in launch.parents and not (run/'sync-status.json').exists() and time.time()-(run/'metadata.json').stat().st_mtime<180:continue
   meta=json.loads((run/'metadata.json').read_text());workers.append(dict(campaign=launch.parent.parent.name,arm=launch.name,launch=launch,run=run,meta=meta,done=False,terminal_since=None))
  except (OSError,ValueError,KeyError):continue
 (root/'recovery_manifest.json').write_text(json.dumps([dict(campaign=w['campaign'],arm=w['arm'],run=str(w['run']),instance=w['meta']['instance']) for w in workers],indent=2))
def handoff_receiver():
 while True:
  for state in root.rglob('active.json'):
   if 'runs' in state.relative_to(root).parts:continue
   try:
    saved=json.loads(state.read_text());launch=state.parent
    for request_path in (launch/'runs').glob('*/handoff.request.json'):
     run=request_path.parent
     if (run/'handoff.ack.json').exists():continue
     raw=request_path.read_bytes();request=json.loads(raw);meta=json.loads((run/'metadata.json').read_text())
     if request['instance_id']!=saved.get('instance_id') or request['label']!=saved.get('label'):continue
     if request['run_id']!=run.name or request['ssh']!=meta['ssh']:raise RuntimeError('Handoff run/SSH mismatch')
     if request['required_artifacts']!=meta.get('required_artifacts',[]):raise RuntimeError('Handoff required-artifact mismatch')
     h=hashlib.sha256()
     with (run/'submitted-job.py').open('rb') as stream:
      for block in iter(lambda:stream.read(1048576),b''):h.update(block)
     if request['source_sha256']!=h.hexdigest():raise RuntimeError('Handoff source mismatch')
     coverage=root/'watchdog.latest.json'
     if not coverage.exists() or time.time()-coverage.stat().st_mtime>120:continue
     watchdog=json.loads(coverage.read_text());row=next((r for r in watchdog['workers'] if r.get('instance_id')==request['instance_id'] and r.get('label')==request['label']),None)
     if row is None:continue
     due=datetime.fromisoformat(row['due_utc']).timestamp();deadline=datetime.fromisoformat(request['deadline_utc']).timestamp()
     if due>deadline or deadline<=time.time():raise RuntimeError('Handoff lacks bounded watchdog coverage')
     live=subprocess.run(meta['ssh']+["pgrep -af '^bash supervise-workload.sh$'"],capture_output=True,text=True,timeout=25)
     if live.returncode or not live.stdout.strip():continue
     ack=dict(request_sha256=hashlib.sha256(raw).hexdigest(),owner='dh-credit-recovery',instance_id=request['instance_id'],run_id=request['run_id'],deadline_utc=request['deadline_utc'],required_artifacts=request['required_artifacts'],accepted_utc=datetime.now(timezone.utc).isoformat())
     ownership=root/'recovery_ownership';ownership.mkdir(exist_ok=True)
     atomic_json(ownership/(str(request['instance_id'])+'.json'),dict(run=str(run),request=request,ack=ack,watchdog_coverage=row))
     atomic_json(run/'handoff.ack.json',ack)
     print('Durable handoff ACK',request['instance_id'],run.name,flush=True)
   except Exception as exc:
    print('Handoff receiver:',repr(exc),flush=True)
  time.sleep(5)
threading.Thread(target=handoff_receiver,daemon=True).start()
while True:
 discover()
 for w in workers:
  if w['done']:continue
  if not (w['launch']/'active.json').exists():w['done']=True;continue
  ssh=w['meta']['ssh'];run=w['run'];status={'utc':datetime.now(timezone.utc).isoformat(),'instance_id':w['meta']['instance']['instance_id']}
  try:
   p=subprocess.run(ssh+['if test -f /workspace/job/outputs/workload.exit; then cat /workspace/job/outputs/workload.exit; else pgrep -f "^bash supervise-workload.sh$" >/dev/null && echo RUNNING; fi'],capture_output=True,text=True,timeout=25)
   remote=p.stdout.strip().splitlines()[-1] if p.stdout.strip() else 'UNKNOWN';status['remote']=remote
   terminal_remote=remote.lstrip('-').isdigit()
   exclusions=[];exported=set()
   has_histories=any(pattern.startswith('histories/') for pattern in w['meta'].get('required_artifacts',[]))
   if has_histories and (not terminal_remote or remote!='0'):
    exclusions=['--exclude=/histories/*.tar.gz']
    atomic_json(run/'pending-deferred-artifacts.json',dict(state='deferred_not_verified',patterns=['histories/*.tar.gz'],source_retained=True,final_export_complete=False,utc=status['utc']))
    status['deferred_artifact_patterns']=['histories/*.tar.gz']
   if has_histories and terminal_remote and remote=='0':
    inventory_result=subprocess.run(ssh+['python3 -c '+shlex.quote(REMOTE_MANIFEST)],capture_output=True,text=True,timeout=120,check=True)
    remote_inventory=json.loads(inventory_result.stdout);selected=selected_rows(remote_inventory)
    external=verified_external_rows(run,remote_inventory)
    exported={row['path'] for row in external}
    if any(row['path'] not in exported for row in selected):
     request=run/'external-export-request.json'
     if not request.exists():atomic_json(request,dict(status='pending',run_id=run.name,instance_id=w['meta']['instance']['instance_id'],rows=selected,requested_utc=status['utc']))
     status['external_export_pending']=True;status['selected_external_paths']=[row['path'] for row in selected]
     atomic_json(run/'recovery-status.json',status);print(w['campaign'],w['arm'],status,flush=True)
     continue
    if external:exclusions=rsync_excludes(run)
   cmd=['rsync','-rlt','--safe-links','--partial','--delay-updates','--exclude=*.tmp*','--timeout=45','-e',shlex.join(ssh[:-1]),ssh[-1]+':/workspace/job/outputs/',str(run/'outputs')+'/']
   cmd[1:1]=exclusions
   with (run/'recovery-sync.log').open('a') as log:sync=subprocess.run(cmd,stdout=log,stderr=log,timeout=120)
   status['sync_rc']=sync.returncode
   if sync.returncode==0:
    old=json.loads((run/'sync-status.json').read_text()) if (run/'sync-status.json').exists() else {'successes':0,'failures':0,'final_sync_complete':False};old.update(last_success_utc=status['utc'],recovery_owner=os.getpid());old['successes']+=1
    terminal=remote.lstrip('-').isdigit()
    if terminal and (run/'export-hold.json').exists():
     hold=json.loads((run/'export-hold.json').read_text());created=datetime.fromisoformat(hold['created_utc']).timestamp();expires=min(created+180,datetime.fromisoformat(hold['expires_utc']).timestamp()) if hold.get('expires_utc') else created+180
     if time.time()<expires:status['export_hold']=hold;terminal=False
     else:status['export_hold_expired']=hold

    if terminal:
     old['final_sync_complete']=True
     inventory=[]
     for file in sorted((run/'outputs').rglob('*')):
      if file.is_file() and not file.is_symlink() and '.tmp' not in file.name:
       h=hashlib.sha256()
       with file.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
       inventory.append({'path':str(file.relative_to(run/'outputs')),'bytes':file.stat().st_size,'sha256':h.hexdigest()})
     (run/'artifacts.json').write_text(json.dumps(inventory,indent=2))
     expected=('gate/complete.json','gate/results.json') if '-gate-' in str(w['launch']) else (('evaluation/complete.json','evaluation/results.json') if w['arm']=='evaluation' else ('complete.json','cost-10bp/learned/selected.eqx'))
     expected=tuple(w['meta'].get('required_artifacts') or expected)
     status['complete_artifacts']=all(any(p.is_file() and not p.is_symlink() for p in (run/'outputs').glob(n)) or any(Path(name).match(n) for name in exported) for n in expected)
     atomic_json(run/'sync-status.json',old)
     if (w['launch']/'export_safety.py').exists():
      spec=importlib.util.spec_from_file_location('recovery_export_safety',w['launch']/'export_safety.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
      try:module.verify_export(run,ssh,w['meta'].get('required_artifacts',list(expected)))
      except Exception as exc:status['export_verification_error']=str(exc)
      if (run/'export-verified-manifest.json').exists() and 'export_verification_error' not in status:atomic_json(run/'artifacts.json',json.loads((run/'export-verified-manifest.json').read_text()))
     with (run/'recovery-cleanup.log').open('a') as log:cleanup=subprocess.run(['uv','run','--no-project','python',str(w['launch']/'smoke.py'),'--cleanup'],stdout=log,stderr=log,timeout=120)
     status['cleanup_rc']=cleanup.returncode
     w['done']=cleanup.returncode==0 and not (w['launch']/'active.json').exists()
    (run/'sync-status.json').write_text(json.dumps(old,indent=2))
  except Exception as e:status['error']=repr(e)
  (run/'recovery-status.json').write_text(json.dumps(status,indent=2));print(w['campaign'],w['arm'],status,flush=True)
 time.sleep(30)
