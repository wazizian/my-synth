"""Control-plane cleanup guard copied alongside new managed launchers."""
from pathlib import Path
import hashlib,json,shlex,subprocess,time
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
def external_required(output):
 marker=Path(output)/'recovery-requirements.json'
 if not marker.exists():return False
 obj=json.loads(marker.read_text())
 return bool(obj.get('external_history_required_until_complete',obj.get('external_history_required',False))) and not obj.get('external_history_recovery_resolved',False)
def verify_export(run,ssh,required):
 run=Path(run);out=run/'outputs'
 (run/'export-cleanup-permit.json').unlink(missing_ok=True)
 if not json.loads((run/'sync-status.json').read_text()).get('final_sync_complete'):raise RuntimeError('Final output sync is not complete')
 if external_required(out):raise RuntimeError('External selfplay history remains required; preserve worker disk')
 missing=[pattern for pattern in required if not any(p.is_file() and not p.is_symlink() for p in out.glob(pattern))]
 if missing:raise RuntimeError('Required output exports missing:'+str(missing))
 proc=subprocess.run(ssh+['python3 -c '+shlex.quote(REMOTE_MANIFEST)],capture_output=True,text=True,timeout=120,check=True)
 rows=json.loads(proc.stdout)
 if not rows:raise RuntimeError('Empty remote output inventory')
 for row in rows:
  rel=Path(row['path'])
  if rel.is_absolute() or '..' in rel.parts:raise RuntimeError('Invalid artifact path')
  p=out/rel
  if not p.is_file() or p.stat().st_size!=row['bytes']:raise RuntimeError('Missing/size mismatch:'+str(rel))
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1048576),b''):h.update(b)
  if h.hexdigest()!=row['sha256']:raise RuntimeError('SHA256 mismatch:'+str(rel))
 (run/'export-verified-manifest.json').write_text(json.dumps(rows,indent=2))
 metadata=json.loads((run/'metadata.json').read_text())
 (run/'export-cleanup-permit.json').write_text(json.dumps({'verified':True,'instance_id':metadata['instance']['instance_id'],'required_artifacts':required,'time':time.time()},indent=2))
def guarded_cleanup(smoke,expected_label=None):
 if not smoke.STATE.exists():return
 saved=json.loads(smoke.STATE.read_text())
 if expected_label is not None and saved.get('label')!=expected_label:return
 matches=[]
 for run in (smoke.ROOT/'runs').glob('*'):
  meta=run/'metadata.json'
  if meta.exists() and json.loads(meta.read_text()).get('instance',{}).get('instance_id')==saved.get('instance_id'):matches.append(run)
 run=max(matches) if matches else None
 # Workload upload/launch always follows metadata and the local intent marker.
 if run is None or not (run/'workload-launch-intent.json').exists():return smoke._unsafe_cleanup(expected_label)
 permit=run/'export-cleanup-permit.json'
 allowed=False
 try:
  obj=json.loads(permit.read_text());allowed=obj.get('verified') and obj.get('instance_id')==saved.get('instance_id') and json.loads((run/'sync-status.json').read_text()).get('final_sync_complete') and not external_required(run/'outputs')
 except (OSError,ValueError):pass
 if allowed:return smoke._unsafe_cleanup(expected_label)
 item=next((i for i in smoke.list_instances() if i.get('id')==saved.get('instance_id') and i.get('label')==saved.get('label')),None)
 if item is not None and item.get('actual_status')!='stopped':smoke.api('PUT',f"instances/{saved['instance_id']}/",{'state':'stopped'})
 remaining=next((i for i in smoke.list_instances() if i.get('id')==saved.get('instance_id') and i.get('label')==saved.get('label')),None)
 confirmed=remaining is not None and remaining.get('actual_status')=='stopped'
 alert={'state':('stopped_disk_preserved_export_unverified' if confirmed else 'stop_requested_disk_preserved_export_unverified'),'compute_stop_confirmed':confirmed,'instance_id':saved.get('instance_id'),'label':saved.get('label'),'run':str(run),'reason':'Required exports or external selfplay recovery cache are not verified','time':time.time(),'storage_charges_continue':True}
 (smoke.ROOT/'deadline-alert.json').write_text(json.dumps(alert,indent=2))
 print('Preserved worker disk; export/recovery verification is incomplete.',flush=True)
