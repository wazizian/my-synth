"""Filesystem and mocked-provider safety checks; execute only on a GPU worker."""
import json,pathlib,tempfile,types,hashlib,subprocess
import export_safety as safety
from harden_export import harden

def main():
 checks=[]
 with tempfile.TemporaryDirectory() as tmp:
  root=pathlib.Path(tmp);state=root/'active.json';saved={'instance_id':123,'label':'test'};state.write_text(json.dumps(saved));calls=[]
  fake=types.SimpleNamespace(ROOT=root,STATE=state,list_instances=lambda:[{'id':123,'label':'test','actual_status':'running'}],api=lambda *a:calls.append(a),_unsafe_cleanup=lambda *a:calls.append(('destroy',)))
  safety.guarded_cleanup(fake);assert calls==[('destroy',)];checks.append('premetadata_boot_failure_allows_cleanup');calls.clear()
  run=root/'runs/20261018T000000Z-test';out=run/'outputs';out.mkdir(parents=True);(run/'metadata.json').write_text(json.dumps({'instance':saved}));safety.guarded_cleanup(fake);assert calls==[('destroy',)];checks.append('prelaunch_setup_failure_allows_cleanup');calls.clear()
  (run/'workload-launch-intent.json').write_text('{}');safety.guarded_cleanup(fake);assert calls[0][0]=='PUT' and calls[0][2]=={'state':'stopped'} and not any(c[0]=='destroy' for c in calls);assert state.exists() and not json.loads((root/'deadline-alert.json').read_text())['compute_stop_confirmed'];checks.append('unverified_workload_requests_stop_preserves_disk_without_false_confirmation');calls.clear()
  (run/'sync-status.json').write_text(json.dumps({'final_sync_complete':True}));(out/'required.json').write_text('{}');data=(out/'required.json').read_bytes();rows=[{'path':'required.json','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}]
  original=safety.subprocess.run;safety.subprocess.run=lambda *a,**k:types.SimpleNamespace(stdout=json.dumps(rows))
  try:
   safety.verify_export(run,[],['required.json']);safety.guarded_cleanup(fake);assert calls==[('destroy',)];checks.append('verified_required_hashes_allow_cleanup');calls.clear()
   (out/'recovery-requirements.json').write_text(json.dumps({'external_history_required_until_complete':True}));safety.guarded_cleanup(fake);assert calls[0][0]=='PUT' and not any(c[0]=='destroy' for c in calls);checks.append('external_history_overrides_export_permission');calls.clear()
   try:safety.verify_export(run,[],['required.json'])
   except RuntimeError:pass
   else:raise AssertionError('Must preserve incomplete external history')
   (out/'recovery-requirements.json').write_text(json.dumps({'external_history_required_until_complete':False}));rows[0]['sha256']='bad'
   try:safety.verify_export(run,[],['required.json'])
   except RuntimeError:pass
   else:raise AssertionError('Must reject hash mismatch')
   assert not (run/'export-cleanup-permit.json').exists()
   checks.append('hash_mismatch_rejected_and_stale_permission_removed')
  finally:safety.subprocess.run=original
  fixture=pathlib.Path(__file__).resolve().parent/'fixture';harden(fixture)
  for name in ('job.py','smoke.py','export_safety.py'):compile((fixture/name).read_text(),name,'exec')
  job=(fixture/'job.py').read_text();smoke=(fixture/'smoke.py').read_text();assert 'verify_export(run_dir, ssh, required)' in job and 'workload-launch-intent' in job and 'subprocess.Popen([sys.executable' not in smoke;checks.append('fresh_launcher_patch_compiles_and_disables_unsafe_watchdog')
 return {'passed':True,'checks':checks,'provider_operations':'mocked; no live instance changed'}
if __name__=='__main__':print(json.dumps(main()))
