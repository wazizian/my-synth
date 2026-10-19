"""Record an explicit metadata-only source migration between campaign phases."""
import argparse, datetime, fcntl, hashlib, json, os
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--reason');args=parser.parse_args()
ROOT=Path('/home/jump/deep-hedging')
lock=(ROOT/'queue/controller.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if list((ROOT/'workers').glob('*/active.json')): raise RuntimeError('Active workers remain')
state_path=ROOT/'queue/state.json'
state=json.loads(state_path.read_text())
if any(lease.get('status')=='running' for phase in state.get('phases',{}).values() for lease in phase.get('leases',{}).values()): raise RuntimeError('Running lease remains')
new={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ('src','scripts','tests','configs') for p in sorted((ROOT/directory).rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
old=state['source_files']
changed=[k for k in sorted(set(old)|set(new)) if old.get(k)!=new.get(k)]
allowed={'src/runner.py','scripts/run_group.py','scripts/run_campaign.py','scripts/launch_worker.py','scripts/profile.py','scripts/make_report.py'}
if not set(changed)<=allowed: raise RuntimeError('Unexpected source changes: '+str(set(changed)-allowed))
if not changed: raise RuntimeError('No source migration needed')
now=datetime.datetime.now(datetime.timezone.utc)
record={'timestamp':now.isoformat(),'reason':'Reviewed operational storage, bounded handoff, allocator telemetry, and reporting integrity improvements after diagnostics; no pricing/model/teacher/optimizer/data/accounting mathematics changes. Selfplay planned storage cap increased from8 to12GiB without changing exact lookback semantics. Final frozen pricing sensitivity and raw-array consistency are post-selection reporting only.','changed_files':changed,'old_source_files':old,'new_source_files':new,'prior_phase_statuses':{k:v.get('status') for k,v in state['phases'].items()}}
if args.reason:record['reason']=args.reason
def atomic(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('w') as f: json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
p=ROOT/'queue/source_migrations'/(now.strftime('%Y%m%dT%H%M%SZ')+'.json')
atomic(p,record)
state['source_files']=new
state.setdefault('source_migrations',[]).append(str(p.relative_to(ROOT)))
atomic(state_path,state)
print(json.dumps({'migration':str(p),'changed_files':changed}))
