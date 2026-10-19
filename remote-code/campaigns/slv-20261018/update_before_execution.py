"""Atomic worker source update, only while dependency setup is still running.
Preserve original submitted source and exact new source/provenance in local run.
No scientific parameter or threshold edits.
"""
from pathlib import Path
import json,subprocess,hashlib,datetime
root=Path(__file__).resolve().parent;run=Path('/home/jump/vast-tools/runs/20261018T110514Z-84bd4f5b')
metadata=json.loads((run/'metadata.json').read_text());source=(root/'worker_full.py').read_bytes()
original=hashlib.sha256((run/'submitted-job.py').read_bytes()).hexdigest();new=hashlib.sha256(source).hexdigest()
history=[]
if (run/'source_update.json').exists():
    previous=json.loads((run/'source_update.json').read_text());history=previous.get('history',[])+[{k:v for k,v in previous.items() if k!='history'}]
    folder=run/'source_revisions';folder.mkdir(exist_ok=True);(folder/(previous['actual_source_sha256']+'.py')).write_bytes((run/'actual-job.py').read_bytes())
record=dict(history=history,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),instance_id=56541866,original_submitted_sha256=original,actual_source_sha256=new,reason='Before execution: validate both frozen families before any diagnostic training, preventing a later numerical failure from requiring a partial campaign rerun. Scientific design and thresholds unchanged.',guard='Refuse update if preflight/result.json already exists; installation log still incomplete')
command='set -eu; cd /workspace/job; test ! -e outputs/preflight/result.json; cat > job.py.tmp; test ! -e outputs/preflight/result.json; mv job.py.tmp job.py; sha256sum job.py'
result=subprocess.run(metadata['ssh']+[command],input=source,capture_output=True,timeout=60,check=True)
assert new in result.stdout.decode(),result.stdout
(run/'actual-job.py').write_bytes(source);(run/'source_update.json').write_text(json.dumps(record,indent=2));(root/'source_update.json').write_text(json.dumps(record,indent=2))
subprocess.run(metadata['ssh']+['cat > /workspace/job/outputs/source_update.json'],input=json.dumps(record,indent=2).encode(),capture_output=True,timeout=30,check=True)
p=root/'ledger.json';d=json.loads(p.read_text());d['jobs'][-1]['original_source_sha256']=original;d['jobs'][-1]['source_sha256']=new;d['jobs'][-1]['source_updated_before_execution']=True;p.write_text(json.dumps(d,indent=2))
print(json.dumps(record,indent=2))
