"""Wait for the bounded gate result, then audit on its existing GPU worker."""
import json,subprocess,shlex,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent
run=Path('/home/jump/vast-tools/runs/20261018T090952Z-5439b23c')
metadata=json.loads((run/'metadata.json').read_text())
code='''import pathlib,json,time,os
p=pathlib.Path('/workspace/job/outputs/gate_summary.json')
while True:
    if p.exists():
        try:
            s=json.loads(p.read_text())
            if s.get('accepted') or len(s.get('results',[]))==2:break
        except (OSError,ValueError):pass
    time.sleep(1)
os.execv('/opt/jax/bin/python',['/opt/jax/bin/python','/workspace/job/audit_numeric.py','/workspace/job/outputs'])
'''
r=subprocess.run(metadata['ssh']+['/opt/jax/bin/python -c '+shlex.quote(code)],capture_output=True,text=True,timeout=1500)
(ROOT/'numeric_audit_final.stderr.txt').write_text(r.stderr)
if r.stdout:
    (ROOT/'numeric_audit_final.json').write_text(r.stdout)
    try:
        report=json.loads(r.stdout);print(json.dumps({k:v for k,v in report.items() if k!='datasets'},indent=2))
    except ValueError:print(r.stdout[-2000:])
(ROOT/'numeric_audit_execution.json').write_text(json.dumps(dict(returncode=r.returncode,completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),worker_instance=56529076),indent=2))
raise SystemExit(r.returncode)
