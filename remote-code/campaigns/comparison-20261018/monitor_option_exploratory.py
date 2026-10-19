"""Read-only bounded SSH/control metadata snapshot; no numerical imports or workloads."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
NAMESPACE=ROOT/'option-exploratory-20261019'
ARMS=('baseline','easy_reference','hard_reference','dpg_hard','soar_hard','selfplay','dpg_mixed','soar_mixed')
REMOTE=r'''import datetime,json,pathlib,subprocess
root=pathlib.Path('/workspace/job/outputs');rows=[]
for trial in sorted((root/'training').glob('lr-*')):
 row={'trial':trial.name};log=trial/'training.jsonl'
 if log.exists():
  row['training_log_bytes']=log.stat().st_size
  with log.open('rb') as stream:
   stream.seek(0,2);end=stream.tell();stream.seek(max(0,end-16384));lines=stream.read().splitlines()
  for line in reversed(lines):
   try:
    value=json.loads(line);row['progress']={k:value[k] for k in ['active_seconds','outer_round','learner_updates'] if k in value};break
   except ValueError:pass
 if (trial/'complete.json').exists():row['complete']=True
 rows.append(row)
gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
supervisor=subprocess.run(['pgrep','-af','^bash supervise-workload.sh$'],capture_output=True,text=True,timeout=10)
exit_path=root/'workload.exit'
print(json.dumps({'worker_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'trials':rows,'gpu':gpu.stdout.strip(),'supervisor':supervisor.stdout.strip(),'workload_exit':exit_path.read_text().strip() if exit_path.exists() else None,'overall_complete':(root/'complete.json').exists()}))
'''

def main():
    result={'utc':datetime.now(timezone.utc).isoformat(),'namespace':str(NAMESPACE),
            'host_free_bytes':shutil.disk_usage(ROOT).free,'host_free_floor_bytes':300_000_000,'workers':[]}
    for arm in ARMS:
        launch=NAMESPACE/arm/'launcher';record={'arm':arm};active=launch/'active.json'
        if not active.exists():
            record['state']='not_allocated_or_already_cleaned';result['workers'].append(record);continue
        state=json.loads(active.read_text());record.update(instance_id=state['instance_id'],state='allocated_waiting_metadata')
        metadata=None
        for run in sorted((launch/'runs').glob('*'),reverse=True):
            path=run/'metadata.json'
            if path.exists():
                candidate=json.loads(path.read_text())
                if candidate['instance']['instance_id']==state['instance_id']:
                    metadata=candidate;record['run_id']=run.name;break
        if metadata:
            try:
                response=subprocess.run(metadata['ssh']+['python3 -c '+shlex.quote(REMOTE)],capture_output=True,text=True,timeout=40)
                if response.returncode:raise RuntimeError(response.stderr[-500:])
                record.update(state='ssh_verified',**json.loads(response.stdout.strip().splitlines()[-1]))
            except Exception as error:record.update(state='reconnect_needed_not_failure',error=repr(error))
        result['workers'].append(record)
    path=ROOT/'option-exploratory-health-latest.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(path)
    print(json.dumps(result))

if __name__=='__main__':main()
