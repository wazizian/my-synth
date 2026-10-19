"""Read small campaign status JSON/JSONL only; no numerical imports."""
from pathlib import Path
import datetime,json,shutil
ROOT=Path('/home/jump/deep-hedging')
print(datetime.datetime.now(datetime.timezone.utc).isoformat(), 'free_GiB',round(shutil.disk_usage(ROOT).free/1024**3,2))
state=json.loads((ROOT/'queue/state.json').read_text())
for phase,record in state.get('phases',{}).items():
 for name,lease in record.get('leases',{}).items():
  if lease.get('status')!='running':continue
  projects=sorted((ROOT/'workers'/name/'runs').glob('*/outputs/project'))
  print(name,'lease',lease['status'])
  if not projects:continue
  project=projects[-1]
  if phase=='tests':
   group_path=project/'group_status.json'
   if group_path.exists():
    group=json.loads(group_path.read_text())
    counts={}
    for checkpoint in group.get('checkpoints',[]):
     key=checkpoint.get('status','unknown');counts[key]=counts.get(key,0)+1
    print('evaluation',group.get('status'),'checkpoint_statuses',counts,'completed_runs',len(group.get('completed_runs',[])),'failures',group.get('failures',[]))
   continue
  for run in lease['run_ids']:
   path=project/'runs'/run
   status=json.loads((path/'status.json').read_text()) if (path/'status.json').exists() else {}
   if status.get('status')=='completed':continue
   file=path/'train_metrics.jsonl'
   if file.exists():
    with file.open('rb') as f:
     f.seek(max(0,file.stat().st_size-18000)); lines=f.read().decode().splitlines()
    rows=[]
    for line in lines:
     try:rows.append(json.loads(line))
     except json.JSONDecodeError:pass
    if rows:
     d=next((row for row in reversed(rows) if row.get('event_type') in ('training','training_terminal')),rows[-1])
     print(run,{k:d.get(k) for k in ('active_training_seconds','learner_steps','state_finite','teacher_grad_norm','learner_loss_nonfinite_count')},'status',status.get('status'))
   elif status:print(run,status.get('status'),status.get('error'))
   if status.get('status') in ('failed','stopped'):print('ATTENTION',run,status)
replacement=Path('/home/jump/campaign-control/extensions-retry0.json')
if replacement.exists():
 lease=json.loads(replacement.read_text());worker=ROOT/'workers'/lease['replacement_worker']
 if (worker/'active.json').exists():
  print(lease['replacement_worker'],'replacement lease active')
  projects=sorted((worker/'runs').glob('*/outputs/project'))
  if projects:
   for run in lease['run_ids']:
    p=projects[-1]/'runs'/run/'status.json'
    if p.exists():
     d=json.loads(p.read_text())
     if d.get('status')=='completed':continue
     print(run,{k:d.get(k) for k in ('status','active_training_seconds','learner_steps','error')})

record_path=Path('/home/jump/campaign-control/tests-retry0.json')
if record_path.exists():
 record=json.loads(record_path.read_text());worker=ROOT/'workers'/record['replacement_worker']
 if (worker/'active.json').exists():
  print(record['replacement_worker'],'replacement evaluation lease active')
  groups=sorted((worker/'runs').glob('*/outputs/project/group_status.json'))
  if groups:
   group=json.loads(groups[-1].read_text());counts={}
   for row in group.get('checkpoints',[]):
    key=row.get('status','unknown');counts[key]=counts.get(key,0)+1
   print('evaluation',group.get('status'),'checkpoint_statuses',counts,'completed_runs',len(group.get('completed_runs',[])),'failures',group.get('failures',[]))
