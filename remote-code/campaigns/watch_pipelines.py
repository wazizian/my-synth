"""Read-only status for credit and volatility, including booting instances."""
from pathlib import Path
import datetime,json,time,csv
ROOT=Path(__file__).resolve().parent
NAMES=('credit-ordinary-long-20261018','credit-high-default-20261018','volatility-shocks-20261018')
while True:
 rows=[];alerts=[]
 for name in NAMES:
  c=ROOT/name
  if (c/'supervisor_failure.txt').exists():alerts.append(name+': supervisor failed')
  for l in (c/'launchers').glob('*'):
   if not l.is_dir():continue
   active=json.loads((l/'active.json').read_text()) if (l/'active.json').exists() else None
   runs=sorted((l/'runs').glob('*'));r=runs[-1] if runs else None
   row=dict(campaign=name,arm=l.name,active=bool(active))
   if active:
    row['instance_id']=active['instance_id'];row['listed_hourly']=active.get('listed_hourly')
    if not r or not (r/'metadata.json').exists() or json.loads((r/'metadata.json').read_text())['instance']['instance_id']!=active['instance_id']:
     row['stage']='booting';rows.append(row);continue
   if not r:continue
   row['run']=r.name;out=r/'outputs';row['stage']='setup' if active else 'stopped'
   if (r/'sync-status.json').exists():
    try:row['sync']=json.loads((r/'sync-status.json').read_text())
    except ValueError:pass
   log=out/'workload.log'
   if log.exists():row['stage']='validation or initialization' if active else row['stage']
   if (out/'failure.json').exists():alerts.append(name+'/'+l.name+': worker failure')
   logs=list(out.glob('cost-*bp/learned/lr-*/train.jsonl'))+list(out.glob('training/cost-*bp/lr-*/train.jsonl'))
   # Restored historical train logs are excluded from the upload; this detects new updates.
   if logs:
    p=max(logs,key=lambda x:x.stat().st_mtime);row['trial']=str(p.parent.relative_to(out))
    for line in reversed(p.read_text().splitlines()):
     try:row['last_update']=json.loads(line);break
     except ValueError:pass
    row['training_age_seconds']=round(time.time()-p.stat().st_mtime)
    if active:row['stage']='training'
   if (out/'diagnostic/diagnostic.json').exists() and json.loads((out/'diagnostic/diagnostic.json').read_text()).get('status')=='passed':row['stage']='complete'
   if (out/'audit.json').exists() and json.loads((out/'audit.json').read_text()).get('status')=='passed':row['stage']='complete'
   if (out/'complete.json').exists() or (out/'training/complete.json').exists() or (out/'evaluation/complete.json').exists():row['stage']='complete'
   gpu=out/'gpu.csv'
   if gpu.exists():
    with gpu.open() as f:values=list(csv.DictReader(f))[-12:]
    vals=[]
    for v in values:
     for k,a in v.items():
      if k.strip()=='utilization.gpu [%]':
       try:vals.append(float(a.strip().split()[0]))
       except ValueError:pass
    if vals:row['gpu_utilization_percent']=round(sum(vals)/len(vals),1)
    row['telemetry_age_seconds']=round(time.time()-gpu.stat().st_mtime)
    if active and row['telemetry_age_seconds']>600:alerts.append(name+'/'+l.name+': stale telemetry')
   if not active and row['stage']!='complete' and l.name not in ('audit',):alerts.append(name+'/'+l.name+': stopped without completion')
   rows.append(row)
 snapshot=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),workers=rows,alerts=alerts)
 p=ROOT/'live_monitor.latest.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(snapshot,indent=2));tmp.replace(p)
 with (ROOT/'live_monitor.history.jsonl').open('a') as f:f.write(json.dumps(snapshot)+'\n')
 print(json.dumps(dict(utc=snapshot['utc'],active=sum(r['active'] for r in rows),training=sum(r['stage']=='training' and r['active'] for r in rows),alerts=alerts)),flush=True)
 if all((ROOT/n/'results.json').exists() for n in NAMES):break
 time.sleep(30)
