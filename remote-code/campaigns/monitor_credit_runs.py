"""Read-only control-plane monitoring; no numerical work or instance mutations."""
from pathlib import Path
import json,time,csv,datetime,subprocess
ROOT=Path(__file__).resolve().parent
CAMPAIGNS=[ROOT/'credit-ordinary-long-20261018',ROOT/'credit-high-default-20261018']
last_billing=0
while True:
 now=time.time(); rows=[]; alerts=[]
 for campaign in CAMPAIGNS:
  if (campaign/'supervisor_failure.txt').exists(): alerts.append(f'{campaign.name}: supervisor failure')
  for arm in ('baseline','easy','hard','evaluation'):
   launcher=campaign/'launchers'/arm
   runs=sorted((launcher/'runs').glob('*'))
   if not runs: continue
   run=runs[-1]; out=run/'outputs'
   item={'campaign':campaign.name,'arm':arm,'run':run.name,'active':(launcher/'active.json').exists()}
   sync=run/'sync-status.json'
   if sync.exists(): item['sync']=json.loads(sync.read_text())
   logs=list(out.glob('cost-*bp/learned/lr-*/train.jsonl'))
   if logs:
    log=max(logs,key=lambda p:p.stat().st_mtime)
    lines=log.read_text().splitlines(); records=[]
    for line in lines[-100:]:
     try: records.append(json.loads(line))
     except ValueError: pass
    item['trial']=str(log.parent.relative_to(out));item['last_training']=records[-1] if records else None
    item['training_age_seconds']=round(now-log.stat().st_mtime)
   gpu=out/'gpu.csv'
   if gpu.exists():
    with gpu.open() as f: samples=[{k.strip():v.strip() for k,v in r.items()} for r in csv.DictReader(f)][-120:]
    for field in ('utilization.gpu [%]','power.draw [W]','memory.used [MiB]'):
     vals=[]
     for r in samples:
      try: vals.append(float(r[field].split()[0]))
      except (KeyError,ValueError): pass
     if vals:item[field]=round(sum(vals)/len(vals),1)
    item['telemetry_age_seconds']=round(now-gpu.stat().st_mtime)
    if item['active'] and item['telemetry_age_seconds']>600:alerts.append(f'{campaign.name}/{arm}: telemetry stale >10 minutes')
   if item['active'] and item.get('sync',{}).get('failures',0):alerts.append(f'{campaign.name}/{arm}: sync reports failures')
   if not item['active'] and not (out/'complete.json').exists() and arm!='evaluation':alerts.append(f'{campaign.name}/{arm}: stopped without completion')
   rows.append(item)
 snapshot={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'workers':rows,'alerts':alerts}
 target=ROOT/'credit_monitor.latest.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(snapshot,indent=2));tmp.replace(target)
 with (ROOT/'credit_monitor.history.jsonl').open('a') as f:f.write(json.dumps(snapshot)+'\n')
 print(json.dumps({'utc':snapshot['utc'],'active':sum(r['active'] for r in rows),'alerts':alerts}),flush=True)
 if now-last_billing>=300:
  try:subprocess.run(['uv','run','--no-project','python',str(CAMPAIGNS[0]/'check_billing.py')],timeout=90,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  except Exception as e:print(f'billing check: {e}',flush=True)
  last_billing=now
 if all((c/'results.json').exists() for c in CAMPAIGNS):print('Both evaluations complete',flush=True);break
 time.sleep(60)
