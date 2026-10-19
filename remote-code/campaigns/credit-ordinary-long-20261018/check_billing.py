from pathlib import Path
import json,sys,datetime,time,urllib.parse
sys.path.insert(0,'/home/jump/vast-tools');import smoke
roots=[Path('/home/jump/deep-hedging/campaigns')/n for n in ('credit-ordinary-long-20261018','credit-high-default-20261018')]
ids=set()
for root in roots:
 for p in list(root.glob('launchers/*/runs/*/metadata.json'))+list(root.glob('launchers/*/active.json')):
  d=json.loads(p.read_text());d=d.get('instance',d)
  if d.get('instance_id'):ids.add(str(d['instance_id']))
query={'select_filters':json.dumps({'day':{'gte':datetime.datetime(2026,10,18,16,tzinfo=datetime.timezone.utc).timestamp(),'lte':time.time()}}),'limit':100,'format':'table'}
rows=[]
while True:
 page=smoke.api('GET','charges?'+urllib.parse.urlencode(query));rows+=page.get('results',[])
 if not page.get('next_token'):break
 query['after_token']=page['next_token']
owned=[r for r in rows if r.get('source','').removeprefix('instance-') in ids]
active=[{k:r.get(k) for k in ('id','actual_status','gpu_name','dph_total')} for r in smoke.list_instances() if str(r.get('id')) in ids]
record=dict(queried_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),total_reported_usd=sum(r.get('amount',0) for r in owned),active=active,charges=owned,ids=sorted(ids))
(roots[0]/'combined_billing.latest.json').write_text(json.dumps(record,indent=2))
print(json.dumps({k:record[k] for k in ('queried_at','total_reported_usd','active')}))
