"""Read-only provider accounting; no secrets emitted."""
import importlib.util,json,datetime,time,urllib.parse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('provider','/home/jump/vast-tools/smoke.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
ledger=json.loads((ROOT/'ledger.json').read_text());ids={str(x['instance_id']) for x in ledger['jobs'] if 'instance_id' in x}
start=datetime.datetime.fromisoformat(ledger['started_at']).timestamp();q={'select_filters':json.dumps({'day':{'gte':start-86400,'lte':time.time()}}),'limit':100,'format':'table'}
rows=[]
while True:
    page=p.api('GET','charges?'+urllib.parse.urlencode(q));rows.extend(x for x in page.get('results',[]) if str(x.get('source','')).removeprefix('instance-') in ids)
    if not page.get('next_token'):break
    q['after_token']=page['next_token']
active=[{k:x.get(k) for k in ['id','actual_status','label']} for x in p.list_instances() if str(x['id']) in ids and x.get('actual_status')!='destroyed']
reported={str(x.get('source','')).removeprefix('instance-') for x in rows}
r=dict(queried_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),instance_ids=sorted(ids),active_campaign_instances=active,reported_usd=sum(x.get('amount',0) for x in rows),unreported_instance_ids=sorted(ids-reported),charges=rows,scope='Provider-rounded charges may settle later')
(ROOT/'provider_billing.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='charges'},indent=2))
