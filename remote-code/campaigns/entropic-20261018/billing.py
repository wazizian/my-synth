"""Read-only provider cost and shutdown evidence for this campaign only."""
import datetime,importlib.util,json,time,urllib.parse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('provider','/home/jump/vast-tools/smoke.py')
provider=importlib.util.module_from_spec(spec);spec.loader.exec_module(provider)
ledger=json.loads((ROOT/'ledger.json').read_text())
ids={str(x['instance_id']) for x in ledger.get('failed_attempts',[]) if 'instance_id' in x}
if 'instance_id' in ledger.get('current_job',{}):ids.add(str(ledger['current_job']['instance_id']))
for x in ledger.get('additional_jobs',[]):
    if 'instance_id' in x:ids.add(str(x['instance_id']))
start=datetime.datetime.fromisoformat(ledger['started_at']).timestamp()
query={'select_filters':json.dumps({'day':{'gte':start-86400,'lte':time.time()}}),'limit':100,'format':'table'}
rows=[]
while True:
    page=provider.api('GET','charges?'+urllib.parse.urlencode(query))
    rows.extend(x for x in page.get('results',[]) if str(x.get('source','')).removeprefix('instance-') in ids)
    if not page.get('next_token'):break
    query['after_token']=page['next_token']
active=[{k:x.get(k) for k in ('id','actual_status','label')} for x in provider.list_instances() if str(x['id']) in ids and x.get('actual_status')!='destroyed']
observed={str(x.get('source','')).removeprefix('instance-') for x in rows}
record=dict(queried_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),instance_ids=sorted(ids),active_campaign_instances=active,unreported_instance_ids=sorted(ids-observed),reported_usd=sum(x.get('amount',0) for x in rows),charges=rows,scope='Provider-reported GPU/disk/bandwidth charges; rounded and may settle after shutdown')
p=ROOT/'provider_billing.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(p)
print(json.dumps({k:v for k,v in record.items() if k!='charges'},indent=2))
