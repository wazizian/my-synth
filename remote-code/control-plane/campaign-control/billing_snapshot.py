"""Read-only provider billing snapshot; credentials never enter saved output."""
import argparse,datetime,importlib.util,json,time,urllib.parse
from pathlib import Path
ROOT=Path('/home/jump/deep-hedging')
p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args()
spec=importlib.util.spec_from_file_location('vast_control','/home/jump/vast-tools/smoke.py')
api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
known={}
for path in (ROOT/'workers').glob('*/rental_history.json'):
    row=json.loads(path.read_text());known[str(row['instance_id'])]=path.parent.name
for path in (ROOT/'workers').glob('*/runs/*/metadata.json'):
    row=json.loads(path.read_text());instance=row.get('instance',{}).get('instance_id')
    if instance:known[str(instance)]=path.parents[2].name
start=datetime.datetime(2026,10,17,tzinfo=datetime.timezone.utc).timestamp()
query={'select_filters':json.dumps({'day':{'gte':start,'lte':time.time()}}),'limit':100,'format':'table'}
rows=[]
while True:
    page=api.api('GET','charges?'+urllib.parse.urlencode(query));rows+=page.get('results',[])
    if not page.get('next_token'):break
    query['after_token']=page['next_token']
owned=[row for row in rows if row.get('source','').removeprefix('instance-') in known]
instances=api.list_instances();account=api.api('GET','users/current/')
active=[{'id':x['id'],'status':x.get('actual_status'),'worker':known.get(str(x['id']))}
        for x in instances if str(x['id']) in known and x.get('actual_status')!='destroyed']
per_worker={}
for row in owned:
    worker=known[row['source'].removeprefix('instance-')]
    per_worker[worker]=per_worker.get(worker,0.)+row.get('amount',0.)
observed={row['source'].removeprefix('instance-') for row in owned}
record=dict(queried_at=time.time(),source='https://docs.vast.ai/api-reference/billing/show-charges',
    scope='Provider-reported campaign GPU/disk/bandwidth charges, rounded by provider; billing may settle after shutdown',
    status='final_snapshot' if a.final else 'provisional_provider_snapshot',results=owned,
    instance_workers=known,active_campaign_instances=active,unreported_instance_ids=sorted(set(known)-observed),
    total_reported_usd=sum(row.get('amount',0.) for row in owned),per_worker_reported_usd=per_worker,
    account_credit=account.get('credit'),account_balance=account.get('balance'))
if a.final and (active or record['unreported_instance_ids']):
    record['status']='finalization_blocked_by_active_or_unreported_instances'
name='provider_charges_final.json' if a.final else 'provider_charges_latest.json'
path=ROOT/'budget'/name;temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2));temporary.replace(path)
with (ROOT/'budget/account_snapshots.jsonl').open('a') as stream:
    stream.write(json.dumps({k:record[k] for k in ('queried_at','account_credit','account_balance','active_campaign_instances','total_reported_usd')})+'\n')
print(json.dumps({k:record[k] for k in ('status','total_reported_usd','active_campaign_instances','unreported_instance_ids','account_credit')}))
if record['status'].startswith('finalization_blocked'):raise SystemExit(1)
