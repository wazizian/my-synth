"""Read-only campaign-scoped provider billing and live-instance snapshot."""
from pathlib import Path
import datetime
import importlib.util
import json
import time
import urllib.parse

root=Path(__file__).resolve().parent
ledger=json.loads((root/'ledger.json').read_text())
ids={str(j['instance_id']) for j in ledger['jobs'] if 'instance_id' in j}
spec=importlib.util.spec_from_file_location('control','/home/jump/vast-tools/smoke.py')
control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
start=datetime.datetime.fromisoformat(ledger['started_at']).timestamp()-3600
query={'select_filters':json.dumps({'day':{'gte':start,'lte':time.time()}}),'limit':100,'format':'table'}
rows=[]
while True:
    page=control.api('GET','charges?'+urllib.parse.urlencode(query))
    rows+=page.get('results',[])
    if not page.get('next_token'):break
    query['after_token']=page['next_token']
owned=[r for r in rows if r.get('source','').removeprefix('instance-') in ids]
active=[{'id':r['id'],'status':r.get('actual_status')} for r in control.list_instances()
        if str(r['id']) in ids and r.get('actual_status')!='destroyed']
record=dict(queried_at=time.time(),total_reported_usd=sum(r.get('amount',0) for r in owned),
            results=owned,active_campaign_instances=active,
            unreported_instance_ids=sorted(ids-{r.get('source','').removeprefix('instance-') for r in owned}),
            scope='Provider-reported campaign charges; provider rounding and settlement can change totals')
(root/'provider_billing.latest.json').write_text(json.dumps(record,indent=2)+'\n')
ledger['spent_reported_usd']=record['total_reported_usd']
(root/'ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ('total_reported_usd','active_campaign_instances','unreported_instance_ids')}))
