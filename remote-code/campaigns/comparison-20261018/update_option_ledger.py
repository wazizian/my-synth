"""Compact provider-backed rental/billing ledger; control metadata only."""
from pathlib import Path
import datetime,json,sys,time,urllib.parse,fcntl
sys.path.insert(0,'/home/jump/vast-tools')
import smoke
ROOT=Path(__file__).resolve().parent

def update():
 lock=(ROOT/'option-ledger.lock').open('w')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 prior_path=ROOT/'option-instance-ledger.json';prior=json.loads(prior_path.read_text()) if prior_path.exists() else {};old={row['instance_id']:row for row in prior.get('instances',[])}
 entries={};roots=[ROOT.parent/'volatility-shocks-20261018']
 roots.extend(p for p in ROOT.iterdir() if p.is_dir() and (p.name.startswith(('option-','final-evaluator-')) or p.name in ('handoff-probe','deadline-option-proof')))
 for root in roots:
  for path in root.rglob('metadata.json'):
   if path.parent.parent.name!='runs':continue
   metadata=json.loads(path.read_text());instance=metadata.get('instance',{});iid=instance.get('instance_id')
   if not iid:continue
   run=path.parent;entry=dict(old.get(iid,{}));entry.update(instance_id=iid,label=instance.get('label'),listed_usd_per_hour=instance.get('listed_hourly'),campaign=root.name,arm=run.parent.parent.name,run_id=run.name,run_path=str(run),metadata_path=str(path),submitted_at_run_timestamp=run.name[:16],provider_actual_create_utc=entry.get('provider_actual_create_utc'),provider_actual_end_utc=entry.get('provider_actual_end_utc'),provider_actual_runtime_seconds=entry.get('provider_actual_runtime_seconds'))
   entry['export_manifests']=[str(run/name) for name in ('export-verified-manifest.json','deadline-export-manifest.json','artifacts.json') if (run/name).exists()]
   entry['final_sync_status_path']=str(run/'sync-status.json') if (run/'sync-status.json').exists() else None
   entries[iid]=entry
 historical={56609467:'original_demo_audit_success',56603812:'original_demo_audit_failed',56603656:'original_demo_audit_failed',56617341:'baseline_infrastructure_retry',56617340:'hard_infrastructure_retry',56617610:'hard_docker_network_failure',56609796:'pricing_diagnostic_mcse_failure',56616281:'pricing_diagnostic_infrastructure_failure',56617080:'pricing_diagnostic_success'}
 historical[56632049]='gate_setup_cancelled_before_workload'
 historical[56720058]='exploratory_easy_direct_ssh_bootstrap_failure_no_workload'
 for iid,role in historical.items():
  if iid not in entries:entries[iid]=dict(instance_id=iid,label=None,role=role,historical_evidence=str(ROOT.parent/'volatility-shocks-20261018/AUDIT.md'),provider_actual_create_utc=None,provider_actual_end_utc=None,provider_actual_runtime_seconds=None,export_manifests=[])
  if iid==56632049:entries[iid]['historical_evidence']=str(ROOT.parent/'volatility-shocks-20261018/option-early-gate.log')
  if iid==56720058:
   receipt=ROOT/'control-recovery/extension-easy-bootstrap-20261019T1143Z/manual-retry-receipt.json'
   entries[iid]['historical_evidence']=str(receipt)
   if receipt.exists():
    identity=json.loads(receipt.read_text())['instance'];entries[iid].update(label=identity['label'],listed_usd_per_hour=identity['listed_hourly'],offer=identity['offer'])
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();user=smoke.api('GET','users/current');instances={row['id']:row for row in smoke.list_instances()};query={'select_filters':json.dumps({'day':{'gte':datetime.datetime(2026,10,18,0,tzinfo=datetime.timezone.utc).timestamp(),'lte':time.time()}}),'limit':100,'format':'table'};charges=[]
 while True:
  page=smoke.api('GET','charges?'+urllib.parse.urlencode(query));charges.extend(page.get('results',[]))
  if not page.get('next_token'):break
  query['after_token']=page['next_token']
 for iid,entry in entries.items():
  provider=instances.get(iid)
  if provider and provider.get('label')!=entry.get('label'):entry['provider_identity_error']='ID/label mismatch; no lifecycle assertion'
  elif provider:
   entry['provider_snapshot']={key:provider.get(key) for key in ('id','label','actual_status','intended_status','start_date','end_date','duration','dph_total')}
   entry['provider_state']=provider.get('actual_status');entry['provider_checked_utc']=now
   entry['provider_last_present_utc']=now
   if provider.get('start_date'):
    entry['provider_create_timestamp_raw']=provider['start_date'];entry['provider_actual_create_utc']=datetime.datetime.fromtimestamp(provider['start_date'],datetime.timezone.utc).isoformat()
   # Provider contract end_date is NOT the actual destruction time.
   entry['provider_contract_end_timestamp_raw']=provider.get('end_date')
  else:
   entry.update(provider_state='absent_confirmed',provider_checked_utc=now,provider_absence_confirmed_utc=entry.get('provider_absence_confirmed_utc',now))
  own=[row for row in charges if str(row.get('source','')).removeprefix('instance-')==str(iid)]
  entry['actual_reported_usd']=sum(row.get('amount',0) for row in own);entry['billing_query_utc']=now;entry['billing_lag_possible']=True
  entry['provider_charges']=own
 wallet={'query_utc':now,'wallet':{key:user.get(key) for key in ('balance','credit','balance_threshold') if key in user},'active_listed_usd_per_hour':sum((row.get('dph_total') or 0) for row in instances.values() if row.get('actual_status') not in ('destroyed','stopped')),'active_running_or_loading_count':sum(row.get('actual_status') not in ('destroyed','stopped') for row in instances.values()),'note':'Listed active rental estimate; provider credit/balance are separate wallet fields; bandwidth/storage and billing lag may differ.'}
 (ROOT/'wallet.latest.json').write_text(json.dumps(wallet,indent=2))
 record=dict(query_utc=now,scope='Option discovery/restarts/audits/gates/eight-arm comparison/evaluation plus shared final evaluator audits and handoff control probe; deduplicate shared instance IDs when combining ledgers',instances=sorted(entries.values(),key=lambda row:row['instance_id']),total_actual_reported_usd=sum(e['actual_reported_usd'] for e in entries.values()),cost_kind='Provider-reported charges only; listed hourly prices are separately labeled; missing actual lifecycle times remain null')
 temporary=prior_path.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2));temporary.replace(prior_path)
 return record
if __name__=='__main__':
 result=update();print(json.dumps({'query_utc':result['query_utc'],'instances':len(result['instances']),'total_actual_reported_usd':result['total_actual_reported_usd']}))
