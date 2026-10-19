"""Provider and runtime-metadata budget admission; no scientific metrics processed."""
from pathlib import Path
import datetime,json,sys,time
from offer_exclusions import excluded_offer_ids
from campaign_deadlines import authorized_namespace,deadline_for,deadline_utc_for
sys.path.insert(0,'/home/jump/vast-tools')
import smoke
ROOT=Path(__file__).resolve().parent
ARMS=('baseline','easy_reference','hard_reference','dpg_hard','soar_hard','selfplay','dpg_mixed','soar_mixed')

def last_active(path):
 if not path.exists():return 0.
 with path.open('rb') as stream:
  stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-16384));lines=stream.read().splitlines()
 for line in reversed(lines):
  try:return min(1800.,float(json.loads(line).get('active_seconds',0.)))
  except (ValueError,TypeError):continue
 return 0.
def active_run(launch):
 state=launch/'active.json'
 if not state.exists():return None,None
 saved=json.loads(state.read_text())
 for run in sorted((launch/'runs').glob('*'),reverse=True):
  meta=run/'metadata.json'
  if meta.exists() and json.loads(meta.read_text()).get('instance',{}).get('instance_id')==saved.get('instance_id'):return saved,run
 return saved,None

def snapshot(benchmark,namespace):
 namespace=Path(namespace);now=datetime.datetime.now(datetime.timezone.utc).isoformat();wallet=smoke.api('GET','users/current');provider={r['id']:r for r in smoke.list_instances()};credit=float(wallet.get('credit',wallet.get('balance',0)) or 0);old=[];committed=[]
 for name in ('credit-ordinary-long-20261018','credit-high-default-20261018','volatility-shocks-20261018'):
  for arm in ('baseline','easy','hard'):
   launch=ROOT.parent/name/'launchers'/arm;saved,run=active_run(launch)
   if not saved:continue
   row=provider.get(saved.get('instance_id'))
   if not row or row.get('actual_status') in ('stopped','destroyed'):continue
   rate=float(row.get('dph_total') or saved.get('listed_hourly') or .6)
   elapsed=0.
   if run:
    for bp in (1,3,10):
     folder=run/'outputs'/('training/cost-'+str(bp)+'bp' if name.startswith('volatility') else 'cost-'+str(bp)+'bp/learned')
     if (folder/'selected.eqx').exists():elapsed+=5400.;continue
     for lr in (.0001,.0003,.001):elapsed+=last_active(folder/f'lr-{lr:g}'/'train.jsonl')
   hours=max(0.,16200.-elapsed)/3600.+.10
   old.append(dict(campaign=name,arm=arm,instance_id=saved.get('instance_id'),listed_usd_per_hour=rate,remaining_active_plus_overhead_hours=hours,estimated_remaining_usd=hours*rate,timing_basis='Latest logged active seconds; .10h overhead allowance; no loss/return metrics read'))
 extension=authorized_namespace()
 committed_names=['credit-final','option-final']
 if extension is not None:committed_names.append(extension.name)
 for name in committed_names:
  for arm in ARMS:
   launch=ROOT/name/arm/'launcher';saved,run=active_run(launch)
   if not saved:continue
   row=provider.get(saved.get('instance_id'))
   if not row or row.get('actual_status') in ('stopped','destroyed'):continue
   rate=float(row.get('dph_total') or saved.get('listed_hourly') or .6);elapsed=0.
   if run:
    for lr in (.0001,.0003,.001):elapsed+=last_active(run/'outputs/training'/f'lr-{lr:g}'/'training.jsonl')
   hours=max(0.,5400.-elapsed)/3600.+.30
   committed.append(dict(namespace=name,arm=arm,instance_id=saved.get('instance_id'),listed_usd_per_hour=rate,remaining_active_plus_overhead_hours=hours,estimated_remaining_usd=hours*rate))
 missing=[]
 for arm in ARMS:
  launch=namespace/arm/'launcher'
  if (launch/'active.json').exists():continue
  if any((r/'outputs/complete.json').exists() and (r/'artifacts.json').exists() for r in (launch/'runs').glob('*')):continue
  missing.append(arm)
 offers=[]
 if missing:
  query={'gpu_name':{'eq':'RTX 4090'},'num_gpus':{'eq':1},'gpu_ram':{'gte':24000},'verified':{'eq':True},'rentable':{'eq':True},'rented':{'eq':False},'external':{'eq':False},'direct_port_count':{'gte':1},'reliability2':{'gte':.98},'type':'on-demand','allocated_storage':40,'cuda_max_good':{'gte':13.0},'inet_down':{'gte':500},'order':[['dph_total','asc']],'limit':100}
  bad=excluded_offer_ids()
  offers=sorted((r for r in smoke.api('POST','bundles/',query).get('offers',[]) if r.get('id') not in bad and float(r['dph_total'])<=.60),key=lambda r:r['dph_total'])[:len(missing)]
 offered_rate=sum(float(row['dph_total']) for row in offers);unknown=len(missing)-len(offers);planned=1.8*(offered_rate+.60*unknown);old_cost=sum(row['estimated_remaining_usd'] for row in old);committed_cost=sum(row['estimated_remaining_usd'] for row in committed);reserve=1.5;artifact_reserve=1.0;required=old_cost+committed_cost+planned+reserve+artifact_reserve
 result=dict(query_utc=now,benchmark=benchmark,namespace=str(namespace),provider_available_credit_usd=credit,wallet_fields={key:wallet.get(key) for key in ('credit','balance') if key in wallet},current_listed_active_usd_per_hour=sum(float(row.get('dph_total') or 0) for row in provider.values() if row.get('actual_status') not in ('stopped','destroyed')),old_remaining=old,old_remaining_estimate_usd=old_cost,already_committed_comparisons=committed,already_committed_remaining_estimate_usd=committed_cost,new_arms=missing,new_arm_hours_assumption=1.8,cheapest_qualifying_offers=[dict(id=row['id'],listed_usd_per_hour=row['dph_total']) for row in offers],unknown_offer_count_priced_at_cap=unknown,cap_usd_per_hour=.60,new_comparison_estimate_usd=planned,final_evaluation_reserve_usd=reserve,artifact_retention_reserve_usd=artifact_reserve,total_required_remaining_usd=required,estimated_margin_usd=credit-required,admissible=credit>=required and not (ROOT/'funding_halt.json').exists() and time.time()<=deadline_for(namespace)-9000,authorized_deadline_utc=deadline_utc_for(namespace),funding_halt_present=(ROOT/'funding_halt.json').exists(),assumptions='Timing and provider financial metadata only. Offers are listings, not reservations; all fixed scientific budgets and arms preserved. Future optional unapproved comparisons are not committed until their admission snapshot. Runtime, transfer and billing lag remain uncertain.')
 target=ROOT/f'budget-admission.{benchmark}.json';temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2));temporary.replace(target);return result
def wait_for_admission(benchmark,namespace):
 deadline=deadline_for(namespace)
 while True:
  if time.time()>deadline-9000:raise RuntimeError('Operational deadline disallows another full-budget training allocation')
  result=snapshot(benchmark,namespace)
  if result['admissible']:return result
  if time.time()>deadline-9000:raise RuntimeError('Wallet reservation insufficient before latest full-budget training start')
  # Wait for paid commitments to finish and cheap capacity to return; never cut scientific arms/time.
  time.sleep(300)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('benchmark',choices=('credit','option'));p.add_argument('namespace',type=Path);a=p.parse_args();r=snapshot(a.benchmark,a.namespace);print(json.dumps({key:r[key] for key in ('query_utc','provider_available_credit_usd','old_remaining_estimate_usd','already_committed_remaining_estimate_usd','new_comparison_estimate_usd','final_evaluation_reserve_usd','artifact_retention_reserve_usd','estimated_margin_usd','admissible')}))
