"""Paired held-out evaluation for ordinary-only longer training and cost sweep."""
from pathlib import Path
import json,gc
import jax,numpy as np,equinox as eqx,optax
from .learned_run import configs,simulate,evaluate,write
from .learned_evaluation import metrics,bootstrap_gains
from .learned_policy import TargetHedger
from .curve_market import build_curve_tables
from .diagnostic_policies import FixedProtection

def main(source,output,checkpoints):
 source=Path(source);output=Path(output);output.mkdir(exist_ok=True);checkpoints=Path(checkpoints)
 design=json.loads((source/'learned_design.json').read_text());cfgs=configs(source)
 results={}
 for bp in (1,3,10):
  from dataclasses import replace
  cfgs={d:replace(c,transaction_cost=bp/10000) for d,c in cfgs.items()}
  models={};choices={};out=output/f'cost-{bp}bp';out.mkdir(exist_ok=True)
  for arm in ('baseline','easy','hard'):
   folder=checkpoints/f'cost-{bp}bp'/arm
   meta=json.loads((folder/'selection.json').read_text());lr=meta['selected']['lr']
   model=TargetHedger(jax.random.key(design['seed']));opt=optax.chain(optax.clip_by_global_norm(1.),optax.adam(lr));state=opt.init(eqx.filter(model,eqx.is_inexact_array))
   models[arm],_,_=eqx.tree_deserialise_leaves(folder/'selected.eqx',(model,state,jax.random.key_data(jax.random.key(0))))
   choices[arm]=meta
  write(out/'frozen_selection.json',choices)
  results[bp]={}
  for d in ('easy','ordinary','hard'):
   cfg=cfgs[d];tables=eqx.filter_jit(build_curve_tables)(cfg);jax.block_until_ready(tables)
   targets=np.r_[np.arange(.5,1.50001,.05),0.,2.];dev0=[];dev1=[]
   for start in range(0,2048,128):
    key=jax.random.fold_in(jax.random.key(design['selection_seed']),('easy','ordinary','hard').index(d)*100000+start)
    paths=simulate(key,tables,cfg,128)
    dev0.append(np.asarray(evaluate(FixedProtection(0.),paths,tables,cfg)['loss']));dev1.append(np.asarray(evaluate(FixedProtection(1.),paths,tables,cfg)['loss']))
   l0=np.concatenate(dev0);l1=np.concatenate(dev1)
   def risk(x):
    z=10*x;m=z.max();return float((m+np.log(np.exp(z-m).mean()))/10)
   scores=[risk(l0+t*(l1-l0)) for t in targets];target=float(targets[np.argmin(scores)])
   own='baseline' if d=='ordinary' else d
   replay=[]
   for start in range(0,2048,128):
    key=jax.random.fold_in(jax.random.key(design['selection_seed']),('easy','ordinary','hard').index(d)*100000+start)
    replay.append(np.asarray(evaluate(models[own],simulate(key,tables,cfg,128),tables,cfg)['loss']))
   error=abs(risk(np.concatenate(replay))-choices[own]['selected']['entropic'])
   if error>1e-10:raise RuntimeError('Selected checkpoint replay failed')
   policies=dict(models,static_tuned=FixedProtection(target),static_unit=FixedProtection(1.))
   aggregate={n:dict(loss=[],daily_sum=np.zeros(1260),daily_sq=np.zeros(1260),cost=0.,turnover=0.) for n in policies};flags=[];audits=[]
   for start in range(0,8192,128):
    key=jax.random.fold_in(jax.random.key(820261017),('easy','ordinary','hard').index(d)*100000+start)
    paths=simulate(key,tables,cfg,128);alive=np.asarray(paths['alive']);flags.append(np.asarray(paths['default']).any(1))
    for name,model in policies.items():
     res=evaluate(model,paths,tables,cfg);loss=np.asarray(res['loss']);r={k:np.asarray(v) for k,v in res['records'].items()}
     e=max(float(np.max(np.abs(r['daily_loss'].sum(1)-loss))),float(np.max(np.abs(np.asarray(res['initial_price'])-r['discounted_cashflow'].sum(1)-loss))))
     if e>1e-10 or not all(np.isfinite(v).all() for v in r.values()) or not np.isfinite(loss).all():raise RuntimeError('Accounting/finite failure')
     if np.any(r['notional']<0) or np.any(r['notional']>2+1e-10) or np.any(r['daily_loss'][~alive[:,:-1]]!=0):raise RuntimeError('Bounds/absorption failure')
     a=aggregate[name];a['loss'].append(loss);a['daily_sum']+=r['daily_loss'].sum(0);a['daily_sq']+=(r['daily_loss']**2).sum(0);a['cost']+=r['discounted_cost'].sum();a['turnover']+=(r['buy']+r['sold']).sum();audits.append(e)
    if start%1024==0:print(json.dumps(dict(event='evaluation',bp=bp,domain=d,episodes=start+128)),flush=True)
   flags=np.concatenate(flags);rows={};losses={}
   for name,a in aggregate.items():
    losses[name]=np.concatenate(a['loss']);rows[name]=metrics(losses[name],a['daily_sum'],a['daily_sq'],8192,a['cost'],a['turnover'],flags)
   np.savez_compressed(out/(d+'-losses.npz'),**losses,defaulted=flags)
   gains={name:bootstrap_gains(losses['baseline'],losses[name],820261017+bp) for name in ('easy','hard','static_tuned')}
   results[bp][d]=dict(metrics=rows,gains=gains,static_target=target,static_development_scores=scores,static_targets=targets.tolist(),replay_error=error,accounting_checks=len(audits),max_accounting_error=max(audits))
   write(output/'results.partial.json',results)
   del tables;gc.collect();jax.clear_caches()
 write(output/'results.json',results)
 write(output/'complete.json',dict(status='complete',base_distribution='ordinary only',scope='single initialization; conditional paired episode bootstrap; triple default intensity',hazard_multiplier=3.))
