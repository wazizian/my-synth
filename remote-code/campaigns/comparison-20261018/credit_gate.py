"""GPU-only development screen and independent confirmation of frozen credit models."""
import gc,json,os
from pathlib import Path
import jax,numpy as np,equinox as eqx,optax
jax.config.update('jax_enable_x64',True)
from credit_hedging.learned_run import configs,simulate,evaluate,write
from credit_hedging.learned_evaluation import metrics,bootstrap_gains
from credit_hedging.learned_policy import TargetHedger
from credit_hedging.curve_market import build_curve_tables

def main(source,checkpoints,output,bp=1):
 assert jax.default_backend()=='gpu'
 source=Path(source);checkpoints=Path(checkpoints);output=Path(output);output.mkdir(exist_ok=True)
 design=json.loads((source/'learned_design.json').read_text());cfgs=configs(source)
 models={}
 for arm in ('baseline','easy','hard'):
  root=checkpoints/arm;meta=json.loads((root/'selection.json').read_text());m=TargetHedger(jax.random.key(design['seed']));opt=optax.chain(optax.clip_by_global_norm(1.),optax.adam(meta['selected']['lr']));state=opt.init(eqx.filter(m,eqx.is_inexact_array));models[arm],_,_=eqx.tree_deserialise_leaves(root/'selected.eqx',(m,state,jax.random.key_data(jax.random.key(0))))
 results={'bp':bp,'threshold':.001,'streams':{},'scope':'Conditional paired episode uncertainty, one fitted seed; development only, final seed520261017 unopened'}
 for stream,seed in [('screen',620261017),('confirmation',720261017)]:
  rows={}
  for d in ('easy','ordinary','hard'):
   from dataclasses import replace
   cfg=replace(cfgs[d],transaction_cost=bp/10000);tables=eqx.filter_jit(build_curve_tables)(cfg);jax.block_until_ready(tables)
   policies={'baseline':models['baseline'],d:models['baseline' if d=='ordinary' else d]};aggregate={n:dict(loss=[],daily_sum=np.zeros(1260),daily_sq=np.zeros(1260),cost=0.,turnover=0.) for n in policies};flags=[]
   for start in range(0,8192,128):
    paths=simulate(jax.random.fold_in(jax.random.key(seed),('easy','ordinary','hard').index(d)*100000+start),tables,cfg,128);flags.append(np.asarray(paths['default']).any(1));alive=np.asarray(paths['alive'])
    for name,m in policies.items():
     res=evaluate(m,paths,tables,cfg);loss=np.asarray(res['loss']);r={k:np.asarray(v) for k,v in res['records'].items()}
     if not np.isfinite(loss).all() or abs(r['daily_loss'].sum(1)-loss).max()>1e-10 or np.any(r['daily_loss'][~alive[:,:-1]]!=0):raise RuntimeError('Accounting/finite/absorption failure')
     a=aggregate[name];a['loss'].append(loss);a['daily_sum']+=r['daily_loss'].sum(0);a['daily_sq']+=(r['daily_loss']**2).sum(0);a['cost']+=r['discounted_cost'].sum();a['turnover']+=(r['buy']+r['sold']).sum()
    if start%1024==0:print(json.dumps(dict(event='gate',stream=stream,domain=d,episodes=start+128)),flush=True)
   flags=np.concatenate(flags);losses={n:np.concatenate(a['loss']) for n,a in aggregate.items()};row={'metrics':{n:metrics(losses[n],a['daily_sum'],a['daily_sq'],8192,a['cost'],a['turnover'],flags) for n,a in aggregate.items()}}
   if d!='ordinary':row['gain']=bootstrap_gains(losses['baseline'],losses[d],seed+bp)
   np.savez_compressed(output/f'{stream}-{d}.npz',**losses,defaulted=flags);rows[d]=row
   results['streams'][stream]=rows;write(output/'results.partial.json',results)
   del tables;gc.collect();jax.clear_caches()
  gainpass=all(rows[d]['gain']['ci95'][0]>.001 for d in ('easy','hard'))
  orderpass=all(rows['hard']['metrics']['hard'][k]>rows['easy']['metrics']['easy'][k] for k in ('terminal_std','daily_error_std'))
  results['streams'][stream]['gate']={'gain_pass':gainpass,'risk_order_pass':orderpass,'passed':gainpass and orderpass}
 results['passed']=all(results['streams'][s]['gate']['passed'] for s in ('screen','confirmation'))
 write(output/'results.json',results);write(output/'complete.json',{'status':'complete','passed':results['passed']})
