"""Paired fresh episodes and terminal/daily marked hedge diagnostics."""
from model import *

def bootstrap_gain(baseline,specialist,seed=712,count=1000):
 rng=np.random.default_rng(seed);est=[]
 for _ in range(count):
  indices=rng.integers(0,len(baseline),len(baseline));est.append(risk(baseline[indices])-risk(specialist[indices]))
 return dict(gain=risk(baseline)-risk(specialist),ci95=np.quantile(est,[.025,.975]).tolist(),resampling='paired whole episodes; conditional on one trained initialization')

def main(root,models,diagnostic):
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 initial={d:diagnostic['quotes'][d]['126']['price'] for d in ('easy','base','hard')}
 result=dict(costs={},initial_premiums=initial,seed=520261017,episodes=8192,bootstrap=1000,training_seed_replicates=1)
 model0=Hedger(jax.random.key(20261017))
 # Compute hidden-state dealer marks only for evaluation, never passed into rollout.
 # Same independent subset and same marks used for every policy and transaction cost.
 daily={};quote_noise={}
 for domain in ('easy','base','hard'):
  paths=simulate(jax.random.key(620261017),domain,256);logs=paths['logs'];states=paths['states'];marks=[];replicate=[]
  for t in range(HORIZON):
   mark=price(jax.random.fold_in(jax.random.key(848),t),states[t],jnp.exp(logs[t]),jnp.ones(256),domain,HORIZON-t,4096)
   marks.append(np.asarray(mark))
   if t in (0,21,63,105):
    independent=price(jax.random.fold_in(jax.random.key(849),t),states[t],jnp.exp(logs[t]),jnp.ones(256),domain,HORIZON-t,4096)
    replicate.append(dict(day=t,rms_independent_price_difference=float(jnp.sqrt(jnp.mean((mark-independent)**2)))))
  marks.append(np.maximum(np.exp(np.asarray(logs[-1]))-1.,0.));marks=np.stack(marks)
  daily[domain]=(logs,np.diff(marks,axis=0));quote_noise[domain]=replicate
  np.savez_compressed(root/f'daily-marks-{domain}.npz',logs=np.asarray(logs),states=np.asarray(states),marks=marks)
 result['daily_quote_error']=quote_noise
 for cost in COSTS:
  tag=f'cost-{round(cost*10000)}bp';loaded={}
  for arm in ('baseline','easy','hard'):
   loaded[arm]=eqx.tree_deserialise_leaves(Path(models)/arm/tag/'selected.eqx',model0)
  cr={}
  for domain in ('easy','base','hard'):
   # Own-domain fixed-hedge tuning only on development episodes.
   candidates=np.linspace(0.,1.,41);development=[]
   for start in range(0,2048,256):development.append(simulate(jax.random.fold_in(jax.random.key(320261017),start),domain,256)['logs'])
   scores=[]
   for holding in candidates:
    values=np.concatenate([np.asarray(constant_rollout(logs,cost,float(holding))['loss']) for logs in development]);scores.append(risk(values))
   fixed=float(candidates[np.argmin(scores)])
   raw={name:[] for name in (*loaded,'static','bs')};raw_cost={name:[] for name in raw}
   for start in range(0,8192,256):
    paths=simulate(jax.random.fold_in(jax.random.key(520261017),start),domain,256)
    for name in raw:
     out=constant_rollout(paths['logs'],cost,fixed) if name=='static' else rollout(model0 if name=='bs' else loaded[name],paths['logs'],cost,'bs' if name=='bs' else 'learned')
     raw[name].append(np.asarray(out['loss'])-initial[domain]);raw_cost[name].append(np.asarray(out['costs']))
   values={name:np.concatenate(rows) for name,rows in raw.items()}
   np.savez_compressed(root/f'{tag}-{domain}-terminal.npz',**values,**{k+'_cost':np.concatenate(v) for k,v in raw_cost.items()})
   terminal={k:dict(**metrics(v),mean_cost=float(np.concatenate(raw_cost[k]).mean())) for k,v in values.items()}
   # Daily marked error = option mark increment - holding*stock increment + trading cost.
   logs,dmark=daily[domain];daily_metrics={}
   for name in values:
    if name=='static':
     s=np.exp(np.asarray(logs));gain=fixed*np.diff(s,axis=0);daily_cost=np.zeros_like(gain);daily_cost[0]=cost*fixed*s[0];daily_cost[-1]+=cost*fixed*s[-1]
    else:
     out=rollout(model0 if name=='bs' else loaded[name],logs,cost,'bs' if name=='bs' else 'learned');gain=np.asarray(out['gains']);h=np.asarray(out['holdings']);s=np.exp(np.asarray(logs));old=np.concatenate((np.zeros((1,256)),h[:-1]));daily_cost=cost*np.abs(h-old)*s[:-1];daily_cost[-1]+=np.asarray(out['liquidation'])
    error=dmark-gain+daily_cost
    daily_metrics[name]=dict(sd=float(error.std(ddof=1)),mean=float(error.mean()),episodes=256,quote_paths=4096)
    # The marks telescope to payoff minus the current path's initial mark.
    if name!='static':
     expected=np.asarray(out['loss'])-(np.maximum(s[-1]-1,0)-dmark.sum(axis=0))
     if np.max(np.abs(error.sum(axis=0)-expected))>1e-10:raise RuntimeError('daily marked ledger mismatch')
   comparison={name:bootstrap_gain(values['baseline'],v) for name,v in values.items() if name!='baseline'}
   cr[domain]=dict(terminal=terminal,daily_marked_errors=daily_metrics,baseline_minus=comparison,static_holding=fixed)
  easy=cr['easy']['baseline_minus']['easy'];hard=cr['hard']['baseline_minus']['hard']
  gate=dict(easy_useful_gain=easy['ci95'][0]>.0005,hard_useful_gain=hard['ci95'][0]>.0005,hard_residual_sd_above_easy=cr['hard']['terminal']['hard']['sd']>cr['easy']['terminal']['easy']['sd'])
  cr['gate']=dict(checks=gate,passed=all(gate.values()),margin=.0005)
  result['costs'][tag]=cr;write(root/'results.partial.json',result)
 result['status']='complete';write(root/'results.json',result);write(root/'complete.json',dict(status='complete'))
