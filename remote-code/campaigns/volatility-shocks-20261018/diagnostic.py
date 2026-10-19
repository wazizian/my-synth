"""GPU-only diagnostic and numerical gate before training."""
from pathlib import Path
import json
import jax
import jax.numpy as jnp
import numpy as np
from model import *

def event_paths(key,domain,n=256,days=140):
 kz,ku=jax.random.split(key);z=jax.random.normal(kz,(days,n,3));u=jax.random.uniform(ku,(days,n))
 def step(carry,inputs):
  x,s=carry;t,zz,uu=inputs
  # Conditional first arrival at day35. After that arrivals remain stochastic.
  x=x+jnp.where((t==35)&(domain!='easy'),jnp.log(2.),0.)
  v=variance(x);r=-.5*v*DT+jnp.sqrt(v*DT)*zz[:,2]
  next_x=advance(x,zz[:,:2],jnp.where(t<35,1.,uu),P_HALF[domain],PROB[domain])
  return (next_x,s+r),(x,s,r,v)
 _,out=jax.lax.scan(step,(jnp.zeros((n,2)),jnp.zeros(n)),(jnp.arange(days),z,u))
 return out

def main(root):
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 assert jax.default_backend()=='gpu'
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 results={};arrays={};fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
 for ax,domain in zip(axes,('easy','base','hard')):
  states,logs,returns,var=eqx.filter_jit(event_paths)(jax.random.key(20261017),domain)
  # Common Q draws give smooth comparisons; independent replicates below quantify error.
  iv,call=atm_iv(jax.random.key(9871),states,domain,21,1024)
  returns=np.asarray(returns);iv=np.asarray(iv);rv=np.full_like(returns,np.nan)
  for t in range(5,140):rv[t]=np.sqrt(252*np.mean(returns[t-5:t]**2,axis=0))
  pooled=np.sqrt(np.nanmean(rv**2,axis=1));mean_iv=iv.mean(axis=1)
  results[domain]={str(t):dict(rv=float(pooled[35+t]),iv=float(mean_iv[35+t])) for t in (0,5,15,30,60)}
  arrays.update({domain+'_states':np.asarray(states),domain+'_logs':np.asarray(logs),domain+'_rv':rv,domain+'_iv':iv,domain+'_call':np.asarray(call)})
  ax.plot(np.arange(140)-35,pooled,label='Observable 5-day RV (diagnostic only)');ax.plot(np.arange(140)-35,mean_iv,label='Actual rolling 21-day Q IV')
  ax.set(title=domain,xlabel='Days from conditioned first shock');ax.grid(alpha=.2)
 axes[0].legend();fig.tight_layout();fig.savefig(root/'stochastic_rv_iv.png',dpi=160);plt.close(fig)
 np.savez_compressed(root/'event_paths.npz',**arrays)
 checks={}
 checks['base_rv_decay']=results['base']['15']['rv']<results['base']['5']['rv']
 checks['base_iv_decay']=results['base']['15']['iv']<results['base']['5']['iv']
 checks['hard_rv_above_base']=results['hard']['30']['rv']>results['base']['30']['rv']+.025
 checks['hard_rv_above_iv']=results['hard']['30']['rv']>results['hard']['30']['iv']+.015
 checks['hard_iv_decay']=results['hard']['15']['iv']<results['hard']['0']['iv']-.02
 # Quantify future-arrival pricing contribution and MC errors at multiple maturities.
 quotes={}
 for domain in ('easy','base','hard'):
  quotes[domain]={}
  for tenor in (7,21,63,126):
   estimates=np.array([float(price(jax.random.key(70000+i),jnp.zeros((1,2)),jnp.ones(1),jnp.ones(1),domain,tenor,65536)[0]) for i in range(8)])
   se=float(estimates.std(ddof=1)/np.sqrt(8));quotes[domain][str(tenor)]=dict(price=float(estimates.mean()),replicate_se=se,nq_per_replicate=65536,replicates=8)
   checks[f'quote_error_{domain}_{tenor}']=se<=.0001
  # Horizon1 must equal BS at current variance: end-of-day arrivals affect the next interval.
  one=float(price(jax.random.key(22),jnp.zeros((1,2)),jnp.ones(1),jnp.ones(1),domain,1,8192)[0])
  analytic=float(2*ndtr(.1*jnp.sqrt(DT))-1)
  checks[f'one_day_timing_{domain}']=abs(one-analytic)<1e-12
  # MC convergence split independent samples and report rather than inventing exact equality.
  prices=[float(price(jax.random.key(811+i),jnp.zeros((1,2)),jnp.ones(1),jnp.ones(1),domain,126,nq)[0]) for i,nq in enumerate((4096,16384,65536))]
  quotes[domain]['convergence_126']=dict(nq=[4096,16384,65536],prices=prices)
  checks[f'convergence_{domain}']=abs(prices[-1]-quotes[domain]['126']['price'])<.0005
  # Stock martingale under the actual bounded-variance Q process.
  qpaths=simulate(jax.random.key(282),domain,65536,measure='Q',days=21,burn=0)
  terminal=np.exp(np.asarray(qpaths['logs'][-1]));error=float(terminal.mean()-1);se=float(terminal.std(ddof=1)/np.sqrt(len(terminal)))
  checks[f'stock_martingale_{domain}']=abs(error)<5*se
  quotes[domain]['stock_martingale_21']=dict(error=error,se=se)
  # Tower check: C(0,21) versus E_Q[C(1,20)] using sampled day1 states/spots.
  step=simulate(jax.random.key(972),domain,4096,measure='Q',days=1,burn=0)
  continuation=np.asarray(price(jax.random.key(973),step['states'][-1],jnp.exp(step['logs'][-1]),jnp.ones(4096),domain,20,2048))
  start=quotes[domain]['21']['price'];tower_error=float(continuation.mean()-start);tower_se=float(continuation.std(ddof=1)/64)
  checks[f'option_tower_{domain}']=abs(tower_error)<5*(tower_se+quotes[domain]['21']['replicate_se'])+.0001
  quotes[domain]['option_tower_21']=dict(error=tower_error,outer_se=tower_se,outer=4096,inner=2048)
  # Independent raw payoff cross-check against conditional integration.
  direct=simulate(jax.random.key(581),domain,65536,measure='Q',days=126,burn=0)
  payoff=np.maximum(np.exp(np.asarray(direct['logs'][-1]))-1,0);err=float(payoff.mean()-quotes[domain]['126']['price']);se=float(payoff.std(ddof=1)/256)
  checks[f'direct_payoff_{domain}']=abs(err)<5*(se+quotes[domain]['126']['replicate_se'])
  quotes[domain]['direct_payoff_126']=dict(error=err,se=se)
 # Information barrier and exact ledger equivalence at constant initialization.
 model=Hedger(jax.random.key(20261017));paths=simulate(jax.random.key(512),'hard',256)
 for cost in COSTS:
  out=rollout(model,paths['logs'],cost);static=constant_rollout(paths['logs'],cost,.5)
  error=float(jnp.max(jnp.abs(out['loss']-static['loss'])))
  checks[f'ledger_{cost}']=error<1e-12
  checks[f'position_bounds_{cost}']=bool(jnp.all((out['holdings']>=0)&(out['holdings']<=1)))
  # Direct pathwise cash ledger: purchase at today's price, liquidation at terminal price.
  s=jnp.exp(paths['logs'][BURN:]);hold=out['holdings'];old=jnp.concatenate((jnp.zeros((1,256)),hold[:-1]),axis=0)
  cash=-jnp.sum((hold-old)*s[:-1]+cost*jnp.abs(hold-old)*s[:-1],axis=0)+hold[-1]*s[-1]-cost*hold[-1]*s[-1]
  checks[f'cash_ledger_{cost}']=float(jnp.max(jnp.abs(out['loss']-(out['payoff']-cash))))<1e-12
 checks={k:bool(v) for k,v in checks.items()}
 report=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,event_summary=results,quotes=quotes,backend=jax.default_backend(),observation_audit='rollout signature accepts only model, price history, cost and rule; hidden states and quotes are separate')
 write(root/'diagnostic.json',report);print(json.dumps(report,indent=2),flush=True)
 if not all(checks.values()):raise RuntimeError('stochastic numerical/economic gate failed: '+str([k for k,v in checks.items() if not v]))
