# ruff: noqa
"""Daily two-factor stochastic-arrival volatility law and price-history-only hedger."""
import os,json,time,math
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from jax.scipy.special import ndtr,logsumexp
import equinox as eqx
import numpy as np
import optax
DT=1/252
HORIZON=126
BURN=0
P_HALF={'easy':(3.,3.),'base':(3.,3.),'hard':(40.,3.)}
Q_HALF={'easy':(3.,3.),'base':(40.,40.),'hard':(1.,40.)}
PROB={'easy':0.,'base':1/63,'hard':1/63}
COSTS=(.0001,.0003,.001)
LRS=(.0001,.0003,.001)
BOUNDARIES=(0,60,180,360,600,1200,1800)

def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(tmp,path)

def coefficients(half):
 a=jnp.exp(-jnp.log(2.)/jnp.asarray(half));sd=.35*jnp.sqrt(DT*(1-a*a)/(-2*jnp.log(a)))
 return a,sd

def variance(x):return jnp.minimum(.04*jnp.exp(jnp.minimum(jnp.sum(x,axis=-1),10.)),1.)

def advance(x,z,u,half,prob):
 a,sd=coefficients(half)
 return a*x+sd*z+(u<prob)[...,None]*jnp.log(2.)

@eqx.filter_jit
def simulate(key,domain,n,measure='P',days=HORIZON,burn=BURN):
 """Return causal normalized price history and latent diagnostic states separately."""
 half=(P_HALF if measure=='P' else Q_HALF)[domain]
 kz,ku=jax.random.split(key)
 z=jax.random.normal(kz,(burn+days,n,3));u=jax.random.uniform(ku,(burn+days,n))
 def step(carry,inputs):
  x,log_s=carry;zz,uu=inputs
  v=variance(x);r=-.5*v*DT+jnp.sqrt(v*DT)*zz[:,2]
  return (advance(x,zz[:,:2],uu,half,PROB[domain]),log_s+r),(log_s,x,r)
 (last_x,last_s),(logs,states,returns)=jax.lax.scan(step,(jnp.zeros((n,2)),jnp.zeros(n)),(z,u))
 logs=jnp.concatenate((logs,last_s[None]),axis=0)
 logs=logs-logs[burn][None]
 return {'logs':logs,'states':jnp.concatenate((states,last_x[None]),axis=0),'returns':returns}

@eqx.filter_jit
def integrated_variance(key,states,domain,tenor,nq):
 """Future Q jump at end of each interval; no future jump is omitted."""
 kz,ku=jax.random.split(key);z=jax.random.normal(kz,(tenor,nq,2));u=jax.random.uniform(ku,(tenor,nq))
 x=jnp.broadcast_to(states[...,None,:],states.shape[:-1]+(nq,2))
 def step(carry,inputs):
  x,integ=carry;zz,uu=inputs
  return (advance(x,zz,uu,Q_HALF[domain],PROB[domain]),integ+variance(x)*DT),None
 (_,integ),_=jax.lax.scan(step,(x,jnp.zeros(x.shape[:-1])),(z,u))
 return integ

@eqx.filter_jit
def price(key,states,spot,strike,domain,tenor,nq):
 integ=integrated_variance(key,states,domain,tenor,nq)
 vol=jnp.sqrt(integ);spot=jnp.asarray(spot);strike=jnp.asarray(strike)
 d1=(jnp.log(spot/strike)[...,None]+.5*integ)/vol;d2=d1-vol
 conditional=spot[...,None]*ndtr(d1)-strike[...,None]*ndtr(d2)
 return jnp.mean(conditional,axis=-1)

@eqx.filter_jit
def atm_iv(key,states,domain,tenor,nq):
 call=price(key,states,jnp.ones(states.shape[:-1]),jnp.ones(states.shape[:-1]),domain,tenor,nq)
 def bisect(_,bounds):
  lo,hi=bounds;mid=(lo+hi)/2;value=2*ndtr(.5*mid*jnp.sqrt(tenor*DT))-1
  return jnp.where(value<call,mid,lo),jnp.where(value<call,hi,mid)
 lo,hi=jax.lax.fori_loop(0,48,bisect,(jnp.zeros_like(call),jnp.ones_like(call)*3))
 return (lo+hi)/2,call

class Hedger(eqx.Module):
 gru:eqx.nn.GRUCell
 head:eqx.nn.Linear
 def __init__(self,key):
  a,b=jax.random.split(key);self.gru=eqx.nn.GRUCell(4,32,key=a);self.head=eqx.nn.Linear(32,1,key=b)
  self.head=eqx.tree_at(lambda m:(m.weight,m.bias),self.head,(jnp.zeros_like(self.head.weight),jnp.zeros_like(self.head.bias)))
 def __call__(self,features,hidden):
  hidden=self.gru(features,hidden)
  return jax.nn.sigmoid(self.head(hidden)[0]),hidden

@eqx.filter_jit
def rollout(model,logs,cost,rule='learned'):
 """No states, RV or quotes accepted: a structural observation barrier."""
 n=logs.shape[1];h=jnp.zeros((n,32));holding=jnp.zeros(n)
 def warm(hidden,t):
  features=jnp.stack((logs[t],logs[jnp.maximum(t-1,0)],jnp.ones(n),jnp.zeros(n)),axis=1)
  _,hidden=jax.vmap(model)(features,hidden)
  return hidden,None
 h,_=jax.lax.scan(warm,h,jnp.arange(BURN))
 def step(carry,t):
  hidden,old=carry;lp=logs[BURN+t];prev=logs[jnp.maximum(BURN+t-1,0)];remaining=(HORIZON-t)/HORIZON
  features=jnp.stack((lp,prev,jnp.full(n,remaining),old),axis=1)
  proposed,hidden=jax.vmap(model)(features,hidden)
  if rule=='bs':proposed=ndtr((lp+.5*.04*(HORIZON-t)*DT)/(.2*jnp.sqrt((HORIZON-t)*DT)))
  if rule=='constant':proposed=jnp.full(n,model) # not used; constant evaluated separately
  s=jnp.exp(lp);next_s=jnp.exp(logs[BURN+t+1]);tc=cost*jnp.abs(proposed-old)*s
  gain=proposed*(next_s-s)
  return (hidden,proposed),(gain,tc,proposed)
 (_,holding),(gains,costs,holdings)=jax.lax.scan(step,(h,holding),jnp.arange(HORIZON))
 final_s=jnp.exp(logs[-1]);liquidation=cost*holding*final_s
 payoff=jnp.maximum(final_s-1.,0.)
 loss=payoff-jnp.sum(gains,axis=0)+jnp.sum(costs,axis=0)+liquidation
 return dict(loss=loss,gains=gains,costs=jnp.sum(costs,axis=0)+liquidation,holdings=holdings,payoff=payoff,liquidation=liquidation)

@eqx.filter_jit
def constant_rollout(logs,cost,holding):
 s=jnp.exp(logs[BURN:]);payoff=jnp.maximum(s[-1]-1,0.)
 costs=cost*holding*(s[0]+s[-1]);loss=payoff-holding*(s[-1]-s[0])+costs
 return dict(loss=loss,costs=costs)

def risk(x):
 x=np.asarray(x);z=10*x;m=z.max();return float((m+np.log(np.mean(np.exp(z-m))))/10)

def metrics(x):
 x=np.asarray(x);return dict(entropic=risk(x),cvar95=float(np.sort(x)[int(.95*len(x)):].mean()),sd=float(x.std(ddof=1)),mse=float(np.mean(x*x)),mean=float(x.mean()))

