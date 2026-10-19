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

@eqx.filter_jit
def update(model,state,key,domain,cost,optimizer):
 key,sub=jax.random.split(key);logs=simulate(sub,domain,256)['logs']
 def objective(m):return (logsumexp(10*rollout(m,logs,cost)['loss'])-jnp.log(256.))/10
 value,grad=eqx.filter_value_and_grad(objective)(model)
 updates,state=optimizer.update(grad,state,eqx.filter(model,eqx.is_inexact_array))
 return eqx.apply_updates(model,updates),state,key,value,optax.global_norm(grad)

def samples(model,domain,cost,seed,count=2048,rule='learned',archive=None):
 rows=[];tc=[]
 for start in range(0,count,256):
  logs=simulate(jax.random.fold_in(jax.random.key(seed),start),domain,256)['logs']
  out=rollout(model,logs,cost,rule);rows.append(np.asarray(out['loss']));tc.append(np.asarray(out['costs']))
 values=np.concatenate(rows)[:count]
 if not np.isfinite(values).all():raise RuntimeError('nonfinite loss')
 if archive:np.savez_compressed(archive,loss=values,cost=np.concatenate(tc)[:count])
 return values

def save_model(path,model):
 path=Path(path);tmp=path.with_name(path.name+'.tmp');eqx.tree_serialise_leaves(tmp,model);os.replace(tmp,path)

def train(root,arm):
 domain='base' if arm=='baseline' else arm;root=Path(root);root.mkdir(parents=True,exist_ok=True)
 model0=Hedger(jax.random.key(20261017))
 for cost in COSTS:
  out=root/f'cost-{round(cost*10000)}bp';out.mkdir(exist_ok=True);trials=[]
  if (out/'selection.json').exists():continue
  initial=risk(samples(model0,domain,cost,320261017))
  for lr in LRS:
   trial=out/f'lr-{lr:g}';trial.mkdir(exist_ok=True)
   model=model0;optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(lr));state=optimizer.init(eqx.filter(model,eqx.is_inexact_array));key=jax.random.key(220261017)
   active=0.;steps=0;best=initial;best_time=0.;history=[];compile_time=0.
   checkpoint=trial/'checkpoint.json'
   if checkpoint.exists():
    saved=json.loads(checkpoint.read_text())
    model,state,key_data=eqx.tree_deserialise_leaves(trial/saved['state_file'],(model,state,jax.random.key_data(key)))
    key=jax.random.wrap_key_data(key_data)
    active=saved['active_seconds'];steps=saved['steps'];best=saved['best'];best_time=saved['best_time'];history=saved['history'];compile_time=saved['compile_seconds']
    replay=risk(samples(model,domain,cost,320261017))
    if abs(replay-history[-1]['entropic'])>1e-10:raise RuntimeError('Resume development replay mismatch')
    write(trial/'resume.json',dict(step=steps,active_seconds=active,replay_error=abs(replay-history[-1]['entropic'])))
    t=time.monotonic();warm=update(model,state,key,domain,cost,optimizer);jax.block_until_ready(warm[3]);compile_time+=time.monotonic()-t
   else:
    save_model(trial/'best.eqx',model)
    t=time.monotonic();model,state,key,value,gn=update(model,state,key,domain,cost,optimizer);jax.block_until_ready(value);compile_time=time.monotonic()-t;steps+=1
   for boundary in BOUNDARIES[1:]:
    if history and history[-1]['active_seconds']>=boundary:continue
    with (trial/'train.jsonl').open('a') as log:
     while active<boundary:
      t=time.monotonic();model,state,key,value,gn=update(model,state,key,domain,cost,optimizer);jax.block_until_ready(value);active+=time.monotonic()-t;steps+=1
      if not np.isfinite(float(value)) or not np.isfinite(float(gn)):raise RuntimeError('nonfinite update')
      if steps%10==0:log.write(json.dumps(dict(step=steps,active_seconds=active,loss=float(value),gradient=float(gn)))+'\n');log.flush()
    score=risk(samples(model,domain,cost,320261017,archive=trial/f'dev-{boundary}.npz'))
    history.append(dict(active_seconds=active,steps=steps,entropic=score))
    save_model(trial/'latest.eqx',model)
    if score<best:best=score;best_time=active;save_model(trial/'best.eqx',model)
    write(trial/'progress.json',dict(history=history,best=best,best_time=best_time,compile_seconds=compile_time,initial=initial,lr=lr))
    state_file=f'state-{boundary}.eqx';save_model(trial/state_file,(model,state,jax.random.key_data(key)))
    write(checkpoint,dict(state_file=state_file,active_seconds=active,steps=steps,best=best,best_time=best_time,history=history,compile_seconds=compile_time))
    # Retain the current complete boundary; checkpoint.json is the atomic commit marker.
    for old in trial.glob('state-*.eqx'):
     if old.name!=state_file:old.unlink()
    print(json.dumps(dict(event='selection',arm=arm,cost=cost,lr=lr,**history[-1])),flush=True)
   trials.append(dict(lr=lr,best=best,best_time=best_time,steps=steps,active_seconds=active,checkpoint=str(trial/'best.eqx')))
   write(out/'trials.json',trials)
  selected=min(trials,key=lambda x:x['best']);(out/'selected.eqx').write_bytes(Path(selected['checkpoint']).read_bytes())
  write(out/'selection.json',dict(arm=arm,domain=domain,cost=cost,selected=selected,trials=trials,initial=initial))
 write(root/'complete.json',dict(status='complete',arm=arm))
