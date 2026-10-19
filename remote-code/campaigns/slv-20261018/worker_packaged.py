"""Standalone GPU worker. Frozen SLV experiment; no provider credentials.
CIR transition: https://arxiv.org/abs/0802.4411 .
Quadrature convention: https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite.hermgauss.html
"""
import os, json, time, pathlib, hashlib, pickle, argparse, traceback
import numpy as np
from scipy.stats import ncx2
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax.scipy.special import ndtr, logsumexp
import equinox as eqx
import optax

D = json.loads('{\n  "campaign_id": "slv-20261018",\n  "authorization": "User set active goal execute the plan after reviewing updated BS/SLV plans; new execution authorization, not reuse of previous budget",\n  "training_seed": 20261017,\n  "gamma": 10.0,\n  "alpha": 0.95,\n  "common": {"context":32,"horizon":20,"dt":0.003968253968253968,"stock_cost":0.001,"hidden_size":32,"batch_size":256,"learning_rate":0.001,"block_updates":8,"gate_active_seconds":180,"checkpoint_seconds":[0,60,120,180],"adaptive_active_seconds":600,"adaptive_checkpoint_seconds":[0,200,400,600],"position_bound":2},\n  "families": [\n    {"name":"A","easy":{"kappa":8,"theta":[0.04,0.04],"xi":[0,0],"v0":0.04,"leverage":0},"base":{"kappa":6,"theta":[0.04,0.04],"xi":[0.35,0.35],"v0":0.04,"leverage":0.15},"hard":{"kappa":8,"theta":[0.02,0.06],"xi":[0.25,0.65],"v0":0.04,"leverage":0.15},"base_probabilities":[0,1,0],"generator_initial_probabilities":[0.01,0.98,0.01]},\n    {"name":"B","easy":{"kappa":8,"theta":[0.032,0.032],"xi":[0.04,0.04],"v0":0.032,"leverage":0.15},"base":{"kappa":6,"theta":[0.04,0.04],"xi":[0.30,0.30],"v0":0.04,"leverage":0.15},"hard":{"kappa":8,"theta":[0.015,0.065],"xi":[0.30,0.80],"v0":0.04,"leverage":0.15},"base_probabilities":[0.05,0.90,0.05],"generator_initial_probabilities":[0.05,0.90,0.05]}\n  ],\n  "choice_rule":"Evaluate both frozen families; A preferred if both screen and freshly confirm. No added candidate or parameter/threshold changes after outcomes.",\n  "hard_regime":"Uniform initial regime in {0,1}; one opposite-regime switch uniformly at trading intervals 6,7,8,9,10, persistent thereafter. Initial regime and switch time hidden. Ten equally likely joint scenarios.",\n  "correlation":0,\n  "leverage":"Synthetic, not calibrated: L=1+amplitude*tanh(-2*log(S/anchor)), strictly between 0.85 and 1.15. Anchor is start-of-context spot during context and fixed inception spot during hedging; this predictable contract-inception reset is identical across domains.",\n  "simulation":"Daily log-Euler stock with current local variance; independent exact CIR transition over dt via Poisson-gamma noncentral-chi-square sampling; regime coefficients frozen per interval. Continuous variance, not a quantized market. Physical equals pricing dynamics. Explicit daily approximation to continuous-time SLV, with a two-substep refinement diagnostic.",\n  "contract":"ATM European call struck at observed inception spot after 32 context days, 20 daily hedge intervals. Normalize spot, marks, payoff and gains by inception spot; cash zero rate. Initial and liquidation costs assigned once.",\n  "information":"GRU sees causal daily log return, inception-relative log moneyness (zero during context), remaining time, prior holding, trading flag. Five float32 features, GRU32 and bounded linear head. No marks, latent variance/regime, component label, switch date or future inputs.",\n  "pricing":"Known domain model specification, unknown realized variance and regime; family B base marks filter over entire mixture. Deterministic joint scenario/variance filter with exact CIR bin-transition probabilities and lognormal observation likelihoods; daily price table by backward conditional expectation, stock Gauss-Hermite and linear log-spot interpolation. Mark at t uses returns through t only. Independent doubled-grid/filter and nested forward Monte Carlo checks. Easy A analytic BS marks and diagnostic delta.",\n  "pricing_grid":{"variance_nodes":96,"variance_max":0.6,"log_spot_nodes":801,"log_spot_min":-1.2,"log_spot_max":1.2,"hermite_nodes":32,"refinement_factor":2},\n  "minimum_improvement":{"easy":0.0002,"hard":0.0005},\n  "threshold_calibration":"Same absolute thresholds as predeclared previous campaign, roughly 3%/4% of historical baseline SD 0.0064463/0.0116544; not selected using new outcomes.",\n  "numerical_tolerances":{"accounting_absolute":1e-10,"cash_translation":1e-12,"mark_refinement_rms":0.00002,"mark_refinement_max":0.0001,"nested_mc_standard_errors":4,"martingale_standard_errors":4,"integrated_variance_max_ratio":1.35},\n  "gate":"Both paired 95% bootstrap lower endpoints above minimum target gains. Specialist easy below hard in daily-error RMS SD, terminal SD, centered entropic and centered CVaR. Bootstrap lower endpoint for hard-minus-easy daily SD and terminal SD positive. Numerical validation passes before training; all accounting arrays rechecked independently. Reject contradictory terminal difficulty. Then fresh confirmation using same rules.",\n  "evaluation":{"selection_episodes":2048,"screen_episodes":8192,"confirmation_episodes":8192,"paired_bootstrap_replicates":400,"validation_episodes":4096,"pricing_validation_paths":128,"pricing_mc_paths":65536},\n  "namespaces":{"initialization":1,"training":2,"selection":3,"screen":4,"confirmation":5,"feedback":6,"monitor":7,"final":8,"bootstrap":9,"validation":10,"pricing_reference":11},\n  "precision_repair":"Cast positions to float64 before all differences, costs, gains and reductions. Old gate formed turnover/liquidation terms partly in float32. Independent NumPy reconstruction checks every exported dataset at stricter 1e-10; no tolerance relaxation.",\n  "preservation":"Old pilots and entropic-20261018 remain immutable. Prior authorized cleanup deleted some pilot data; no recovery claim. New raw outputs and resumable state in separate namespace."\n}\n')
OUT=pathlib.Path(os.environ['JOB_OUTPUT_DIR']); OUT.mkdir(parents=True,exist_ok=True)
C,H,DT=32,20,1/252

def save(path,data):
    path=pathlib.Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp'); tmp.write_text(json.dumps(data,indent=2,allow_nan=False)); tmp.replace(path)

def archive(path,**arrays):
    path=pathlib.Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f,**jax.device_get(arrays))
    tmp.replace(path)

def key(ns,i=0): return jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(D['training_seed']),ns),i)
def sync(x): return jax.block_until_ready(x)
def risk(x): return (logsumexp(D['gamma']*x)-jnp.log(x.size))/D['gamma']
def bs(s,w):
    root=jnp.sqrt(jnp.maximum(w,1e-30)); d=jnp.log(s)/root
    return jnp.where(w>1e-20,s*ndtr(d+root/2)-ndtr(d-root/2),jnp.maximum(s-1,0))

def scenario_parameters(family):
    configs=[family['easy'],family['base']]+[family['hard']]*10
    return {name:jnp.array([x[name] for x in configs],dtype=jnp.float64) for name in ['kappa','theta','xi','v0','leverage']}

def regime(s,t):
    initial=jnp.maximum(s-2,0)//5; switch=C+6+jnp.maximum(s-2,0)%5
    return jnp.where(s<2,0,jnp.where(t>=switch,1-initial,initial)).astype(jnp.int32)

def probabilities(family,domain):
    p={'base':family['base_probabilities'],'easy':[1,0,0],'hard':[0,0,1]}[domain]
    return jnp.array(p[:2]+[p[2]/10]*10,dtype=jnp.float64)

def cir_step(k,v,kap,theta,xi,dt):
    # Exact CIR marginal draw with frozen coefficients, independent of spot shock.
    safe=jnp.where(xi>0,xi,1.); decay=jnp.exp(-kap*dt)
    scale=safe**2*(1-decay)/(4*kap); df=4*kap*theta/safe**2
    a,b=jax.random.split(k); count=jax.random.poisson(a,v*decay/(2*scale),shape=v.shape)
    draw=2*scale*jax.random.gamma(b,df/2+count,shape=v.shape,dtype=jnp.float64)
    return jnp.where(xi>0,draw,theta+(v-theta)*decay)

def make_sim(family,substeps=1):
    p=scenario_parameters(family)
    @eqx.filter_jit
    def simulate(k,prob,n):
        ks,kn=jax.random.split(k); ids=jax.random.choice(ks,12,(n,),p=prob)
        kap=p['kappa'][ids]; lev=p['leverage'][ids]
        def day(carry,inputs):
            s,v,anchor=carry; t,k=inputs; anchor=jnp.where(t==C,s,anchor)
            z=regime(ids,t); theta=p['theta'][ids,z]; xi=p['xi'][ids,z]
            def substep(state,sk):
                s,v,iv=state; a,b=jax.random.split(sk)
                local=1+lev*jnp.tanh(-2*jnp.log(s/anchor)); w=local**2*v*DT/substeps
                ns=s*jnp.exp(-w/2+jnp.sqrt(w)*jax.random.normal(a,(n,),dtype=jnp.float64))
                nv=cir_step(b,v,kap,theta,xi,DT/substeps)
                return (ns,nv,iv+w),None
            (s,v,iv),_=jax.lax.scan(substep,(s,v,jnp.zeros(n)),jax.random.split(k,substeps))
            return (s,v,anchor),(s,v,iv)
        initial=(jnp.ones(n),p['v0'][ids],jnp.ones(n))
        _,(s,v,iv)=jax.lax.scan(day,initial,(jnp.arange(C+H),jax.random.split(kn,C+H)))
        return dict(spot=jnp.concatenate([jnp.ones((n,1)),s.T],1),variance=jnp.concatenate([p['v0'][ids,None],v.T],1),integrated_variance=iv.T,scenario=ids,switch=jnp.where(ids>=2,6+(ids-2)%5,-1))
    return simulate

class Pricer:
    """Joint hidden scenario/variance filter; stock-only daily observations.
    CIR transition cells approximate the continuous variance distribution.
    The market simulator itself has continuous variance and uses no pricing grid.
    """
    def __init__(self,family,factor=1):
        self.family=family; self.params=scenario_parameters(family)
        g=D['pricing_grid']; nv=g['variance_nodes']*factor
        self.v=jnp.linspace(0,np.sqrt(g['variance_max']),nv)**2+1e-10
        self.x=jnp.linspace(g['log_spot_min'],g['log_spot_max'],(g['log_spot_nodes']-1)*factor+1)
        nodes,weights=np.polynomial.hermite.hermgauss(g['hermite_nodes']*factor)
        self.nodes=jnp.array(nodes*np.sqrt(2)); self.weights=jnp.array(weights/np.sqrt(np.pi))
        vv=np.asarray(self.v); bounds=np.r_[0,(vv[1:]+vv[:-1])/2,np.inf]
        transitions=[]
        for i in range(12):
            regimes=[]
            for z in range(2):
                kap=float(self.params['kappa'][i]); theta=float(self.params['theta'][i,z]); xi=float(self.params['xi'][i,z])
                if xi==0:
                    target=theta+(vv-theta)*np.exp(-kap*DT)
                    rows=np.zeros((nv,nv)); hi=np.clip(np.searchsorted(vv,target),1,nv-1); frac=np.clip((target-vv[hi-1])/(vv[hi]-vv[hi-1]),0,1)
                    rows[np.arange(nv),hi-1]=1-frac; rows[np.arange(nv),hi]=frac
                else:
                    decay=np.exp(-kap*DT); scale=xi**2*(1-decay)/(4*kap)
                    cdf=ncx2.cdf(bounds[None,:]/scale,4*kap*theta/xi**2,vv[:,None]*decay/scale)
                    rows=np.diff(cdf,axis=-1); rows=np.maximum(rows,0); rows/=rows.sum(-1,keepdims=True)
                regimes.append(rows)
            transitions.append(regimes)
        self.transitions=jnp.asarray(np.array(transitions)); self.tables=self.build_tables()
        self.mark_batch=eqx.filter_jit(self._mark_batch)
    def transition(self,t):
        ids=jnp.arange(12); return self.transitions[ids,regime(ids,t)]
    def build_tables(self):
        x,v=self.x,self.v; spot=jnp.exp(x); lev=self.params['leverage']
        w=(1+lev[:,None,None]*jnp.tanh(-2*x[None,:,None]))**2*v[None,None,:]*DT
        terminal=jnp.broadcast_to(jnp.maximum(spot[None,:,None]-1,0),(12,x.size,v.size))
        values=[terminal]
        # Last date integrates kink analytically; earlier dates interpolate in spot.
        @jax.jit
        def backward(nxt,t):
            future=jnp.einsum('sxj,sij->sxi',nxt,self.transition(C+t))
            def quadrature(acc,zw):
                z,weight=zw; newx=x[None,:,None]-w/2+jnp.sqrt(w)*z
                ix=jnp.clip(jnp.searchsorted(x,newx)-1,0,x.size-2)
                frac=(jnp.exp(newx)-spot[ix])/(spot[ix+1]-spot[ix])
                sc=jnp.arange(12)[:,None,None]; vi=jnp.arange(v.size)[None,None,:]
                interp=(1-frac)*future[sc,ix,vi]+frac*future[sc,ix+1,vi]
                interp=jnp.where(newx<x[0],0,jnp.where(newx>x[-1],jnp.exp(newx)-1,interp))
                return acc+weight*interp,None
            out,_=jax.lax.scan(quadrature,jnp.zeros_like(nxt),(self.nodes,self.weights))
            return out
        last=bs(spot[None,:,None],w); values.append(last)
        for t in reversed(range(H-1)): values.append(sync(backward(values[-1],jnp.int32(t))))
        return jnp.stack(values[::-1],axis=1)
    def initial(self,prob,n):
        ids=jnp.arange(12); hi=jnp.clip(jnp.searchsorted(self.v,self.params['v0']),1,self.v.size-1)
        frac=(self.params['v0']-self.v[hi-1])/(self.v[hi]-self.v[hi-1])
        prior=jnp.zeros((12,self.v.size)).at[ids,hi-1].set(prob*(1-frac)).at[ids,hi].set(prob*frac)
        return jnp.broadcast_to(prior,(n,)+prior.shape)
    def _mark_batch(self,spot,prob):
        n=spot.shape[0]; p=self.initial(prob,n); inc=spot[:,C]
        def valuation(p,t):
            xx=jnp.log(spot[:,t]/inc); ix=jnp.clip(jnp.searchsorted(self.x,xx)-1,0,self.x.size-2)
            frac=(jnp.exp(xx)-jnp.exp(self.x[ix]))/(jnp.exp(self.x[ix+1])-jnp.exp(self.x[ix]))
            vals=self.tables[:,jnp.clip(t-C,0,H),:,:]
            interpolated=(1-frac[None,:,None])*vals[:,ix,:]+frac[None,:,None]*vals[:,ix+1,:]
            mark=jnp.einsum('nsv,snv->n',p,interpolated)
            return jnp.where(t==C+H,jnp.maximum(spot[:,t]/inc-1,0),mark)
        def step(p,t):
            anchor=jnp.where(t>=C,inc,1.); x=jnp.log(spot[:,t]/anchor)
            local=1+self.params['leverage'][None,:]*jnp.tanh(-2*x[:,None])
            w=local[:,:,None]**2*self.v[None,None,:]*DT
            r=jnp.log(spot[:,t+1]/spot[:,t])[:,None,None]
            ll=-.5*(jnp.log(2*jnp.pi*w)+(r+w/2)**2/w)
            lp=jnp.where(p>0,jnp.log(jnp.maximum(p,1e-300))+ll,-jnp.inf)
            posterior=jnp.exp(lp-logsumexp(lp,axis=(1,2),keepdims=True))
            updated=jnp.einsum('nsi,sij->nsj',posterior,self.transition(t))
            return updated,valuation(updated,t+1)
        _,marks=jax.lax.scan(step,p,jnp.arange(C+H))
        return marks.T[:,C-1:]
    def marks(self,spot,prob):
        # Pure BS target is analytical; mixed domains retain all components.
        if self.family['name']=='A' and float(prob[0])==1:
            rel=spot[:,C:]/spot[:,C,None]
            return bs(rel,.04*jnp.arange(H,-1,-1)[None,:]*DT)
        return self.mark_batch(spot,prob)

class Hedger(eqx.Module):
    cell:eqx.nn.GRUCell
    head:eqx.nn.Linear
    def __init__(self):
        a,b=jax.random.split(key(1)); self.cell=eqx.nn.GRUCell(5,32,key=a,dtype=jnp.float32); self.head=eqx.nn.Linear(32,1,key=b,dtype=jnp.float32)

def rollout(model,spot,marks):
    def one(s,m):
        inc=s[C]; normalized=s[C:]/inc
        def step(carry,t):
            hidden,prev=carry; active=t>=C
            features=jnp.array([10*jnp.log(s[t]/s[jnp.maximum(t-1,0)]),jnp.where(active,jnp.log(s[t]/inc),0),jnp.where(active,(C+H-t)/H,0),prev,active.astype(jnp.float32)],jnp.float32)
            hidden=model.cell(features,hidden); pos=jnp.where(active,2*jnp.tanh(model.head(hidden)[0]),jnp.float32(0))
            return (hidden,pos),pos
        _,positions=jax.lax.scan(step,(jnp.zeros(32,jnp.float32),jnp.float32(0)),jnp.arange(C+H))
        # Critical precision repair: convert BEFORE differences and cost arithmetic.
        h=positions[C:].astype(jnp.float64); delta=jnp.diff(jnp.concatenate([jnp.zeros(1,jnp.float64),h]))
        costs=.001*jnp.abs(delta)*normalized[:-1]; costs=costs.at[-1].add(.001*jnp.abs(h[-1])*normalized[-1])
        gains=h*jnp.diff(normalized); payoff=jnp.maximum(normalized[-1]-1,0)
        return dict(loss=payoff-gains.sum()+costs.sum(),daily=jnp.diff(m)-gains+costs,costs=costs,positions=h,turnover=jnp.abs(delta).sum()+jnp.abs(h[-1]),v0=m[0])
    return jax.vmap(one)(spot,marks)
EVALUATE=eqx.filter_jit(rollout)
OPT=optax.chain(optax.clip_by_global_norm(1.),optax.adam(.001,eps_root=1e-12))
@eqx.filter_jit
def block(model,state,spots):
    p,static=eqx.partition(model,eqx.is_inexact_array)
    def step(carry,s):
        p,state=carry
        loss,grad=jax.value_and_grad(lambda p:risk(rollout(eqx.combine(p,static),s,jnp.zeros((s.shape[0],H+1)))['loss']))(p)
        update,state=OPT.update(grad,state,p)
        return (eqx.apply_updates(p,update),state),(loss,optax.global_norm(grad))
    (p,state),logs=jax.lax.scan(step,(p,state),spots)
    return eqx.combine(p,static),state,logs

def metrics(raw):
    x=jnp.array(raw['loss']); n=x.size; ordered=jnp.sort(x)[::-1]; mass=.05*n; count=int(mass)
    cvar=(ordered[:count].sum()+(mass-count)*ordered[count])/mass; mean=x.mean(); w=jax.nn.softmax(D['gamma']*x)
    daily=np.asarray(raw['daily'])
    return dict(entropic=float(risk(x)),cvar95=float(cvar),var95=float(jnp.quantile(x,.95)),mean=float(mean),terminal_sd=float(x.std(ddof=1)),centered_entropic=float(risk(x)-mean),centered_cvar95=float(cvar-mean),daily_error_std=float(np.sqrt(daily.var(0,ddof=1).mean())),daily_mean=daily.mean(0).tolist(),daily_sd=daily.std(0,ddof=1).tolist(),initial_liability_mean=float(np.mean(raw['v0'])),cost_mean=float(np.mean(np.sum(raw['costs'],axis=-1))),turnover_mean=float(np.mean(raw['turnover'])),sample_count=n,finite_count=int(jnp.isfinite(x).sum()),exponential_ess=float(1/jnp.sum(w*w)),max_exponential_weight=float(w.max()))

def accounting(raw,spot,marks):
    # Independent NumPy reconstruction, on worker host; strict float64 throughout.
    rel=np.asarray(spot,dtype=np.float64)[:,C:]/np.asarray(spot,dtype=np.float64)[:,C,None]
    h=np.asarray(raw['positions'],dtype=np.float64); changes=np.diff(np.concatenate([np.zeros((len(h),1)),h],1),axis=1)
    costs=.001*np.abs(changes)*rel[:,:-1]; costs[:,-1]+=.001*np.abs(h[:,-1])*rel[:,-1]
    gains=h*np.diff(rel,axis=1); loss=np.maximum(rel[:,-1]-1,0)-gains.sum(1)+costs.sum(1)
    daily=np.diff(np.asarray(marks),axis=1)-gains+costs
    errors=dict(loss=float(np.max(np.abs(loss-raw['loss']))),daily=float(np.max(np.abs(daily-raw['daily']))),costs=float(np.max(np.abs(costs-raw['costs']))),telescoping=float(np.max(np.abs(np.sum(raw['daily'],1)-raw['loss']+raw['v0']))))
    assert max(errors.values())<D['numerical_tolerances']['accounting_absolute'],errors
    return errors


def validate(family,pricer,path):
    sim=make_sim(family); fine=Pricer(family,2); model=Hedger(); diagnostics={}; all_pass=True
    for domain in ['easy','base','hard']:
        prob=probabilities(family,domain); batch=sim(key(10,0),prob,4096); sync(batch)
        rel=batch['spot'][:,C:]/batch['spot'][:,C,None]; terminal=rel[:,-1]
        martingale_z=float(jnp.abs(terminal.mean()-1)/(terminal.std(ddof=1)/jnp.sqrt(terminal.size)))
        marks=pricer.marks(batch['spot'][:128],prob); reference=fine.marks(batch['spot'][:128],prob); sync(reference)
        err=np.asarray(marks-reference); rms=float(np.sqrt(np.mean(err**2))); maximum=float(np.max(np.abs(err)))
        raw=jax.device_get(EVALUATE(model,batch['spot'][:128],marks)); accounts=accounting(raw,batch['spot'][:128],marks)
        future=batch['spot'][:128].at[:,C+10:].multiply(1.1)
        causal=np.array_equal(np.asarray(EVALUATE(model,future,marks)['positions'])[:,:10],raw['positions'][:,:10])
        irrelevant=np.array_equal(np.asarray(EVALUATE(model,batch['spot'][:128],marks+10)['positions']),raw['positions'])
        iv=np.asarray(batch['integrated_variance'][:,C:]).sum(1)
        # Fresh doubled time-resolution simulation; report sampling uncertainty.
        refined=make_sim(family,2)(key(10,1),prob,4096); sync(refined)
        payoff=np.maximum(np.asarray(terminal)-1,0); finer=np.maximum(np.asarray(refined['spot'][:,-1]/refined['spot'][:,C])-1,0)
        sampling_se=float(np.sqrt(payoff.var(ddof=1)/len(payoff)+finer.var(ddof=1)/len(finer)))
        discretization_delta=float(finer.mean()-payoff.mean())
        passed=martingale_z<4 and rms<=2e-5 and maximum<=1e-4 and causal and irrelevant
        all_pass &= passed
        diagnostics[domain]=dict(passed=passed,martingale_z=martingale_z,mark_refinement_rms=rms,mark_refinement_max=maximum,accounting=accounts,causality=causal,marks_excluded_from_policy=irrelevant,mean_integrated_variance=float(iv.mean()),mean_initial_value=float(np.mean(marks[:,0])),time_refinement_payoff_change=discretization_delta,time_refinement_sampling_se=sampling_se)
        archive(path/f'validation-{domain}.npz',spot=batch['spot'][:128],marks=marks,reference_marks=reference,**raw)
        save(path/'validation-progress.json',diagnostics)
        print('VALIDATION',family['name'],domain,diagnostics[domain],flush=True)
    ratio=max(x['mean_integrated_variance'] for x in diagnostics.values())/min(x['mean_integrated_variance'] for x in diagnostics.values())
    all_pass &= ratio<=1.35
    # Entropic translation, CVaR fractional mass, finite differentiation and resume.
    assert float(abs(risk(jnp.array([0.,1.,2.,3.])-.13)-risk(jnp.array([0.,1.,2.,3.]))+.13))<1e-12
    state=OPT.init(eqx.filter(model,eqx.is_inexact_array)); b=sim(key(10,22),probabilities(family,'base'),2048)['spot'].reshape(8,256,C+H+1)
    trained,st,logs=sync(block(model,state,b)); finite=bool(jnp.all(jnp.isfinite(logs[0]))) and bool(jnp.all(jnp.isfinite(logs[1])))
    tmp=path/'validation-checkpoint.pkl.tmp'; tmp.write_bytes(pickle.dumps(jax.device_get((trained,st,key(2),0,0)))); tmp.replace(path/'validation-checkpoint.pkl')
    restored=pickle.loads((path/'validation-checkpoint.pkl').read_bytes()); next1=sync(block(trained,st,b)); next2=sync(block(restored[0],restored[1],b))
    resume=all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(jax.tree.leaves(next1),jax.tree.leaves(next2)))
    report=dict(passed=all_pass and finite and resume,domains=diagnostics,integrated_variance_ratio=ratio,finite_gradients=finite,resume_exact=resume,device=str(jax.devices()),limitations='Time-discretization comparison is Monte Carlo with reported uncertainty; doubled grid checks numerical convergence, independent nested price reference still required before training.')
    save(path/'validation.json',report); return report

def main():
    assert all(d.platform=='gpu' for d in jax.devices()),'GPU required'
    save(OUT/'design.json',D)
    parser=argparse.ArgumentParser(); parser.add_argument('--mode',default='validate',choices=['validate']); args=parser.parse_args()
    reports=[]
    for family in D['families']:
        path=OUT/family['name']; path.mkdir(exist_ok=True)
        print('BUILD PRICER',family['name'],flush=True); start=time.monotonic(); pricer=Pricer(family)
        print('PRICER BUILT',family['name'],time.monotonic()-start,flush=True)
        reports.append(validate(family,pricer,path))
    save(OUT/'validation_summary.json',dict(passed=all(x['passed'] for x in reports),families=reports,training_authorized=False,reason='Independent nested reference and complete protocol tests required before training'))

if __name__=='__main__':
    try: main()
    except Exception:
        save(OUT/'failure.json',dict(traceback=traceback.format_exc())); raise
