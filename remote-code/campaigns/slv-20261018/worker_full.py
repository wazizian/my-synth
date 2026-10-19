RESIDUAL_PROTOCOL = {'declared_at': '2026-10-18T11:34:00Z', 'timing': 'Before any scientific diagnostic training or screening outcomes; first-family numerical validation underway', 'purpose': 'Apply plan requirement to reject difficulty ordering explained by initial liability values or materially contradicted by terminal residual diagnostics. With hidden volatility, V0 varies across contexts, so subtract V0 per episode in this supplementary review.', 'quantities': ['SD(X-V0)', 'centered entropic risk of X-V0', 'centered CVaR95 of X-V0'], 'comparison': 'Corresponding selected easy/hard specialists, all existing paired-evaluation data retained; hard minus easy. Independent whole-episode resampling across domains, 400 draws, conditional on trained policies.', 'material_contradiction': '95% interval upper endpoint below a conservative numerical margin: minus2e-4 for SD, minus4e-4 for centered entropic/CVaR. Derived from fixed1e-4 per-domain maximum refinement tolerance (centering doubles risk sensitivity). Also report unadjusted statistical contradictions. This does not relax primary gate conditions.', 'numerical_noise': 'Interpret differences alongside recorded mark refinement errors; do not infer optimal irreducible risk from specialists. Report V0 mean/SD, mean loss, costs and raw/centered metrics separately. No threshold or severity adjustment.', 'execution': 'All raw-array calculations occur on the GPU worker host. Supervisor reviews before any adaptive phase; no final-test data used.', 'declared_at_actual': '2026-10-18T11:36:58.183736+00:00', 'pretraining_evidence': 'Latest job terminated in objective_validation before either family diagnostic training'}
AUDIT_SOURCE = '"""Run only on GPU worker host; read-only scientific outputs, atomic audit reports."""\nimport os,time,json,hashlib,pathlib,subprocess\nimport numpy as np\nfrom scipy.special import logsumexp\nROOT=pathlib.Path(os.environ.get(\'JOB_OUTPUT_DIR\',\'/workspace/job/outputs\'))\nsubprocess.run([\'nvidia-smi\',\'--query-gpu=name\',\'--format=csv,noheader\'],check=True)\nseen={}; failures=[]; maximum={}; checks=0\n\ndef check(name,error,tolerance,path):\n    global checks\n    checks+=1;error=float(error);maximum[name]=max(maximum.get(name,0),error)\n    if not np.isfinite(error) or error>tolerance: failures.append(dict(file=str(path.relative_to(ROOT)),check=name,error=error,tolerance=tolerance))\n\ndef inspect(path):\n    global checks\n    with np.load(path) as f:\n        if not {\'loss\',\'daily\',\'positions\',\'costs\'}.issubset(f.files):return\n        a={name:f[name] for name in f.files};x=a[\'loss\'];daily=a[\'daily\'];n=len(x)\n        check(\'finite\',sum(np.size(v)-np.isfinite(v).sum() for v in a.values()),0,path)\n        check(\'position_bound\',max(0,float(np.max(np.abs(a[\'positions\'])))-2),1e-12,path)\n        check(\'telescoping\',np.max(np.abs(daily.sum(1)-x+a[\'v0\'])),1e-10,path)\n        if \'spot\' in a and \'marks\' in a:\n            normalized=a[\'spot\'][:,32:]/a[\'spot\'][:,32,None];h=a[\'positions\'].astype(np.float64)\n            delta=np.diff(np.column_stack([np.zeros(n),h]),axis=1);costs=.001*np.abs(delta)*normalized[:,:-1];costs[:,-1]+=.001*np.abs(h[:,-1])*normalized[:,-1]\n            gains=h*np.diff(normalized,axis=1);loss=np.maximum(normalized[:,-1]-1,0)-gains.sum(1)+costs.sum(1)\n            check(\'loss_reconstruction\',np.max(np.abs(loss-x)),1e-10,path)\n            check(\'daily_reconstruction\',np.max(np.abs(np.diff(a[\'marks\'],axis=1)-gains+costs-daily)),1e-10,path)\n            check(\'cost_reconstruction\',np.max(np.abs(costs-a[\'costs\'])),1e-10,path)\n            check(\'payoff_mark\',np.max(np.abs(a[\'marks\'][:,-1]-np.maximum(normalized[:,-1]-1,0))),1e-12,path)\n            check(\'turnover\',np.max(np.abs(np.abs(delta).sum(1)+np.abs(h[:,-1])-a[\'turnover\'])),1e-10,path)\n        mp=path.parent/\'metrics.json\'\n        if mp.exists() and path.stem in [\'base\',\'easy\',\'hard\']:\n            metrics=json.loads(mp.read_text())[path.stem]\n            entropic=(logsumexp(10*x)-np.log(n))/10; sortedx=np.sort(x)[::-1]; mass=.05*n;count=int(mass);tail=(sortedx[:count].sum()+(mass-count)*sortedx[count])/mass\n            reconstructed={\'entropic\':entropic,\'cvar95\':tail,\'mean\':x.mean(),\'terminal_sd\':x.std(ddof=1),\'centered_entropic\':entropic-x.mean(),\'centered_cvar95\':tail-x.mean(),\'daily_error_std\':np.sqrt(daily.var(0,ddof=1).mean()),\'cost_mean\':a[\'costs\'].sum(1).mean(),\'turnover_mean\':a[\'turnover\'].mean()}\n            for name,value in reconstructed.items():check(\'metric_\'+name,abs(value-metrics[name]),1e-10,path)\n            check(\'sample_count\',abs(metrics[\'sample_count\']-n),0,path)\n            check(\'finite_count\',abs(metrics[\'finite_count\']-n),0,path)\n\ndef snapshot(complete):\n    result=dict(complete=complete,passed=not failures,checks=checks,files=len(seen),max_errors=maximum,failures=failures,file_sha256=seen,scope=\'Independent NumPy/SciPy accounting and metric reconstruction on the same GPU worker host; does not by itself certify scientific design or continuous-time pricing.\')\n    tmp=ROOT/\'independent_audit.json.tmp\';tmp.write_text(json.dumps(result,indent=2));tmp.replace(ROOT/\'independent_audit.json\')\n\nstart=time.monotonic()\nwhile time.monotonic()-start<11000:\n    for p in sorted(ROOT.rglob(\'*.npz\')):\n        name=str(p.relative_to(ROOT))\n        if name in seen:continue\n        # Metrics may be committed immediately after arrays: defer until available.\n        if p.stem in [\'base\',\'easy\',\'hard\'] and not (p.parent/\'metrics.json\').exists():continue\n        inspect(p); seen[name]=hashlib.sha256(p.read_bytes()).hexdigest()\n    complete=True\n    snapshot(complete)\n    if complete or (ROOT/\'failure.json\').exists():break\n    time.sleep(3)\nprint(json.dumps(dict(checks=checks,files=len(seen),failures=len(failures))),flush=True)\nassert not failures, failures\n'
NUMERICS = {'revision': 3, 'reason': 'Initial GPU validation: production96 variance/801 log-spot/32 GH versus doubled reference exceeded fixed mark tolerances (SLV RMS2.87e-5..4.57e-5, max1.09e-4..1.43e-4). No scientific training ran.', 'production_resolution_factor': 2, 'reference_resolution_factor': 4, 'production_grid': {'variance_nodes': 192, 'log_spot_nodes': 1601, 'hermite_nodes': 64}, 'reference_grid': {'variance_nodes': 384, 'log_spot_nodes': 3201, 'hermite_nodes': 128}, 'unchanged': 'Market parameters, distribution weights, seed, risk objective, specialization thresholds, residual difficulty criteria and numerical tolerances remain frozen. This is a numerical implementation refinement only.', 'additional_repairs': 'Pricer is a PyTree so large pricing arrays are dynamic GPU arguments rather than compiler constants; no mathematical change. Independent forward-pricing/filter and entropic-attribution tests added. All accounting arithmetic remains float64 with1e-10 tolerance.', 'previous_job': '20261018T104734Z-e5bb8dcb', 'policy_and_optimizer_dtype': 'float64', 'precision_probe_run': '20261018T114331Z-c12878c1', 'precision_evidence': 'Same weights/paths: float32 FD errors worsen as perturbation shrinks; float64 AD-FD error converges to5.26e-12. All learners promoted identically before scientific training; initialization numeric values unchanged.'}
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

D = json.loads('{\n  "campaign_id": "slv-20261018",\n  "authorization": "User set active goal execute the plan after reviewing updated BS/SLV plans; new execution authorization, not reuse of previous budget",\n  "training_seed": 20261017,\n  "gamma": 10.0,\n  "alpha": 0.95,\n  "common": {\n    "context": 32,\n    "horizon": 20,\n    "dt": 0.003968253968253968,\n    "stock_cost": 0.001,\n    "hidden_size": 32,\n    "batch_size": 256,\n    "learning_rate": 0.001,\n    "block_updates": 8,\n    "gate_active_seconds": 180,\n    "checkpoint_seconds": [\n      0,\n      60,\n      120,\n      180\n    ],\n    "adaptive_active_seconds": 600,\n    "adaptive_checkpoint_seconds": [\n      0,\n      200,\n      400,\n      600\n    ],\n    "position_bound": 2,\n    "policy_dtype": "float64",\n    "optimizer_dtype": "float64"\n  },\n  "families": [\n    {\n      "name": "A",\n      "easy": {\n        "kappa": 8,\n        "theta": [\n          0.04,\n          0.04\n        ],\n        "xi": [\n          0,\n          0\n        ],\n        "v0": 0.04,\n        "leverage": 0\n      },\n      "base": {\n        "kappa": 6,\n        "theta": [\n          0.04,\n          0.04\n        ],\n        "xi": [\n          0.35,\n          0.35\n        ],\n        "v0": 0.04,\n        "leverage": 0.15\n      },\n      "hard": {\n        "kappa": 8,\n        "theta": [\n          0.02,\n          0.06\n        ],\n        "xi": [\n          0.25,\n          0.65\n        ],\n        "v0": 0.04,\n        "leverage": 0.15\n      },\n      "base_probabilities": [\n        0,\n        1,\n        0\n      ],\n      "generator_initial_probabilities": [\n        0.01,\n        0.98,\n        0.01\n      ]\n    },\n    {\n      "name": "B",\n      "easy": {\n        "kappa": 8,\n        "theta": [\n          0.032,\n          0.032\n        ],\n        "xi": [\n          0.04,\n          0.04\n        ],\n        "v0": 0.032,\n        "leverage": 0.15\n      },\n      "base": {\n        "kappa": 6,\n        "theta": [\n          0.04,\n          0.04\n        ],\n        "xi": [\n          0.3,\n          0.3\n        ],\n        "v0": 0.04,\n        "leverage": 0.15\n      },\n      "hard": {\n        "kappa": 8,\n        "theta": [\n          0.015,\n          0.065\n        ],\n        "xi": [\n          0.3,\n          0.8\n        ],\n        "v0": 0.04,\n        "leverage": 0.15\n      },\n      "base_probabilities": [\n        0.05,\n        0.9,\n        0.05\n      ],\n      "generator_initial_probabilities": [\n        0.05,\n        0.9,\n        0.05\n      ]\n    }\n  ],\n  "choice_rule": "Evaluate both frozen families; A preferred if both screen and freshly confirm. No added candidate or parameter/threshold changes after outcomes.",\n  "hard_regime": "Uniform initial regime in {0,1}; one opposite-regime switch uniformly at trading intervals 6,7,8,9,10, persistent thereafter. Initial regime and switch time hidden. Ten equally likely joint scenarios.",\n  "correlation": 0,\n  "leverage": "Synthetic, not calibrated: L=1+amplitude*tanh(-2*log(S/anchor)), strictly between 0.85 and 1.15. Anchor is start-of-context spot during context and fixed inception spot during hedging; this predictable contract-inception reset is identical across domains.",\n  "simulation": "Daily log-Euler stock with current local variance; independent exact CIR transition over dt via Poisson-gamma noncentral-chi-square sampling; regime coefficients frozen per interval. Continuous variance, not a quantized market. Physical equals pricing dynamics. Explicit daily approximation to continuous-time SLV, with a two-substep refinement diagnostic.",\n  "contract": "ATM European call struck at observed inception spot after 32 context days, 20 daily hedge intervals. Normalize spot, marks, payoff and gains by inception spot; cash zero rate. Initial and liquidation costs assigned once.",\n  "information": "GRU sees causal daily log return, inception-relative log moneyness (zero during context), remaining time, prior holding, trading flag. Five float64 features, GRU32 and bounded linear head. No marks, latent variance/regime, component label, switch date or future inputs.",\n  "pricing": "Known domain model specification, unknown realized variance and regime; family B base marks filter over entire mixture. Deterministic joint scenario/variance filter with exact CIR bin-transition probabilities and lognormal observation likelihoods; daily price table by backward conditional expectation, stock Gauss-Hermite and linear log-spot interpolation. Mark at t uses returns through t only. Independent doubled-grid/filter and nested forward Monte Carlo checks. Easy A analytic BS marks and diagnostic delta.",\n  "pricing_grid": {\n    "variance_nodes": 96,\n    "variance_max": 0.6,\n    "log_spot_nodes": 801,\n    "log_spot_min": -1.2,\n    "log_spot_max": 1.2,\n    "hermite_nodes": 32,\n    "refinement_factor": 2\n  },\n  "minimum_improvement": {\n    "easy": 0.0002,\n    "hard": 0.0005\n  },\n  "threshold_calibration": "Same absolute thresholds as predeclared previous campaign, roughly 3%/4% of historical baseline SD 0.0064463/0.0116544; not selected using new outcomes.",\n  "numerical_tolerances": {\n    "accounting_absolute": 1e-10,\n    "cash_translation": 1e-12,\n    "mark_refinement_rms": 2e-05,\n    "mark_refinement_max": 0.0001,\n    "nested_mc_standard_errors": 4,\n    "martingale_standard_errors": 4,\n    "integrated_variance_max_ratio": 1.35\n  },\n  "gate": "Both paired 95% bootstrap lower endpoints above minimum target gains. Specialist easy below hard in daily-error RMS SD, terminal SD, centered entropic and centered CVaR. Bootstrap lower endpoint for hard-minus-easy daily SD and terminal SD positive. Numerical validation passes before training; all accounting arrays rechecked independently. Reject contradictory terminal difficulty. Then fresh confirmation using same rules.",\n  "evaluation": {\n    "selection_episodes": 2048,\n    "screen_episodes": 8192,\n    "confirmation_episodes": 8192,\n    "paired_bootstrap_replicates": 400,\n    "validation_episodes": 4096,\n    "pricing_validation_paths": 128,\n    "pricing_mc_paths": 65536\n  },\n  "namespaces": {\n    "initialization": 1,\n    "training": 2,\n    "selection": 3,\n    "screen": 4,\n    "confirmation": 5,\n    "feedback": 6,\n    "monitor": 7,\n    "final": 8,\n    "bootstrap": 9,\n    "validation": 10,\n    "pricing_reference": 11,\n    "residual_review": 12\n  },\n  "precision_repair": "Cast positions to float64 before all differences, costs, gains and reductions. Old gate formed turnover/liquidation terms partly in float32. Independent NumPy reconstruction checks every exported dataset at stricter 1e-10; no tolerance relaxation.",\n  "preservation": "Old pilots and entropic-20261018 remain immutable. Prior authorized cleanup deleted some pilot data; no recovery claim. New raw outputs and resumable state in separate namespace.",\n  "shared_feasibility_revision": "Before all scientific training, promote the original float32-initialized weights to float64 without changing their numerical values; all features, learner parameters and optimizer states are float64 for every method. Matched-weights GPU precision probe showed float32 finite-difference instability and float64 convergence to autodiff. Architecture, seed, optimizer settings, runtime and scientific thresholds unchanged."\n}\n')
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

class Pricer(eqx.Module):
    """Joint hidden scenario/variance filter; stock-only daily observations.
    CIR transition cells approximate the continuous variance distribution.
    The market simulator itself has continuous variance and uses no pricing grid.
    """
    family:dict
    params:dict
    v:jax.Array
    x:jax.Array
    nodes:jax.Array
    weights:jax.Array
    transitions:jax.Array
    tables:jax.Array
    def __init__(self,family,factor=2):
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
    @eqx.filter_jit
    def mark_batch(self,spot,prob,details=False):
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
            return updated,(valuation(updated,t+1),updated.sum(-1),jnp.einsum('nsv,v->n',updated,self.v))
        _,(marks,scenario_mass,variance_mean)=jax.lax.scan(step,p,jnp.arange(C+H))
        if details: return dict(marks=marks.T[:,C-1:],scenario_mass=scenario_mass.transpose(1,0,2)[:,C-1:],variance_mean=variance_mean.T[:,C-1:])
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
        self.cell=jax.tree.map(lambda x:x.astype(jnp.float64) if eqx.is_inexact_array(x) else x,self.cell)
        self.head=jax.tree.map(lambda x:x.astype(jnp.float64) if eqx.is_inexact_array(x) else x,self.head)

def rollout(model,spot,marks):
    def one(s,m):
        policy_dtype=model.head.weight.dtype
        inc=s[C]; normalized=s[C:]/inc
        def step(carry,t):
            hidden,prev=carry; active=t>=C
            features=jnp.array([10*jnp.log(s[t]/s[jnp.maximum(t-1,0)]),jnp.where(active,jnp.log(s[t]/inc),0),jnp.where(active,(C+H-t)/H,0),prev,active.astype(policy_dtype)],policy_dtype)
            hidden=model.cell(features,hidden); pos=jnp.where(active,2*jnp.tanh(model.head(hidden)[0]),jnp.asarray(0,dtype=policy_dtype))
            return (hidden,pos),pos
        _,positions=jax.lax.scan(step,(jnp.zeros(32,policy_dtype),jnp.asarray(0,dtype=policy_dtype)),jnp.arange(C+H))
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
        def objective(p):
            losses=rollout(eqx.combine(p,static),s,jnp.zeros((s.shape[0],H+1)))['loss']; weights=jax.nn.softmax(D['gamma']*losses)
            return risk(losses),(1/jnp.sum(weights**2),weights.max())
        (loss,concentration),grad=jax.value_and_grad(objective,has_aux=True)(p)
        update,state=OPT.update(grad,state,p)
        return (eqx.apply_updates(p,update),state),(loss,optax.global_norm(grad),concentration[0],concentration[1])
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
    sim=make_sim(family); fine=Pricer(family,4); model=Hedger(); diagnostics={}; all_pass=True
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
        if domain=='hard':
            posterior=jax.device_get(pricer.mark_batch(batch['spot'],prob,True)); ids=np.asarray(batch['scenario']); initial=(ids-2)//5
            tracking={}
            for day in [0,15]:
                high=np.asarray(posterior['scenario_mass'][:,day,7:].sum(-1) if day==0 else posterior['scenario_mass'][:,day,2:7].sum(-1))
                truth=initial if day==0 else 1-initial
                tracking[str(day)]=dict(brier=float(np.mean((high-truth)**2)),uninformed_brier=.25,accuracy=float(np.mean((high>=.5)==truth)),posterior_variance_mean=float(posterior['variance_mean'][:,day].mean()))
            diagnostics[domain]['regime_tracking']=tracking
            archive(path/'filter_tracking.npz',scenario_mass=posterior['scenario_mass'],variance_mean=posterior['variance_mean'],true_scenario=ids,true_variance=batch['variance'][:,C:])
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



# Independent validation routines, appended before main during packaging.
def numpy_filter_reference(pricer,spot,prob):
    from scipy.special import logsumexp as scipy_lse
    spot=np.asarray(spot); prob=np.asarray(prob); vv=np.asarray(pricer.v); grid=np.asarray(pricer.x)
    params={k:np.asarray(v) for k,v in pricer.params.items()}; trans=np.asarray(pricer.transitions); table=np.asarray(pricer.tables)
    n=len(spot); post=np.zeros((n,12,len(vv)))
    for s in range(12):
        hi=np.clip(np.searchsorted(vv,params['v0'][s]),1,len(vv)-1); w=(params['v0'][s]-vv[hi-1])/(vv[hi]-vv[hi-1])
        post[:,s,hi-1]=prob[s]*(1-w); post[:,s,hi]=prob[s]*w
    answer=[]
    for t in range(C+H):
        x=np.log(spot[:,t]/(spot[:,C] if t>=C else 1))
        scale=((1+params['leverage'][None,:]*np.tanh(-2*x[:,None]))**2)[:,:,None]*vv[None,None,:]*DT
        returns=np.log(spot[:,t+1]/spot[:,t])[:,None,None]
        ll=-.5*(np.log(2*np.pi*scale)+(returns+scale/2)**2/scale)
        with np.errstate(divide='ignore'): lp=np.log(post)+ll
        post=np.exp(lp-scipy_lse(lp,axis=(1,2),keepdims=True))
        for s in range(12):
            z=0 if s<2 else ((s-2)//5 if t<C+6+(s-2)%5 else 1-(s-2)//5)
            post[:,s,:]=post[:,s,:]@trans[s,z]
        if t+1>=C:
            x=np.log(spot[:,t+1]/spot[:,C]); marks=np.zeros(n)
            for i in range(n):
                ix=np.clip(np.searchsorted(grid,x[i])-1,0,len(grid)-2)
                f=(np.exp(x[i])-np.exp(grid[ix]))/(np.exp(grid[ix+1])-np.exp(grid[ix]))
                val=(1-f)*table[:,t+1-C,ix,:]+f*table[:,t+1-C,ix+1,:]
                marks[i]=np.sum(post[i]*val)
            if t+1==C+H: marks=np.maximum(spot[:,-1]/spot[:,C]-1,0)
            answer.append(marks)
    return np.array(answer).T

def independent_prices(family,pricer,path):
    params=scenario_parameters(family); results=[]
    # Latent-state prices: independent direct Monte Carlo, no table/filter calls.
    # It validates the backward solver; a separate NumPy filter validates updates.
    @eqx.filter_jit
    def mc(k,scenario,start,n):
        v=jnp.full((n,),.04); s=jnp.ones(n)
        def step(carry,inputs):
            s,v=carry; t,k=inputs; z=regime(jnp.int32(scenario),C+t); a,b=jax.random.split(k)
            local=1+params['leverage'][scenario]*jnp.tanh(-2*jnp.log(s)); w=local**2*v*DT
            nexts=s*jnp.exp(-w/2+jnp.sqrt(w)*jax.random.normal(a,(n,),dtype=jnp.float64))
            nextv=cir_step(b,v,params['kappa'][scenario],params['theta'][scenario,z],params['xi'][scenario,z],DT)
            return (nexts,nextv),None
        (s,_),_=jax.lax.scan(step,(s,v),(jnp.arange(start,H),jax.random.split(k,H-start)))
        # Deterministic half-delta control variate: E[S_T-1]=0 exactly.
        return jnp.maximum(s-1,0)-.5*(s-1)
    for scenario in [0,1,2,7]:
        for start in [0,10]:
            samples=sync(mc(key(11,scenario*20+start),scenario,start,65536))
            values=pricer.tables[scenario,start,pricer.x.size//2]
            table_value=float(jnp.interp(.04,pricer.v,values)); mean=float(samples.mean()); se=float(samples.std(ddof=1)/jnp.sqrt(samples.size))
            diff=abs(table_value-mean); passed=diff<=4*se+2e-5
            results.append(dict(scenario=scenario,date=start,table=table_value,mc_mean=mean,standard_error=se,difference=diff,passed=passed))
            save(path/'independent_pricing_progress.json',dict(latent_price_reference=results))
    sim=make_sim(family); filters=[]
    for domain in ['base','hard']:
        prob=probabilities(family,domain); spot=sim(key(11,100),prob,8)['spot']; actual=np.asarray(pricer.marks(spot,prob))
        reference=numpy_filter_reference(pricer,spot,prob); error=float(np.max(np.abs(actual-reference)))
        filters.append(dict(domain=domain,max_error=error,passed=error<1e-10))
    cir_checks=[]
    for scenario,z in [(1,0),(2,0),(2,1)]:
        kap=float(params['kappa'][scenario]); theta=float(params['theta'][scenario,z]); xi=float(params['xi'][scenario,z]); decay=np.exp(-kap*DT); scale=xi**2*(1-decay)/(4*kap)
        expected_mean,expected_variance=ncx2.stats(4*kap*theta/xi**2,.04*decay/scale,moments='mv'); expected_mean*=scale; expected_variance*=scale**2
        draws=sync(cir_step(key(11,900+scenario*2+z),jnp.full((131072,),.04),kap,theta,xi,DT))
        zz=abs(float(draws.mean())-expected_mean)/np.sqrt(expected_variance/draws.size)
        cir_checks.append(dict(scenario=scenario,regime=z,mean_z=float(zz),expected_mean=float(expected_mean),observed_mean=float(draws.mean()),expected_variance=float(expected_variance),observed_variance=float(draws.var()),passed=bool(zz<4)))
    report=dict(passed=all(x['passed'] for x in results+filters+cir_checks),cir_sampler=cir_checks,latent_price_reference=results,independent_filter=filters,scope='Independent forward continuous-variance MC checks backward pricing; NumPy likelihood/transition recursion checks JAX filter. MC tolerances include reported sampling uncertainty, not a claim of exact pricing.')
    save(path/'independent_pricing.json',report); return report


# Entropic teacher primitives. Validated on GPU before any adaptive execution.
# Weighted empirical risk normalizes by sum weights, preserving translation.
def weighted_risk(loss,weights):
    return (logsumexp(D['gamma']*loss+jnp.log(weights))-jnp.log(weights.sum()))/D['gamma']

def weighted_block(model,state,spots,weights):
    parameters,static=eqx.partition(model,eqx.is_inexact_array)
    def step(carry,inputs):
        p,state=carry; spot,w=inputs
        value,gradient=jax.value_and_grad(lambda p:weighted_risk(rollout(eqx.combine(p,static),spot,jnp.zeros((spot.shape[0],H+1)))['loss'],w))(p)
        update,state=OPT.update(gradient,state,p)
        return (eqx.apply_updates(p,update),state),value
    (p,state),losses=jax.lax.scan(step,(parameters,state),(spots,weights))
    return eqx.combine(p,static),state,losses

@eqx.filter_jit
def target_risk(model,hard,easy,scales,mixed):
    hard_r=risk(rollout(model,hard,jnp.zeros((hard.shape[0],H+1)))['loss'])
    if not mixed: return hard_r
    easy_r=risk(rollout(model,easy,jnp.zeros((easy.shape[0],H+1)))['loss'])
    return .5*(hard_r/scales[1]+easy_r/scales[0])

@eqx.filter_jit
def entropic_attribution(model,state,spots,hard,easy,scales,mixed):
    # Differentiate exact clipped-Adam eight-update trajectory, carrying moments.
    # No replacement by average exponentials, squared losses or detached optimizer.
    def after(weights):
        post,st,losses=weighted_block(model,state,spots,weights)
        return target_risk(post,hard,easy,scales,mixed),(post,st,losses)
    (value,post),gradient=eqx.filter_value_and_grad(after,has_aux=True)(jnp.ones(spots.shape[:2]))
    return post,-jax.lax.stop_gradient(gradient),value

@eqx.filter_jit
def intrinsic_rewards(model,historical,state,spot):
    p,static=eqx.partition(model,eqx.is_inexact_array); old=eqx.filter(historical,eqx.is_inexact_array)
    def adam_state(tree):
        if hasattr(tree,'nu') and hasattr(tree,'count'): return tree
        for element in tree:
            if hasattr(element,'nu') and hasattr(element,'count'): return element
            if isinstance(element,tuple):
                result=adam_state(element)
                if result is not None: return result
        return None
    adam=adam_state(state); correction=1-.999**jnp.maximum(adam.count,1)
    direction=jax.tree.map(lambda a,b,v:(a-b)*.001/(jnp.sqrt(v/correction)+1e-8),old,p,adam.nu)
    def losses(p): return rollout(eqx.combine(p,static),spot,jnp.zeros((spot.shape[0],H+1)))['loss']
    values,directional=eqx.filter_jvp(losses,(p,),(direction,))
    # Contributions to directional entropic gradient: softmax(gamma X)*dX.
    # Multiply by batch size only to put per-episode rewards on a mean scale.
    return jax.lax.stop_gradient(jnp.abs(values.size*jax.nn.softmax(D['gamma']*values)*directional))

GEN_OPT=optax.adam(.01)
@eqx.filter_jit
def generator_update(logits,state,ids,rewards,prior,selfplay=False):
    rewards=jax.lax.stop_gradient(rewards); ids=jax.lax.stop_gradient(ids)
    def objective(logits):
        prob=.95*jax.nn.softmax(logits)+.05*prior; logprob=jnp.log(prob)[ids]
        if selfplay:
            advantage=(rewards-rewards.mean())/(rewards.std()+1e-8)
            policy=-jnp.mean(advantage*logprob)-jnp.sum(rewards*logprob)/(rewards.sum()+1e-8)
        else: policy=-jnp.sum(rewards*logprob)
        return policy+.01*jnp.sum(prob*(jnp.log(prob)-jnp.log(prior)))
    loss,grad=jax.value_and_grad(objective)(logits); updates,state=GEN_OPT.update(grad,state,logits)
    return optax.apply_updates(logits,updates),state,dict(loss=loss,gradient_norm=optax.global_norm(grad))


def validate_objectives(family,path):
    x=jnp.array([-.04,.01,.02,.05]); w=jnp.array([.5,2.,1.,3.])
    translation=float(abs(weighted_risk(x-.13,w)-weighted_risk(x,w)+.13)); assert translation<1e-12
    grad=jax.grad(lambda w:weighted_risk(x,w))(jnp.ones_like(x))
    expected=(jax.nn.softmax(D['gamma']*x)-1/x.size)/D['gamma']
    derivative_error=float(jnp.max(jnp.abs(grad-expected))); assert derivative_error<1e-12
    dummy=dict(loss=np.arange(21.),daily=np.zeros((21,H)),v0=np.zeros(21),costs=np.zeros((21,H)),turnover=np.zeros(21))
    cvar=metrics(dummy)['cvar95']; expected_cvar=(20+.05*19)/1.05; assert abs(cvar-expected_cvar)<1e-12
    # Small exact same-optimizer unroll to test attribution, without training a method.
    sim=make_sim(family); model=Hedger(); state=OPT.init(eqx.filter(model,eqx.is_inexact_array))
    spots=sim(key(10,800),probabilities(family,'base'),64)['spot'].reshape(2,32,C+H+1)
    hard=sim(key(10,801),probabilities(family,'hard'),32)['spot']; easy=sim(key(10,802),probabilities(family,'easy'),32)['spot']
    scales=jnp.array([.01,.01]); post,rewards,value=sync(entropic_attribution(model,state,spots,hard,easy,scales,True))
    uniform=sync(weighted_block(model,state,spots,jnp.ones((2,32))))
    plain=sync(block(model,state,spots))
    weight_identity=max(float(jnp.max(jnp.abs(a-b))) for a,b in zip(jax.tree.leaves(uniform[:2]),jax.tree.leaves(plain[:2])))
    # Weight normalization implies a common per-batch weight direction has zero derivative.
    row_sum=float(jnp.max(jnp.abs(rewards.sum(1))))
    finite=bool(jnp.all(jnp.isfinite(rewards)))
    def after(weights):
        m,_,_=weighted_block(model,state,spots,weights)
        return target_risk(m,hard,easy,scales,True)
    direction=jnp.zeros((2,32)).at[0,0].set(1.)
    automatic=float(-rewards[0,0]);sweep=[]
    for delta in [.05,.01,.003,.001]:
        fd=float((after(jnp.ones((2,32))+delta*direction)-after(jnp.ones((2,32))-delta*direction))/(2*delta))
        sweep.append(dict(step=delta,finite_difference=fd,error=abs(fd-automatic)))
    tolerance=max(2e-5,abs(automatic)*.1)
    # Keep original error tolerance; require agreement at two adjacent resolutions.
    converged=any(a['error']<tolerance and b['error']<tolerance and abs(a['finite_difference']-b['finite_difference'])<tolerance for a,b in zip(sweep,sweep[1:]))
    passed=finite and weight_identity<1e-6 and row_sum<1e-5 and converged
    report=dict(passed=passed,weighted_cash_translation_error=translation,analytical_weight_derivative_error=derivative_error,cvar_fractional_mass_error=abs(cvar-expected_cvar),weighted_uniform_optimizer_identity_error=weight_identity,attribution_batch_weight_scale_derivative=row_sum,finite_attribution=finite,finite_difference_sweep=sweep,autodiff=automatic,finite_difference_tolerance=tolerance,two_adjacent_steps_agree=converged,scope='Small optimizer-unroll validation using diagnostic namespace only; original tolerance unchanged. Step-size refinement distinguishes nonlinear/nonsmooth finite differences and float32 roundoff. All scientific policies retain untouched initialization.')
    save(path/'objective_validation.json',report); assert passed,report
    return report


# Development datasets are generated once, immutable, paired across policies.
DATA_CACHE={}
def dataset(family,pricer,namespace,n,domain):
    token=(family['name'],namespace,n,domain)
    if token in DATA_CACHE: return DATA_CACHE[token]
    start=time.monotonic(); sim=make_sim(family); chunks=[]
    for i in range(n//256):
        b=sim(key(namespace,i),probabilities(family,domain),256)
        marks=pricer.marks(b['spot'],probabilities(family,domain)); sync(marks)
        chunks.append(jax.device_get(dict(**b,marks=marks)))
    data={name:np.concatenate([c[name] for c in chunks]) for name in chunks[0]}
    assert all(np.isfinite(x).all() for x in data.values())
    digest=hashlib.sha256(b''.join(np.ascontiguousarray(data[k]).tobytes() for k in sorted(data))).hexdigest()
    path=OUT/family['name']/'datasets'/f'{namespace}-{domain}-{n}.npz'; archive(path,**data)
    save(path.with_suffix('.json'),dict(sha256=digest,namespace=namespace,domain=domain,episodes=n,generation_seconds=time.monotonic()-start))
    DATA_CACHE[token]=(data,digest); return data,digest

def evaluate(model,family,pricer,namespace,n,path):
    path.mkdir(parents=True,exist_ok=True); results={}
    for domain in ['base','easy','hard']:
        data,digest=dataset(family,pricer,namespace,n,domain); pieces=[]
        for start in range(0,n,256):
            raw=EVALUATE(model,jnp.asarray(data['spot'][start:start+256]),jnp.asarray(data['marks'][start:start+256])); pieces.append(jax.device_get(raw))
        raw={name:np.concatenate([r[name] for r in pieces]) for name in pieces[0]}
        errors=accounting(raw,data['spot'],data['marks'])
        archive(path/(domain+'.npz'),**raw,spot=data['spot'],marks=data['marks'],switch=data['switch'])
        results[domain]=dict(**metrics(raw),dataset_sha256=digest,accounting=errors)
    save(path/'metrics.json',results); return results

def checkpoint(path,model,state,rng,active,steps,**extra):
    payload=dict(model=model,opt_state=state,rng=rng,active_seconds=active,steps=steps,**extra)
    tmp=path.with_name(path.name+'.tmp'); tmp.write_bytes(pickle.dumps(jax.device_get(payload))); tmp.replace(path)

def train_fixed(family,pricer,arm,domain):
    path=OUT/family['name']/arm; path.mkdir(parents=True,exist_ok=True)
    model=Hedger(); state=OPT.init(eqx.filter(model,eqx.is_inexact_array)); rng=key(2)
    save(path/'initialization.json',dict(seed=D['training_seed'],parameter_sha256=hashlib.sha256(b''.join(np.asarray(x).tobytes() for x in jax.tree.leaves(model))).hexdigest(),architecture='GRU(5,32), Linear(32,1), tanh bound2, float64 policy/optimizer/accounting; original float32 initial values preserved'))
    sim=make_sim(family); prob=probabilities(family,domain)
    def draw(k):
        k,sub=jax.random.split(k); return k,sim(sub,prob,2048)['spot'].reshape(8,256,C+H+1)
    compile_start=time.monotonic(); _,warm=draw(rng); sync(block(model,state,warm)); compile_seconds=time.monotonic()-compile_start
    active=0.; steps=0; best=float('inf'); chosen=None; evaluate_seconds=0.; io_seconds=0.; start=time.monotonic()
    for target in D['common']['checkpoint_seconds']:
        with (path/'training.jsonl').open('a') as stream:
            while active<target:
                tick=time.monotonic(); rng,spots=draw(rng); model,state,logs=sync(block(model,state,spots)); active+=time.monotonic()-tick; steps+=8
                assert bool(jnp.all(jnp.isfinite(logs[0]))) and bool(jnp.all(jnp.isfinite(logs[1])))
                stream.write(json.dumps(dict(active_seconds=active,learner_updates=steps,loss=np.asarray(logs[0]).tolist(),gradient_norm=np.asarray(logs[1]).tolist(),exponential_ess=np.asarray(logs[2]).tolist(),max_exponential_weight=np.asarray(logs[3]).tolist()))+'\n'); stream.flush()
        cp=path/f'checkpoint-{target}.pkl'; tick=time.monotonic(); checkpoint(cp,model,state,rng,active,steps); io_seconds+=time.monotonic()-tick
        tick=time.monotonic(); scores=evaluate(model,family,pricer,3,2048,path/f'selection-{target}'); evaluate_seconds+=time.monotonic()-tick
        if scores[domain]['entropic']<best: best=scores[domain]['entropic']; chosen=cp
        print('CHECKPOINT',family['name'],arm,target,'risk',scores[domain]['entropic'],flush=True)
    record=dict(checkpoint=str(chosen),criterion_domain=domain,score=best,active_seconds=active,learner_updates=steps,simulated_training_paths=steps*256,compile_seconds=compile_seconds,evaluation_and_dataset_seconds=evaluate_seconds,checkpoint_io_seconds=io_seconds,postcompile_wall_seconds=time.monotonic()-start,teacher_feedback_episodes=0)
    save(path/'selection.json',record); return pickle.loads(chosen.read_bytes())['model']

def assess(models,family,pricer,namespace,path):
    scores={arm:evaluate(model,family,pricer,namespace,8192,path/arm) for arm,model in models.items()}
    checks={}; intervals={}; n=8192
    for domain,arm in [('easy','easy_reference'),('hard','hard_reference')]:
        a=jnp.array(np.load(path/'baseline'/(domain+'.npz'))['loss']); b=jnp.array(np.load(path/arm/(domain+'.npz'))['loss'])
        @jax.jit
        def sample(k):
            idx=jax.random.randint(k,(n,),0,n); return risk(a[idx])-risk(b[idx])
        samples=jax.vmap(sample)(jax.random.split(key(9,namespace),400)); low,high=np.asarray(jnp.quantile(samples,jnp.array([.025,.975])))
        intervals[domain]=dict(improvement=float(risk(a)-risk(b)),low=float(low),high=float(high),minimum=D['minimum_improvement'][domain])
        checks[domain+'_specialization']=float(low)>D['minimum_improvement'][domain]
    easy=scores['easy_reference']['easy']; hard=scores['hard_reference']['hard']
    for metric in ['daily_error_std','terminal_sd','centered_entropic','centered_cvar95']:
        checks['difficulty_'+metric]=easy[metric]<hard[metric]
    er=np.load(path/'easy_reference'/'easy.npz'); hr=np.load(path/'hard_reference'/'hard.npz')
    ex,hx=jnp.array(er['loss']),jnp.array(hr['loss']); ed,hd=jnp.array(er['daily']),jnp.array(hr['daily'])
    @jax.jit
    def difficulty(k):
        a,b=jax.random.split(k); ei=jax.random.randint(a,(n,),0,n); hi=jax.random.randint(b,(n,),0,n)
        return jnp.array([hx[hi].std(ddof=1)-ex[ei].std(ddof=1),jnp.sqrt(hd[hi].var(0,ddof=1).mean())-jnp.sqrt(ed[ei].var(0,ddof=1).mean())])
    # Whole paths resampled. Different domains independently resampled.
    differences=jax.vmap(difficulty)(jax.random.split(key(9,100+namespace),400))
    for i,name in enumerate(['terminal_sd','daily_error_std']):
        low,high=np.asarray(jnp.quantile(differences[:,i],jnp.array([.025,.975])))
        intervals['difficulty_'+name]=dict(low=float(low),high=float(high))
        checks['difficulty_interval_'+name]=float(low)>0
    result=dict(passed=all(checks.values()),checks=checks,intervals=intervals,scores=scores,uncertainty='Episode resampling conditional on one trained seed, not training uncertainty')
    save(path/'gate.json',result); print('GATE',family['name'],namespace,result['checks'],flush=True); return result

def bs_diagnostic(family,pricer):
    if family['name']!='A': return
    data,_=dataset(family,pricer,4,8192,'easy'); rel=data['spot'][:,C:]/data['spot'][:,C,None]
    w=.04*np.arange(H,0,-1)[None,:]*DT
    from scipy.special import ndtr as numpy_ndtr
    h=numpy_ndtr((np.log(rel[:,:-1])+w/2)/np.sqrt(w)); changes=np.diff(np.concatenate([np.zeros((len(h),1)),h],1),axis=1)
    costs=.001*np.abs(changes)*rel[:,:-1]; costs[:,-1]+=.001*np.abs(h[:,-1])*rel[:,-1]
    gains=h*np.diff(rel,axis=1); raw=dict(loss=np.maximum(rel[:,-1]-1,0)-gains.sum(1)+costs.sum(1),daily=np.diff(data['marks'],axis=1)-gains+costs,costs=costs,positions=h,turnover=np.abs(changes).sum(1)+np.abs(h[:,-1]),v0=data['marks'][:,0])
    accounting(raw,data['spot'],data['marks']); archive(OUT/'A'/'bs_delta_diagnostic.npz',**raw); save(OUT/'A'/'bs_delta_diagnostic.json',metrics(raw))

def run_gate():
    results=[]; accepted=[]; pricers={}
    # Validate BOTH configurations before spending any diagnostic training budget.
    for family in D['families']:
        path=OUT/family['name']; path.mkdir(parents=True,exist_ok=True); pricer=Pricer(family)
        numerical=validate(family,pricer,path); independent=independent_prices(family,pricer,path)
        objectives=validate_objectives(family,path)
        if not numerical['passed'] or not independent['passed'] or not objectives['passed']:
            save(OUT/'numerical_block.json',dict(family=family['name'],reason='Numerical prerequisites failed; repair required. Not a scientific rejection.'))
            raise RuntimeError('Numerical validation did not pass; no training allowed')
        pricers[family['name']]=pricer
    for family in D['families']:
        path=OUT/family['name']; pricer=pricers[family['name']]
        models={arm:train_fixed(family,pricer,arm,domain) for arm,domain in [('baseline','base'),('easy_reference','easy'),('hard_reference','hard')]}
        bs_diagnostic(family,pricer)
        screen=assess(models,family,pricer,4,path/'screen'); residual=marked_review(path/'screen')
        entry=dict(family=family['name'],primary_screen_passed=screen['passed'],residual_review_passed=not residual['material_contradiction'],screen_passed=screen['passed'] and not residual['material_contradiction'])
        if entry['screen_passed']:
            confirmation=assess(models,family,pricer,5,path/'confirmation'); residual=marked_review(path/'confirmation')
            entry['primary_confirmation_passed']=confirmation['passed'];entry['confirmation_passed']=confirmation['passed'] and not residual['material_contradiction']
            if entry['confirmation_passed']: accepted.append(family['name'])
        results.append(entry)
        save(OUT/'gate_summary.json',dict(complete=False,accepted=None,results=results))
    choice=next((name for name in ['A','B'] if name in accepted),None)
    save(OUT/'gate_summary.json',dict(complete=True,accepted=choice,results=results,choice_rule='A first if both confirm',adaptive_authorized=choice is not None))


def marked_review(path):
    from scipy.special import logsumexp as scipy_logsumexp
    easy=np.load(path/'easy_reference'/'easy.npz');hard=np.load(path/'hard_reference'/'hard.npz')
    ex=easy['loss']-easy['v0'];hx=hard['loss']-hard['v0']
    def summaries(x):
        mean=x.mean();ordered=np.sort(x)[::-1];mass=.05*len(x);count=int(mass)
        return np.array([x.std(ddof=1),(scipy_logsumexp(10*x)-np.log(len(x)))/10-mean,(ordered[:count].sum()+(mass-count)*ordered[count])/mass-mean])
    point=summaries(hx)-summaries(ex);rng=np.random.default_rng(np.random.SeedSequence([20261017,12,0 if path.name=='screen' else 1]));draws=[]
    for _ in range(400):draws.append(summaries(hx[rng.integers(0,len(hx),len(hx))])-summaries(ex[rng.integers(0,len(ex),len(ex))]))
    intervals=np.quantile(draws,[.025,.975],axis=0);margins=np.array([2e-4,4e-4,4e-4])
    details={name:dict(difference=float(point[i]),low=float(intervals[0,i]),high=float(intervals[1,i]),numerical_margin=float(margins[i])) for i,name in enumerate(['marked_sd','marked_centered_entropic','marked_centered_cvar95'])}
    result=dict(material_contradiction=bool(np.any(intervals[1]<-margins)),statistical_contradiction=bool(np.any(intervals[1]<0)),differences=details,easy_marked=summaries(ex).tolist(),hard_marked=summaries(hx).tolist(),easy_v0_mean=float(easy['v0'].mean()),hard_v0_mean=float(hard['v0'].mean()),easy_v0_sd=float(easy['v0'].std(ddof=1)),hard_v0_sd=float(hard['v0'].std(ddof=1)),scope='Supplementary plan-required residual review declared before training; primary gate reported separately. Conservative margins reflect fixed mark-error tolerance. Conditional episode uncertainty only.')
    save(path/'residual_review.json',result);return result


# Conditional final eight-arm campaign. Must only be invoked after confirmed gate.
def generator_prior(family):
    p=family['generator_initial_probabilities']; return jnp.array(p[:2]+[p[2]/10]*10)

def train_adaptive(family,pricer,arm,scales):
    path=OUT/'final_training'/arm; path.mkdir(parents=True,exist_ok=True)
    model=Hedger(); state=OPT.init(eqx.filter(model,eqx.is_inexact_array)); rng=key(2)
    prior=generator_prior(family); logits=jnp.log(prior); genstate=GEN_OPT.init(logits); baseline=jnp.array(0.); histories=[]
    active=0.; steps=0; outer=0; feedback_count={'easy':0,'hard':0}; compile_seconds=0.; eval_seconds=0.; io_seconds=0.; sampling_seconds=0.; learner_teacher_seconds=0.
    sim=make_sim(family); mixed=arm.endswith('mixed'); dpg=arm.startswith('dpg'); intrinsic=arm=='selfplay'
    feedback={}
    for domain in ([] if intrinsic else (['hard','easy'] if mixed else ['hard'])):
        feedback[domain]=dataset(family,pricer,6,8192,domain)[0]['spot']
    def target_batch(round):
        offset=(round%32)*256
        hard=jnp.asarray(feedback['hard'][offset:offset+256]) if not intrinsic else jnp.zeros((256,C+H+1))
        easy=jnp.asarray(feedback['easy'][offset:offset+256]) if mixed else jnp.zeros((256,C+H+1))
        return hard,easy
    def update(model,state,logits,genstate,baseline,historical,spots,ids,hard,easy,round):
        if dpg:
            post,rewards,target_after=entropic_attribution(model,state,spots,hard,easy,scales,mixed)
            new_model,new_state,losses=post
        elif intrinsic:
            rewards=jax.vmap(lambda spot:intrinsic_rewards(model,historical,state,spot))(spots)
            new_model,new_state,logs=block(model,state,spots); losses=logs[0]
        else:
            before=target_risk(model,hard,easy,scales,mixed)
            new_model,new_state,logs=block(model,state,spots); losses=logs[0]
            after=target_risk(new_model,hard,easy,scales,mixed); improvement=jax.lax.stop_gradient(before-after)
            rewards=jnp.full(ids.shape,improvement-baseline); baseline=.9*baseline+.1*improvement
        if not intrinsic or round>0:
            logits,genstate,gm=generator_update(logits,genstate,ids,rewards,prior,intrinsic)
        else: gm=dict(loss=jnp.array(0.),gradient_norm=jnp.array(0.))
        return new_model,new_state,logits,genstate,baseline,dict(loss=losses,reward_mean=rewards.mean(),reward_sd=rewards.std(),teacher_loss=gm['loss'],teacher_gradient_norm=gm['gradient_norm'])
    # Compile exact shapes on disposable state and validation namespace only.
    start=time.monotonic(); warm=sim(key(10,900),prior,2048); spots=warm['spot'].reshape(8,256,C+H+1); ids=warm['scenario'].reshape(8,256)
    hard,easy=target_batch(0); sync(update(model,state,logits,genstate,baseline,model,spots,ids,hard,easy,1)); compile_seconds=time.monotonic()-start
    best=float('inf'); chosen=None; startwall=time.monotonic()
    for target in [0,200,400,600]:
        with (path/'training.jsonl').open('a') as stream:
            while active<target:
                tick=time.monotonic(); rng,sub=jax.random.split(rng); prob=.95*jax.nn.softmax(logits)+.05*prior
                batch=sim(sub,prob,2048); sync(batch); sampling_seconds+=time.monotonic()-tick
                spots=batch['spot'].reshape(8,256,C+H+1); ids=batch['scenario'].reshape(8,256); hard,easy=target_batch(outer)
                if intrinsic:
                    histories.append(jax.device_get(model)); historical=histories[outer//2]
                else: historical=model
                update_start=time.monotonic()
                model,state,logits,genstate,baseline,logs=sync(update(model,state,logits,genstate,baseline,historical,spots,ids,hard,easy,outer))
                learner_teacher_seconds+=time.monotonic()-update_start; active+=time.monotonic()-tick; steps+=8; outer+=1
                for domain in feedback_count: feedback_count[domain]+=256 if domain in feedback else 0
                assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(logs))
                record={k:np.asarray(v).tolist() for k,v in logs.items()}; record.update(active_seconds=active,learner_updates=steps,outer_round=outer,probabilities=np.asarray(.95*jax.nn.softmax(logits)+.05*prior).tolist(),feedback_count=feedback_count.copy())
                stream.write(json.dumps(record)+'\n');stream.flush()
        tick=time.monotonic(); cp=path/f'checkpoint-{target}.pkl'; checkpoint(cp,model,state,rng,active,steps,logits=logits,genstate=genstate,ema=baseline,histories=histories,outer=outer,feedback_count=feedback_count); io_seconds+=time.monotonic()-tick
        tick=time.monotonic(); scores=evaluate(model,family,pricer,3,2048,path/f'selection-{target}'); evaluate(model,family,pricer,7,2048,path/f'monitor-{target}'); eval_seconds+=time.monotonic()-tick
        score=.5*(scores['easy']['entropic']/float(scales[0])+scores['hard']['entropic']/float(scales[1]))
        if score<best:best=score;chosen=cp
    save(path/'selection.json',dict(checkpoint=str(chosen),score=best,criterion='balanced scaled easy/hard entropic',active_seconds=active,learner_updates=steps,simulated_training_paths=steps*256,feedback_count=feedback_count,compile_seconds=compile_seconds,evaluation_seconds=eval_seconds,checkpoint_io_seconds=io_seconds,sampling_seconds=sampling_seconds,learner_and_teacher_seconds=learner_teacher_seconds,postcompile_wall_seconds=time.monotonic()-startwall))
    return pickle.loads(chosen.read_bytes())['model']

def continue_fixed(family,pricer,arm,domain):
    previous=OUT/family['name']/arm; path=OUT/'final_training'/arm; path.mkdir(parents=True,exist_ok=True)
    payload=pickle.loads((previous/'checkpoint-180.pkl').read_bytes())
    model,state,rng=payload['model'],payload['opt_state'],payload['rng']; active=payload['active_seconds']; steps=payload['steps']
    initial=pickle.loads((previous/'checkpoint-0.pkl').read_bytes())['model']
    cp0=path/'checkpoint-0.pkl'; cp0.write_bytes((previous/'checkpoint-0.pkl').read_bytes())
    scores=evaluate(initial,family,pricer,3,2048,path/'selection-0'); evaluate(initial,family,pricer,7,2048,path/'monitor-0')
    best=scores[domain]['entropic']; chosen=cp0; sim=make_sim(family); prob=probabilities(family,domain)
    def draw(k):
        k,sub=jax.random.split(k); return k,sim(sub,prob,2048)['spot'].reshape(8,256,C+H+1)
    tick=time.monotonic(); _,b=draw(rng); sync(block(model,state,b)); compile_seconds=time.monotonic()-tick; eval_seconds=0.; io_seconds=0.; start=time.monotonic()
    for target in [200,400,600]:
        with (path/'training.jsonl').open('a') as stream:
            while active<target:
                tick=time.monotonic(); rng,b=draw(rng); model,state,logs=sync(block(model,state,b)); active+=time.monotonic()-tick;steps+=8
                assert all(np.isfinite(np.asarray(x)).all() for x in logs)
                stream.write(json.dumps(dict(active_seconds=active,learner_updates=steps,loss=np.asarray(logs[0]).tolist(),gradient_norm=np.asarray(logs[1]).tolist()))+'\n');stream.flush()
        tick=time.monotonic(); cp=path/f'checkpoint-{target}.pkl'; checkpoint(cp,model,state,rng,active,steps);io_seconds+=time.monotonic()-tick
        tick=time.monotonic(); scores=evaluate(model,family,pricer,3,2048,path/f'selection-{target}');evaluate(model,family,pricer,7,2048,path/f'monitor-{target}');eval_seconds+=time.monotonic()-tick
        if scores[domain]['entropic']<best:best=scores[domain]['entropic'];chosen=cp
    save(path/'selection.json',dict(checkpoint=str(chosen),score=best,criterion=domain,active_seconds=active,learner_updates=steps,simulated_training_paths=steps*256,compile_seconds=compile_seconds,evaluation_seconds=eval_seconds,checkpoint_io_seconds=io_seconds,additional_wall_seconds=time.monotonic()-start,continued_from=str(previous/'checkpoint-180.pkl'),feedback_count={'easy':0,'hard':0}))
    return pickle.loads(chosen.read_bytes())['model']

def run_final():
    summary=json.loads((OUT/'gate_summary.json').read_text()); assert summary['complete'] and summary['accepted']
    family=next(f for f in D['families'] if f['name']==summary['accepted']); pricer=Pricer(family)
    screen=json.loads((OUT/family['name']/'screen'/'baseline'/'metrics.json').read_text())
    scales=jnp.array([max(.001,screen[d]['terminal_sd']) for d in ['easy','hard']])
    save(OUT/'accepted_freeze.json',dict(family=family,design_sha256=hashlib.sha256(json.dumps(D,sort_keys=True).encode()).hexdigest(),gamma=D['gamma'],scales=np.asarray(scales).tolist(),scale_rule='baseline screening terminal SD floored at.001',final_checkpoints=[0,200,400,600],seed=D['training_seed']))
    models={}
    for arm,domain in [('baseline','base'),('easy_reference','easy'),('hard_reference','hard')]: models[arm]=continue_fixed(family,pricer,arm,domain)
    for arm in ['dpg_hard','soar_hard','selfplay','dpg_mixed','soar_mixed']: models[arm]=train_adaptive(family,pricer,arm,scales)
    selections={arm:json.loads((OUT/'final_training'/arm/'selection.json').read_text()) for arm in models}
    # This file is committed before namespace8 is ever sampled.
    save(OUT/'final_selections_frozen.json',selections)
    final_datasets={domain:dataset(family,pricer,8,16384,domain)[1] for domain in ['base','easy','hard']}
    save(OUT/'final_dataset_hashes_frozen.json',final_datasets)
    scores={arm:evaluate(model,family,pricer,8,16384,OUT/'final_test'/arm) for arm,model in models.items()}
    intervals={}
    for domain in ['base','easy','hard']:
        baseline=jnp.array(np.load(OUT/'final_test'/'baseline'/(domain+'.npz'))['loss'])
        for arm in models:
            if arm=='baseline':continue
            alternative=jnp.array(np.load(OUT/'final_test'/arm/(domain+'.npz'))['loss'])
            @jax.jit
            def comparison(k):
                idx=jax.random.randint(k,(16384,),0,16384);return risk(baseline[idx])-risk(alternative[idx])
            draws=jax.vmap(comparison)(jax.random.split(key(9,800),400));low,high=np.asarray(jnp.quantile(draws,jnp.array([.025,.975])))
            intervals[domain+'/'+arm]=dict(improvement=float(risk(baseline)-risk(alternative)),low=float(low),high=float(high))
    save(OUT/'final_summary.json',dict(scores=scores,paired_entropic_intervals=intervals,uncertainty='Conditional episode bootstrap, one training seed; no across-seed significance claim'))


def make_report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((OUT/'gate_summary.json').read_text())
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    lines=['# Entropic benchmark development gate','',
           'Single shared training seed 20261017. Paired bootstrap intervals are conditional on trained policies; they do not measure training-seed uncertainty.', '',
           'Accepted candidate: '+str(summary['accepted']), '',
           'Two predeclared BS/SLV families, domain-specific causal variance/regime filtering, stock-only trading, gamma=10, CVaR alpha=0.95. Positive pre-premium normalized losses.', '',
           'Training uses a finite-minibatch log-mean-exp estimator (batch 256), which is not exact population entropic risk. Selection and screening use separate random namespaces. Exponential concentration diagnostics are in metrics.json.', '',
           'Policy features contain only causal returns, moneyness, time, previous position and trading indicator. Liability marks and latent states are excluded. Initial trading costs enter interval zero; liquidation enters the last interval. Pricing uses an approximate variance-grid filter and backward integration checked against refined grids, independent filtering and forward Monte Carlo. GPU validation records actual approximation and accounting errors.', '']
    texrows=[]
    for candidate in summary['results']:
        name=candidate['family']; root=OUT/name
        lines.extend(['## '+name,''])
        for phase in ['screen','confirmation']:
            p=root/phase/'gate.json'
            if p.exists():
                result=json.loads(p.read_text());lines.extend([phase+': '+('PASS' if result['passed'] else 'FAIL'), '',json.dumps(result['checks']), '',json.dumps(result['intervals']), ''])
        lines.extend(['Overall screening outcome: '+str(candidate['screen_passed']),''])
        for phase in ['screen','confirmation']:
            review=root/phase/'residual_review.json'
            if review.exists():lines.extend(['Supplementary '+phase+' marked-residual review: '+review.read_text(),''])
        for domain in ['base','easy','hard']:
            for metric in ['entropic','cvar95','daily_error_std','terminal_sd','centered_entropic','centered_cvar95']:
                fig,ax=plt.subplots()
                for arm in ['baseline','easy_reference','hard_reference']:
                    points=[json.loads((root/arm/f'selection-{t}'/'metrics.json').read_text())[domain][metric] for t in [0,60,120,180]]
                    ax.plot([0,60,120,180],points,marker='o',label=arm)
                ax.set(xlabel='Active training seconds (scheduled checkpoint)',ylabel=metric,title=f'{name}: {domain} selection');ax.legend();fig.tight_layout();fig.savefig(figdir/f'{name}-{domain}-{metric}.png');plt.close(fig)
            for kind in ['daily','event']:
                fig,axes=plt.subplots(2,1,figsize=(8,7))
                for arm in ['baseline','easy_reference','hard_reference']:
                    raw=np.load(root/'screen'/arm/(domain+'.npz'));daily=raw['daily'];sw=raw['switch']
                    if kind=='daily':
                        x=np.arange(H);means=daily.mean(0);sd=daily.std(0,ddof=1)
                    else:
                        x=np.arange(-6,8);means=[];sd=[]
                        for delta in x:
                            indices=sw+delta;valid=(sw>=0)&(indices>=0)&(indices<H)
                            v=daily[np.arange(len(sw))[valid],indices[valid]]
                            means.append(v.mean() if len(v) else np.nan);sd.append(v.std(ddof=1) if len(v)>1 else np.nan)
                    axes[0].plot(x,means,label=arm);axes[1].plot(x,sd,label=arm)
                axes[0].set(ylabel='Mean daily error',title=f'{name}: {domain} {kind}');axes[1].set(ylabel='Cross-episode SD',xlabel='Trading day' if kind=='daily' else 'Day relative to hidden switch');axes[0].legend();fig.tight_layout();fig.savefig(figdir/f'{name}-{domain}-{kind}.png');plt.close(fig)
        for arm in ['baseline','easy_reference','hard_reference']:
            events=[json.loads(x) for x in (root/arm/'training.jsonl').read_text().splitlines()]
            fig,ax=plt.subplots();ax.plot([x['active_seconds'] for x in events],[np.mean(x['loss']) for x in events]);ax.set(xlabel='Active training seconds',ylabel='Mean block minibatch entropic loss',title=f'{name}: {arm}');fig.tight_layout();fig.savefig(figdir/f'{name}-{arm}-training.png');plt.close(fig)
            scores=json.loads((root/'screen'/arm/'metrics.json').read_text())
            for d,m in scores.items():
                texrows.append(f"{name} & {arm.replace('_',' ')} & {d} & {m['entropic']:.6f} & {m['cvar95']:.6f} & {m['daily_error_std']:.6f} \\\\")
    if summary['accepted'] is None:
        lines.extend(['## Outcome','','No candidate passed the predeclared gate. This is an uninformative benchmark for the proposed adaptive comparison. Adaptive training and final-test evaluation were not run. The failure is retained without changing the candidate menu or thresholds.'])
    elif not (OUT/'final_summary.json').exists():
        lines.extend(['## Remaining work','','Benchmark confirmation passed. Freeze accepted configuration and datasets, implement/audit entropic teachers, and run all eight arms before claiming campaign completion.'])
    if (OUT/'final_summary.json').exists():
        final=json.loads((OUT/'final_summary.json').read_text())
        lines.extend(['## Final eight-arm evaluation','','All selections frozen before final episodes. Equal600-active-second budgets, one shared seed. Full scores and paired conditional entropic intervals in final_summary.json.','',json.dumps(final['paired_entropic_intervals'],indent=2),''])
        for domain in ['base','easy','hard']:
            for metric in ['entropic','cvar95','daily_error_std','terminal_sd','centered_entropic','centered_cvar95']:
                fig,ax=plt.subplots(figsize=(10,5))
                for arm in final['scores']:
                    values=[json.loads((OUT/'final_training'/arm/f'selection-{t}'/'metrics.json').read_text())[domain][metric] for t in [0,200,400,600]]
                    ax.plot([0,200,400,600],values,marker='o',label=arm)
                ax.set(xlabel='Active seconds',ylabel=metric,title='Final training selection: '+domain);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(figdir/f'final-{domain}-{metric}.png');plt.close(fig)
        for arm in final['scores']:
            events=[json.loads(line) for line in (OUT/'final_training'/arm/'training.jsonl').read_text().splitlines()]
            if arm in ['baseline','easy_reference','hard_reference']:
                candidate=summary['accepted']; prior=[json.loads(line) for line in (OUT/candidate/arm/'training.jsonl').read_text().splitlines()];events=prior+events
            fig,ax=plt.subplots();ax.plot([e['active_seconds'] for e in events],[np.mean(e['loss']) for e in events]);ax.set(xlabel='Active seconds',ylabel='Block minibatch entropic loss',title=arm);fig.tight_layout();fig.savefig(figdir/f'final-{arm}-training.png');plt.close(fig)
        for domain in ['base','easy','hard']:
            for kind in ['daily','event']:
                fig,axes=plt.subplots(2,1,figsize=(10,7))
                for arm in final['scores']:
                    raw=np.load(OUT/'final_test'/arm/(domain+'.npz'));daily=raw['daily'];switch=raw['switch']
                    if kind=='daily':xx=np.arange(H);means=daily.mean(0);deviation=daily.std(0,ddof=1)
                    else:
                        xx=np.arange(-6,10);means=[];deviation=[]
                        for delta in xx:
                            index=switch+delta;valid=(switch>=0)&(index>=0)&(index<H);values=daily[np.arange(len(switch))[valid],index[valid]]
                            means.append(values.mean() if len(values) else np.nan);deviation.append(values.std(ddof=1) if len(values)>1 else np.nan)
                    axes[0].plot(xx,means,label=arm);axes[1].plot(xx,deviation,label=arm)
                axes[0].set(ylabel='Mean error',title=domain+' '+kind);axes[1].set(ylabel='Cross-episode SD',xlabel='Trading day' if kind=='daily' else 'Day relative to hidden switch');axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(figdir/f'final-{domain}-{kind}.png');plt.close(fig)
            fig,axes=plt.subplots(2,1,figsize=(10,7))
            for arm in final['scores']:
                raw=np.load(OUT/'final_test'/arm/(domain+'.npz'));axes[0].plot(raw['positions'].mean(0),label=arm);axes[1].plot(raw['costs'].mean(0),label=arm)
            axes[0].set(ylabel='Mean stock position',title=domain+' position/cost diagnostics');axes[1].set(xlabel='Trading day',ylabel='Mean transaction cost');axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(figdir/f'final-{domain}-positions-costs.png');plt.close(fig)
        lines.extend(['Empty event-aligned plots in a no-switch domain indicate no qualifying events, not zero risk. Mean loss, initial liability, costs, turnover, and centered terminal risks are retained separately in each metrics.json.'])
        for arm,scores in final['scores'].items():
            for domain,m in scores.items():
                texrows.append(f"Final & {arm.replace('_',' ')} & {domain} & {m['entropic']:.6f} & {m['cvar95']:.6f} & {m['daily_error_std']:.6f} "+chr(92)*2)
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    tex = [r"\documentclass{article}", r"\usepackage[margin=1cm]{geometry}", r"\begin{document}", r"\section*{Single-seed entropic benchmark gate}", "Accepted candidate: "+str(summary['accepted'])+". Uncertainty is conditional on trained policies.", r"\begin{tabular}{llllll}", "Candidate & Arm & Domain & Entropic & CVaR95 & Daily SD " + chr(92)*2]
    tex += texrows + [r"\end{tabular}", r"\end{document}"]
    (OUT/'report.tex').write_text(chr(10).join(tex))
    # Hash every scientific artifact only after files have been closed.
    manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and not p.name.endswith('.tmp') and p.name not in ['gpu.csv','artifact_hashes.json']}
    save(OUT/'artifact_hashes.json',manifest)

def main():
    assert all(d.platform=='gpu' for d in jax.devices()),'GPU required'
    save(OUT/'design.json',D)
    save(OUT/'numerics.json',NUMERICS)
    save(OUT/'residual_review_protocol.json',RESIDUAL_PROTOCOL)
    (OUT/'actual_source.py').write_bytes(pathlib.Path(__file__).read_bytes())
    save(OUT/'actual_source_sha256.json',dict(sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()))
    parser=argparse.ArgumentParser(); parser.add_argument('--mode',default='validate',choices=['validate','gate','campaign']); args=parser.parse_args()
    if args.mode in ('gate','campaign'):
        run_gate()
        if args.mode=='campaign' and json.loads((OUT/'gate_summary.json').read_text())['accepted']:
            run_final()
        exec(AUDIT_SOURCE,{'__name__':'independent_audit'})
        make_report()
        return
    reports=[]
    for family in D['families']:
        path=OUT/family['name']; path.mkdir(exist_ok=True)
        print('BUILD PRICER',family['name'],flush=True); pricer=Pricer(family)
        numerical=validate(family,pricer,path)
        independent=independent_prices(family,pricer,path)
        objectives=validate_objectives(family,path)
        reports.append(dict(family=family['name'],numerical=numerical,independent=independent,objectives=objectives,passed=numerical['passed'] and independent['passed'] and objectives['passed']))
    save(OUT/'validation_summary.json',dict(passed=all(x['passed'] for x in reports),families=reports))
if __name__=='__main__':
    try: main()
    except Exception:
        save(OUT/'failure.json',dict(traceback=traceback.format_exc())); raise
