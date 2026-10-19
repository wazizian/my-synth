"""GPU-only bounded benchmark gate; standalone source packaged with frozen design."""
import os, json, time, hashlib, pathlib, pickle
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import equinox as eqx
import optax

DESIGN = json.loads('{\n  "campaign_id": "entropic-20261018",\n  "authorization": "Current user request to execute revised plans using GPUs as needed",\n  "total_budget_usd": 25,\n  "gate_budget_usd": 5,\n  "maximum_wall_hours": 8,\n  "training_seed": 20261017,\n  "gamma": 10.0,\n  "alpha": 0.95,\n  "candidates": [\n    {"name":"trinomial04", "log_step":0.04, "easy_q":0.04, "base_q":0.30, "hard_q":0.90},\n    {"name":"trinomial06", "log_step":0.06, "easy_q":0.02, "base_q":0.20, "hard_q":0.80}\n  ],\n  "common": {"context":32, "horizon":20, "stock_cost":0.001, "hidden_size":32, "batch_size":256, "learning_rate":0.001, "block_updates":8, "gate_active_seconds":180, "checkpoint_seconds":[0,60,120,180]},\n  "scenario_menu": "easy stable, base stable, seven persistent switches after context at trading intervals 6 through 12",\n  "base_probabilities": [0.01,0.92,0.01,0.01,0.01,0.01,0.01,0.01,0.01],\n  "easy_probabilities": [1,0,0,0,0,0,0,0,0],\n  "hard_probabilities": [0,0,0.14285714285714285,0.14285714285714285,0.14285714285714285,0.14285714285714285,0.14285714285714285,0.14285714285714285,0.14285714285714285],\n  "minimum_improvement": {"easy":0.0002,"hard":0.0005},\n  "threshold_calibration": "Before new outcomes: approximately 3% and 4% of old pilot baseline terminal SD 0.0064463 and 0.0116544. Absolute normalized risk units, no percentage division by risk.",\n  "evaluation": {"selection_episodes":2048,"screen_episodes":8192,"confirmation_episodes":8192,"paired_bootstrap_replicates":400},\n  "pricing": "Zero interest, common P0 risk-neutral trinomial mixture. At each step up probability q/(1+exp(a)), down probability q*exp(a)/(1+exp(a)), stay 1-q. Bayesian scenario posterior uses only realized increments. Exact backward recursion per scenario, posterior mixture for liability marks. Target domains are conditional scenario populations within P0, not separately priced economies.",\n  "information": "Causal spot returns, inception-relative moneyness, time to maturity, liability mark and previous stock position. No scenario/switch or future information.",\n  "gate": "Validate numerical/accounting/causality/resume invariants before diagnostic training. Identical initialization and active-time/checkpoint opportunities for baseline/easy_reference/hard_reference. Require paired 95% interval lower endpoint above absolute improvement thresholds on both domains, specialist easy risk/CVaR/daily SD below hard, then repeat on fresh confirmation. Stop after at most two failures. Adaptive runs prohibited until accepted freeze.",\n  "namespaces": {"initialization":1,"training":2,"selection":3,"screen":4,"confirmation":5,"feedback":6,"monitor":7,"final":8,"bootstrap":9},\n  "pilot_preservation": "Existing files remain unchanged. Prior user-authorized cleanup deleted pilot datasets/checkpoints; see results/cleanup_manifest.json. No claim that those deleted artifacts can be recovered."\n}\n')
OUT = pathlib.Path(os.environ['JOB_OUTPUT_DIR'])
OUT.mkdir(parents=True, exist_ok=True)
C, H, B, K = 32, 20, 256, 8
GAMMA = DESIGN['gamma']
P0 = jnp.array(DESIGN['base_probabilities'])
PROBS = {'base': P0, 'easy':jnp.array(DESIGN['easy_probabilities']), 'hard':jnp.array(DESIGN['hard_probabilities'])}

def save_json(path, data):
    path = pathlib.Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+'.tmp'); tmp.write_text(json.dumps(data,indent=2,allow_nan=False)); tmp.replace(path)

def key(namespace, index=0):
    return jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(DESIGN['training_seed']), namespace),index)

def risk(x):
    return (jax.scipy.special.logsumexp(GAMMA*x)-jnp.log(x.size))/GAMMA

class Hedger(eqx.Module):
    cell: eqx.nn.GRUCell
    head: eqx.nn.Linear
    def __init__(self):
        a,b = jax.random.split(key(1)); self.cell=eqx.nn.GRUCell(6,32,key=a,dtype=jnp.float32); self.head=eqx.nn.Linear(32,1,key=b,dtype=jnp.float32)


def setup(candidate):
    a=candidate['log_step']; qs=jnp.full((9,C+H),candidate['base_q'])
    qs=qs.at[0,:].set(candidate['easy_q'])
    for i,t in enumerate(range(6,13),2): qs=qs.at[i,C+t:].set(candidate['hard_q'])
    up=qs/(1+jnp.exp(a)); down=up*jnp.exp(a)
    emission=jnp.stack([down,1-qs,up],-1)
    # Larger lattice supports all actually reachable nodes without boundary use.
    grid=jnp.arange(-2*H,2*H+1); terminal=jnp.maximum(jnp.exp(a*grid)-1,0)
    vals=[None]*(H+1); vals[H]=jnp.broadcast_to(terminal,(9,grid.size))
    for t in reversed(range(H)):
        v=vals[t+1]; vals[t]=emission[:,C+t,0,None]*jnp.roll(v,1,axis=1)+emission[:,C+t,1,None]*v+emission[:,C+t,2,None]*jnp.roll(v,-1,axis=1)
    return emission,jnp.stack(vals,axis=1)


def make_sim(candidate):
    emission,table=setup(candidate); a=candidate['log_step']
    @eqx.filter_jit
    def simulate(k,prob,n):
        ks,ke=jax.random.split(k); scenario=jax.random.choice(ks,9,(n,),p=prob)
        u=jax.random.uniform(ke,(n,C+H)); pe=emission[scenario]
        move=(u>pe[:,:,0]).astype(jnp.int32)+(u>(pe[:,:,0]+pe[:,:,1])).astype(jnp.int32)-1
        lattice=jnp.concatenate([jnp.zeros((n,1),jnp.int32),jnp.cumsum(move,axis=1)],axis=1)
        spot=jnp.exp(a*lattice); relative=lattice[:,C:]-lattice[:,C,None]
        def step(p,inputs):
            t,m=inputs; likelihood=jnp.take(emission[:,t,:],m+1,axis=1).T
            p=p*likelihood; p=p/p.sum(-1,keepdims=True); return p,p
        prior=jnp.broadcast_to(P0,(n,9))
        _,posterior=jax.lax.scan(step,prior,(jnp.arange(C+H),move.T))
        posterior=jnp.concatenate([prior[:,None,:],posterior.transpose(1,0,2)],axis=1)
        def mark(t):
            v=jnp.take(table[:,t,:],relative[:,t]+2*H,axis=1).T
            return (posterior[:,C+t,:]*v).sum(-1)
        marks=jax.vmap(mark)(jnp.arange(H+1)).T
        return dict(spot=spot, marks=marks, scenario=scenario, switch=jnp.where(scenario>=2,scenario+4,-1))
    return simulate


def rollout(model,batch):
    def one(s,marks):
        inc=s[C]; normalized=s[C:]/inc
        def step(carry,t):
            hidden,prev=carry; active=t>=C; r=jnp.log(s[t]/s[jnp.maximum(t-1,0)])
            x=jnp.array([r*10, jnp.where(active,jnp.log(s[t]/inc),0),jnp.where(active,(C+H-t)/H,0),jnp.where(active,marks[jnp.maximum(t-C,0)],0),prev,active.astype(s.dtype)],dtype=jnp.float32)
            hidden=model.cell(x,hidden); pos=jnp.where(active,2*jnp.tanh(model.head(hidden)[0]),0)
            return (hidden,pos),pos
        _,positions=jax.lax.scan(step,(jnp.zeros(32,jnp.float32),jnp.float32(0)),jnp.arange(C+H))
        h=positions[C:]; changes=jnp.diff(jnp.concatenate([jnp.zeros(1),h]))
        costs=.001*jnp.abs(changes)*normalized[:-1]
        costs=costs.at[-1].add(.001*jnp.abs(h[-1])*normalized[-1])
        gains=h*jnp.diff(normalized); daily=jnp.diff(marks)-gains+costs
        loss=jnp.maximum(normalized[-1]-1,0)-gains.sum()+costs.sum()
        turnover=jnp.abs(changes).sum()+jnp.abs(h[-1])
        return dict(loss=loss,daily=daily,costs=costs,positions=h,turnover=turnover,v0=marks[0])
    return jax.vmap(one)(batch['spot'],batch['marks'])

EVALUATE=eqx.filter_jit(rollout)
OPT=optax.chain(optax.clip_by_global_norm(1.),optax.adam(.001,eps_root=1e-12))

@eqx.filter_jit
def block(model,state,batches):
    p,static=eqx.partition(model,eqx.is_inexact_array)
    def step(carry,b):
        p,state=carry
        loss,grad=jax.value_and_grad(lambda p:risk(rollout(eqx.combine(p,static),b)['loss']))(p)
        update,state=OPT.update(grad,state,p)
        return (eqx.apply_updates(p,update),state),(loss,optax.global_norm(grad))
    (p,state),metrics=jax.lax.scan(step,(p,state),batches)
    return eqx.combine(p,static),state,metrics


def metrics(raw):
    x=jnp.asarray(raw['loss']); n=x.size; sorted_x=jnp.sort(x)[::-1]; mass=.05*n; count=int(mass)
    cvar=(sorted_x[:count].sum()+(mass-count)*sorted_x[count])/mass
    w=jax.nn.softmax(GAMMA*x)
    return dict(entropic=float(risk(x)),cvar95=float(cvar),var95=float(jnp.quantile(x,.95)),mean=float(x.mean()),daily_error_std=float(jnp.sqrt(jnp.var(jnp.asarray(raw['daily']),axis=0,ddof=1).mean())),daily_mean=np.asarray(raw['daily']).mean(0).tolist(),daily_sd=np.asarray(raw['daily']).std(0,ddof=1).tolist(),sample_count=n,finite_count=int(jnp.isfinite(x).sum()),exponential_ess=float(1/jnp.sum(w*w)),max_exponential_weight=float(w.max()))


def evaluate(model,simulate,namespace,n,path):
    result={}; path.mkdir(parents=True,exist_ok=True)
    for d,prob in PROBS.items():
        chunks=[]; switches=[]
        for i in range(n//512):
            batch=simulate(key(namespace,i),prob,512); raw=EVALUATE(model,batch)
            raw={**raw,'spot':batch['spot'],'marks':batch['marks']}
            chunks.append(jax.device_get(raw)); switches.append(np.asarray(batch['switch']))
        raw={k:np.concatenate([x[k] for x in chunks]) for k in chunks[0]}; raw['switch']=np.concatenate(switches)
        np.savez_compressed(path/(d+'.npz'),**raw)
        result[d]=metrics(raw)
        result[d]['dataset_sha256']=hashlib.sha256(raw['spot'].tobytes()+raw['marks'].tobytes()+raw['switch'].tobytes()).hexdigest()
    save_json(path/'metrics.json',result); return result


def validate(simulate,path,candidate):
    emission,table=setup(candidate)
    a=candidate['log_step']
    martingale=float(jnp.max(jnp.abs(emission[:,:,0]*jnp.exp(-a)+emission[:,:,1]+emission[:,:,2]*jnp.exp(a)-1)))
    assert martingale<1e-12
    assert bool(jnp.all(P0>0)) and abs(float(P0.sum())-1)<1e-12
    # Exact conditional valuation recursion on every reachable node.
    pricing_error=0.
    for t in range(H):
        nxt=table[:,t+1,:]
        tower=emission[:,C+t,0,None]*jnp.roll(nxt,1,axis=1)+emission[:,C+t,1,None]*nxt+emission[:,C+t,2,None]*jnp.roll(nxt,-1,axis=1)
        pricing_error=max(pricing_error,float(jnp.max(jnp.abs(table[:,t,H:3*H+1]-tower[:,H:3*H+1]))))
    assert pricing_error<1e-12
    assert candidate['easy_q']<candidate['base_q']<candidate['hard_q']
    save_json(path/'distribution_validation.json',dict(passed=True,martingale_error=martingale,pricing_recursion_error=pricing_error,base_full_support=True,hidden_switch_times=list(range(6,13)),remaining_postswitch_intervals=[H-t for t in range(6,13)],easy_q=candidate['easy_q'],base_q=candidate['base_q'],hard_q=candidate['hard_q'],specialization_status='Requires diagnostic training and held-out gate; not yet established'))
    model=Hedger(); state=OPT.init(eqx.filter(model,eqx.is_inexact_array))
    batch=simulate(key(4,99999),P0,256); raw=EVALUATE(model,batch)
    error=float(jnp.max(jnp.abs(raw['daily'].sum(-1)-raw['loss']+raw['v0'])))
    assert error<1e-7,error
    assert float(jnp.max(jnp.abs(batch['marks'][:,-1]-jnp.maximum(batch['spot'][:,-1]/batch['spot'][:,C]-1,0))))<1e-10
    x=jnp.array([0.,1.,2.,3.]); translation=float(jnp.abs(risk(x-.13)-(risk(x)-.13))); assert translation<1e-12
    assert abs(metrics(dict(loss=np.arange(4.),daily=np.zeros((4,H))))['cvar95']-3)<1e-12
    # Deliberately change only future prices/marks; earlier actions must agree.
    future={**batch,'spot':batch['spot'].at[:,C+10:].multiply(1.1),'marks':batch['marks'].at[:,10:].add(.1)}
    assert float(jnp.max(jnp.abs(EVALUATE(model,future)['positions'][:,:10]-raw['positions'][:,:10])))==0
    batches=jax.tree.map(lambda x:jnp.broadcast_to(x,(K,)+x.shape),batch)
    m,s,logs=block(model,state,batches); jax.block_until_ready(logs)
    assert bool(jnp.all(jnp.isfinite(logs[0]))) and bool(jnp.all(jnp.isfinite(logs[1])))
    payload=jax.device_get((m,s,key(2),0.)); temporary=path/'validation.pkl.tmp'; temporary.write_bytes(pickle.dumps(payload)); temporary.replace(path/'validation.pkl')
    restored=pickle.loads((path/'validation.pkl').read_bytes()); a=block(m,s,batches); b=block(restored[0],restored[1],batches)
    assert all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)))
    save_json(path/'validation.json',dict(passed=True,telescoping_max_error=error,telescoping_tolerance=1e-7,cash_translation_error=translation,causality=True,resume_exact=True,finite_gradients=True,device=str(jax.devices())))


def train(arm,domain,simulate,path):
    path.mkdir(parents=True,exist_ok=True); model=Hedger(); save_json(path/'initialization.json',dict(sha256=hashlib.sha256(b''.join(np.asarray(x).tobytes() for x in jax.tree.leaves(model))).hexdigest(),seed=DESIGN['training_seed'],architecture='GRU(6,32), linear(32,1), tanh bound 2')); state=OPT.init(eqx.filter(model,eqx.is_inexact_array)); rng=key(2)
    # Compile and synchronize before the equal active-time budget.
    def draw(rng):
        rng,sub=jax.random.split(rng); b=simulate(sub,PROBS[domain],B*K)
        return rng,jax.tree.map(lambda x:x.reshape((K,B)+x.shape[1:]),b)
    _,batches=draw(rng); compiled=block(model,state,batches); jax.block_until_ready(compiled)
    active=0.; steps=0; best=float('inf'); selected=None; start=time.monotonic()
    for target in DESIGN['common']['checkpoint_seconds']:
        with (path/'training.jsonl').open('a') as f:
            while active<target:
                t=time.monotonic(); rng,batches=draw(rng); model,state,logs=block(model,state,batches); jax.block_until_ready(logs); active+=time.monotonic()-t; steps+=K
                assert bool(jnp.all(jnp.isfinite(logs[0]))) and bool(jnp.all(jnp.isfinite(logs[1])))
                f.write(json.dumps(dict(active_seconds=active,steps=steps,loss=np.asarray(logs[0]).tolist(),gradient_norm=np.asarray(logs[1]).tolist()))+'\n'); f.flush()
        cp=path/f'checkpoint-{target}.pkl'; tmp=cp.with_suffix('.pkl.tmp'); tmp.write_bytes(pickle.dumps(jax.device_get((model,state,rng,active,steps)))); tmp.replace(cp)
        scores=evaluate(model,simulate,3,2048,path/f'selection-{target}'); score=scores[domain]['entropic']
        if score<best: best=score; selected=cp
    save_json(path/'selection.json',dict(checkpoint=str(selected),criterion_domain=domain,score=best,active_seconds=active,steps=steps,simulated_training_paths=steps*B,wall_seconds=time.monotonic()-start))
    return pickle.loads(selected.read_bytes())[0]


def assess(models,simulate,namespace,path):
    scores={arm:evaluate(model,simulate,namespace,8192,path/arm) for arm,model in models.items()}
    checks={}; intervals={}
    for d,arm in [('easy','easy_reference'),('hard','hard_reference')]:
        a=jnp.array(np.load(path/'baseline'/(d+'.npz'))['loss']); b=jnp.array(np.load(path/arm/(d+'.npz'))['loss'])
        # Whole episode paired resampling; trained policies held fixed.
        @jax.jit
        def bootstrap(k):
            indices=jax.random.randint(k,(8192,),0,8192); return risk(a[indices])-risk(b[indices])
        vals=jax.vmap(bootstrap)(jax.random.split(key(9,namespace),400)); low,high=np.asarray(jnp.quantile(vals,jnp.array([.025,.975])))
        intervals[d]=dict(improvement=float(risk(a)-risk(b)),low=float(low),high=float(high),minimum=DESIGN['minimum_improvement'][d])
        checks[d+'_specialization']=float(low)>DESIGN['minimum_improvement'][d]
    for metric in ['entropic','cvar95','daily_error_std']:
        checks['difficulty_'+metric]=scores['easy_reference']['easy'][metric]<scores['hard_reference']['hard'][metric]
    report=dict(passed=all(checks.values()),checks=checks,paired_intervals=intervals,scores=scores,uncertainty_scope='Conditional on one shared training seed; episode resampling only')
    save_json(path/'gate.json',report); return report


def main():
    assert all(d.platform=='gpu' for d in jax.devices()),'GPU required'
    save_json(OUT/'design.json',DESIGN)
    results=[]
    for candidate in DESIGN['candidates']:
        path=OUT/candidate['name']; path.mkdir(exist_ok=True); simulate=make_sim(candidate)
        validate(simulate,path,candidate)
        print('VALIDATION PASSED',candidate['name'],flush=True)
        models={}
        for arm,domain in [('baseline','base'),('easy_reference','easy'),('hard_reference','hard')]:
            print('TRAIN',candidate['name'],arm,flush=True); models[arm]=train(arm,domain,simulate,path/arm)
        screen=assess(models,simulate,4,path/'screen')
        result=dict(candidate=candidate['name'],screen_passed=screen['passed'])
        if screen['passed']:
            confirmation=assess(models,simulate,5,path/'confirmation'); result['confirmation_passed']=confirmation['passed']
            if confirmation['passed']:
                results.append(result); save_json(OUT/'gate_summary.json',dict(accepted=candidate['name'],results=results)); return
        results.append(result); save_json(OUT/'gate_summary.json',dict(accepted=None,results=results))
    print('No candidate passed; adaptive training prohibited.',flush=True)

def make_report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((OUT/'gate_summary.json').read_text())
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    lines=['# Entropic benchmark development gate','',
           'Single shared training seed 20261017. Paired bootstrap intervals are conditional on trained policies; they do not measure training-seed uncertainty.', '',
           'Accepted candidate: '+str(summary['accepted']), '',
           'Two predeclared trinomial candidates, common causal Bayesian liability pricing, stock-only trading, gamma=10, CVaR alpha=0.95. Positive pre-premium normalized losses.', '',
           'Training uses a finite-minibatch log-mean-exp estimator (batch 256), which is not exact population entropic risk. Selection and screening use separate random namespaces. Exponential concentration diagnostics are in metrics.json.', '',
           'Policy features contain only causal returns, moneyness, time, liability mark, previous position and trading indicator. Initial trading costs enter interval zero; liquidation enters the last interval. Exact mathematical pricing is evaluated numerically; GPU validation records observed accounting errors.', '']
    texrows=[]
    for candidate in summary['results']:
        name=candidate['candidate']; root=OUT/name
        lines.extend(['## '+name,''])
        for phase in ['screen','confirmation']:
            p=root/phase/'gate.json'
            if p.exists():
                result=json.loads(p.read_text());lines.extend([phase+': '+('PASS' if result['passed'] else 'FAIL'), '',json.dumps(result['checks']), '',json.dumps(result['paired_intervals']), ''])
        for domain in PROBS:
            for metric in ['entropic','cvar95','daily_error_std']:
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
    else:
        lines.extend(['## Remaining work','','Benchmark confirmation passed. Freeze accepted configuration and datasets, implement/audit entropic teachers, and run all eight arms before claiming campaign completion.'])
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    tex = [r"\documentclass{article}", r"\usepackage[margin=1cm]{geometry}", r"\begin{document}", r"\section*{Single-seed entropic benchmark gate}", "Accepted candidate: "+str(summary['accepted'])+". Uncertainty is conditional on trained policies.", r"\begin{tabular}{llllll}", "Candidate & Arm & Domain & Entropic & CVaR95 & Daily SD " + chr(92)*2]
    tex += texrows + [r"\end{tabular}", r"\end{document}"]
    (OUT/'report.tex').write_text(chr(10).join(tex))
    # Hash every scientific artifact only after files have been closed.
    manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and not p.name.endswith('.tmp') and p.name not in ['gpu.csv','artifact_hashes.json']}
    save_json(OUT/'artifact_hashes.json',manifest)

if __name__=='__main__':
    main()
    make_report()

