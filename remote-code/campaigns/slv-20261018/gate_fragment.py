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
