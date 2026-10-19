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
