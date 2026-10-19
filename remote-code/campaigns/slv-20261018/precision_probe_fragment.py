def precision_probe():
    family=D['families'][0];sim=make_sim(family);original=Hedger()
    spots=sim(key(10,800),probabilities(family,'base'),64)['spot'].reshape(2,32,C+H+1)
    hard=sim(key(10,801),probabilities(family,'hard'),32)['spot'];easy=sim(key(10,802),probabilities(family,'easy'),32)['spot'];scales=jnp.array([.01,.01])
    results={}
    for dtype in [jnp.float32,jnp.float64]:
        name=str(dtype);model=jax.tree.map(lambda x:x.astype(dtype) if eqx.is_inexact_array(x) else x,original);state=OPT.init(eqx.filter(model,eqx.is_inexact_array))
        post,rewards,value=sync(entropic_attribution(model,state,spots,hard,easy,scales,True));ad=float(-rewards[0,0])
        @eqx.filter_jit
        def after(weights):
            m,_,_=weighted_block(model,state,spots,weights);return target_risk(m,hard,easy,scales,True)
        direction=jnp.zeros((2,32)).at[0,0].set(1.);rows=[]
        for delta in [.05,.01,.003,.001,.0003,.0001,.00003]:
            fp=float(after(jnp.ones((2,32))+delta*direction));fm=float(after(jnp.ones((2,32))-delta*direction));fd=(fp-fm)/(2*delta)
            rows.append(dict(step=delta,finite_difference=fd,absolute_error=abs(fd-ad),target_plus=fp,target_minus=fm))
        results[name]=dict(autodiff=ad,target=float(value),finite_difference_sweep=rows,scale_invariance_error=float(jnp.max(jnp.abs(rewards.sum(1)))))
        save(OUT/'precision_probe.json',results);print('PRECISION',name,json.dumps(results[name]),flush=True)
