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
