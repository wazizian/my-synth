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
