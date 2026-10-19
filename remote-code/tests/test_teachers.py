import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
import pytest
from src.generator import base_probabilities, probabilities, sample_regimes, init_generator, score_loss
from src.teachers import dpg_attributions, soar_reward, selfplay_rewards, selfplay_step, find_adam_state, make_target_objective


@pytest.fixture(autouse=True, scope='module')
def gpu_only():
    assert jax.default_backend() == 'gpu', 'Teacher numerical checks require a rented GPU worker'


def test_probability_floor_score_finite_difference():
    q0 = base_probabilities()
    logits = jnp.linspace(-3., 2., 12)
    q = probabilities(logits)
    np.testing.assert_allclose(q.sum(), 1., atol=1e-6)
    assert jnp.all(q >= .05*q0)
    f = lambda a:jnp.log(probabilities(a)[3])
    grad = jax.grad(f)(logits)
    h = .002
    for i in [0, 3, 11]:
        e = jnp.eye(12)[i]*h
        np.testing.assert_allclose(grad[i], (f(logits+e)-f(logits-e))/(2*h), rtol=.02, atol=2e-5)
    ids, _ = sample_regimes(logits, jax.random.key(1), (4, 256))
    assert ids.shape == (4, 256)
    assert len(np.unique(ids[0])) > 1


def test_dpg_carried_adam_multistep_finite_difference():
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(.03, eps_root=1e-10))
    theta = jnp.array([.7, -.4])
    state = optimizer.init(theta)
    updates, state = optimizer.update(jnp.array([.3, -.6]), state, theta)
    theta = optax.apply_updates(theta, updates)
    x = jnp.array([[[1.,2.],[-1.,.2],[.4,-.8]], [[.2,1.],[2.,-.3],[-.4,.8]]])
    y = jnp.array([[.2,-.1,.7],[.5,.2,-.4]])
    batch = (x,y)
    def block(start, data, weights):
        def step(carry, item):
            p,s = carry
            xx,yy,w = item
            def loss(pp):
                return jnp.sum(w*(xx@pp-yy)**2)/jnp.sum(w)
            lossval,grad = jax.value_and_grad(loss)(p)
            u,s = optimizer.update(grad,s,p)
            return (optax.apply_updates(p,u),s),lossval
        return jax.lax.scan(step,start,(data[0],data[1],weights))
    target = lambda p:jnp.sum((p-jnp.array([.1,.2]))**2)
    post,_,r,_ = dpg_attributions((theta,state),batch,block,target,(2,3))
    expected,_ = block((theta,state),batch,jnp.ones((2,3)))
    # Check all carried moments and count as well as persistent parameters.
    for actual, wanted in zip(jax.tree.leaves(post), jax.tree.leaves(expected)):
        np.testing.assert_allclose(actual,wanted,atol=1e-7)
    for i,j in [(0,0),(0,2),(1,1)]:
        d = jnp.zeros((2,3)).at[i,j].set(.003)
        fp = target(block((theta,state),batch,jnp.ones((2,3))+d)[0][0])
        fm = target(block((theta,state),batch,jnp.ones((2,3))-d)[0][0])
        np.testing.assert_allclose(r[i,j],-(fp-fm)/.006,rtol=.03,atol=2e-5)
    np.testing.assert_allclose(r.sum(axis=1),0,atol=1e-7)
    assert jnp.max(jnp.abs(r)) > 1e-5


def test_soar_zero_update_sign_previous_baseline():
    loss = lambda p, data:(data@p)**2
    p = jnp.array([1.,2.]); data=jnp.eye(2)
    a,b,m = soar_reward(p,p,data,loss,jnp.array(0.))
    assert m['raw_improvement'] == 0
    a,b,m = soar_reward(p,p*.5,data,loss,jnp.array(.2))
    assert m['raw_improvement'] > 0
    np.testing.assert_allclose(a,m['raw_improvement']-.2)
    np.testing.assert_allclose(b,.9*.2+.1*m['raw_improvement'])


def test_selfplay_jvp_equals_explicit_gradient_dot():
    p=jnp.array([.4,-.2]); past=jnp.array([.3,.1])
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(.01))
    state=optimizer.init(p)
    _,state=optimizer.update(jnp.array([.2,-.3]),state,p)
    x=jnp.array([[1.,2.],[-.4,.3],[.8,-.1]])
    loss=lambda pp,xx:(xx@pp)**2
    r,m=selfplay_rewards(p,past,state,x,loss,.01,3)
    adam=find_adam_state(state)
    precond=.01/(jnp.sqrt(adam.nu/(1-.999**adam.count))+1e-8)
    explicit=jnp.abs(jax.jacrev(lambda pp:loss(pp,x))(p)@((past-p)*precond))
    np.testing.assert_allclose(r,explicit,rtol=2e-5,atol=1e-6)
    assert m['historical_index']==1
    r,_=selfplay_rewards(p,past,state,x,loss,.01,0)
    np.testing.assert_array_equal(r,jnp.zeros(3))


def test_selfplay_static_leaves_and_zero_movement_preserve_optimizer():
    class TinyModel(eqx.Module):
        weight: object
        activation: object
        integer: object
        def __call__(self, x):
            return self.activation(x @ self.weight)
    p = TinyModel(jnp.array([.4,-.2]), jnp.tanh, jnp.array(2))
    past = eqx.tree_at(lambda m:m.weight,p,jnp.array([.3,.1]))
    optimizer = optax.adam(.01)
    params = eqx.filter(p,eqx.is_inexact_array)
    state = optimizer.init(params)
    gradients = eqx.tree_at(lambda m:m.weight,params,jnp.array([.2,-.3]))
    _, state = optimizer.update(gradients,state,params)
    x = jnp.array([[1.,2.],[-.4,.3],[.8,-.1]])
    losses = lambda model,data:model(data)**2
    rewards,_ = selfplay_rewards(p,past,state,x,losses,.01,2)
    adam = find_adam_state(state)
    direction = (past.weight-p.weight)*.01/(jnp.sqrt(adam.nu.weight/(1-.999**adam.count))+1e-8)
    jac = jax.jacrev(lambda w:losses(eqx.tree_at(lambda m:m.weight,p,w),x))(p.weight)
    np.testing.assert_allclose(rewards,jnp.abs(jac@direction),rtol=2e-5,atol=1e-6)
    generator_optimizer = optax.adam(.01)
    generator = init_generator(generator_optimizer)
    # Populate generator momentum: a zero-reward skip must not decay or apply it.
    updates, gs = generator_optimizer.update(jnp.arange(12.),generator.opt_state,generator.logits)
    generator = generator._replace(logits=optax.apply_updates(generator.logits,updates),opt_state=gs)
    after,metrics = selfplay_step(generator,generator_optimizer,p,p,state,x,losses,
                                 jnp.array([0,3,11]),.01,2)
    for actual,wanted in zip(jax.tree.leaves(after),jax.tree.leaves(generator)):
        np.testing.assert_array_equal(actual,wanted)
    assert not metrics['generator_update_applied']


def test_mixed_objective_exact_and_hard_isolation():
    loss = lambda p,data:(data-p)**2
    easy,hard = jnp.array([.2,.4]),jnp.array([1.,2.])
    scales = {'easy':.1,'hard':2.}
    objective, components = make_target_objective('mixed',loss,hard,scales,easy)
    p = jnp.array(.3)
    c = components(p)
    np.testing.assert_allclose(objective(p),.5*(c['easy_raw']/.1+c['hard_raw']/2))
    expected_grad = .5*(jax.grad(lambda t:jnp.mean(loss(t,easy)))(p)/.1+
                        jax.grad(lambda t:jnp.mean(loss(t,hard)))(p)/2)
    np.testing.assert_allclose(jax.grad(objective)(p),expected_grad)
    with pytest.raises(ValueError,match='may not receive'):
        make_target_objective('hard',loss,hard,scales,easy)


def test_score_sum_and_detached_rewards():
    logits = jnp.log(base_probabilities())
    ids = jnp.array([0,3,11])
    rewards = jnp.array([-.2,.1,.7])
    expected = -jnp.sum(rewards*jnp.log(probabilities(logits))[ids])
    np.testing.assert_allclose(score_loss(logits,ids,rewards,kl_coefficient=0),expected)
    np.testing.assert_array_equal(jax.grad(lambda r:score_loss(logits,ids,r))(rewards),jnp.zeros(3))
