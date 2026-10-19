"""Exact finite categorical curriculum with a base-mixture exploration floor."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import optax


def base_probabilities(n=12, base_index=3):
    return jnp.full((n,), .30 / (n-1)).at[base_index].set(.70)


def easy_probabilities(n=12, easy_index=0):
    return jnp.full((n,), .20 / (n-1)).at[easy_index].set(.80)


def hard_probabilities(n=12, hard_index=11):
    return jnp.full((n,), .20 / (n-1)).at[hard_index].set(.80)


class GeneratorState(NamedTuple):
    logits: object
    opt_state: object


def init_generator(optimizer, q0=None):
    q0 = base_probabilities() if q0 is None else jnp.asarray(q0)
    logits = jnp.log(q0)
    return GeneratorState(logits, optimizer.init(logits))


def probabilities(logits, q0=None, epsilon=.05):
    q0 = base_probabilities() if q0 is None else jnp.asarray(q0)
    return (1-epsilon)*jax.nn.softmax(logits) + epsilon*q0


def sample_regimes(generator_state, key, batch_shape, q0=None, epsilon=.05):
    logits = generator_state.logits if isinstance(generator_state, GeneratorState) else generator_state
    logq = jnp.log(probabilities(logits, q0, epsilon))
    ids = jax.random.categorical(key, logq, shape=tuple(batch_shape))
    return ids, logq[ids]


def distribution_metrics(logits, q0=None, epsilon=.05):
    q0 = base_probabilities() if q0 is None else q0
    q = probabilities(logits, q0, epsilon)
    return {'probabilities': q, 'entropy': -jnp.sum(q*jnp.log(q)),
            'kl': jnp.sum(q*(jnp.log(q)-jnp.log(q0))), 'hard_mass': q[-1], 'easy_mass': q[0], 'base_mass': q[3]}


def score_loss(logits, ids, rewards, q0=None, epsilon=.05, kl_coefficient=.01, reduction='sum'):
    """All rewards/actions detached; default summed score, fixed across block sizes."""
    q0 = base_probabilities() if q0 is None else q0
    q = probabilities(logits, q0, epsilon)
    terms = jax.lax.stop_gradient(rewards)*jnp.log(q)[jax.lax.stop_gradient(ids)]
    if reduction not in ('sum', 'mean'):
        raise ValueError('reduction must be sum or mean')
    policy = -jnp.sum(terms) if reduction == 'sum' else -jnp.mean(terms)
    return policy + kl_coefficient*jnp.sum(q*(jnp.log(q)-jnp.log(q0)))


def update_generator(state, optimizer, ids, rewards, **kwargs):
    loss, grad = jax.value_and_grad(score_loss)(state.logits, ids, rewards, **kwargs)
    updates, opt_state = optimizer.update(grad, state.opt_state, state.logits)
    old_q = probabilities(state.logits, kwargs.get('q0'), kwargs.get('epsilon', .05))
    prior_penalty = kwargs.get('kl_coefficient', .01)*distribution_metrics(state.logits, kwargs.get('q0'), kwargs.get('epsilon', .05))['kl']
    state = GeneratorState(optax.apply_updates(state.logits, updates), opt_state)
    metrics = distribution_metrics(state.logits, kwargs.get('q0'), kwargs.get('epsilon', .05))
    metrics.update(teacher_loss=loss, policy_loss=loss-prior_penalty, prior_penalty=prior_penalty,
                   distribution_movement_l1=jnp.sum(jnp.abs(metrics['probabilities']-old_q)),
                   teacher_grad_norm=optax.global_norm(grad))
    return state, metrics
