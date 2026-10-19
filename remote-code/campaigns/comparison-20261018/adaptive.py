"""Domain-independent persistent-learner entropic curriculum primitives.

No sampling, IO or hidden target data occurs here. Callers own disjoint streams,
checkpointing and historical snapshots. All callbacks must be JAX traceable.
"""
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax.scipy.special import logsumexp


class Generator(NamedTuple):
    logits: object
    opt_state: object


def detached(tree):
    return jax.tree.map(
        lambda x: jax.lax.stop_gradient(x) if eqx.is_array(x) else x, tree
    )


def entropic(losses, gamma=10., weights=None):
    """Normalized weighted empirical entropic risk (positive weights).

    Differentiating this scalar, including its denominator, is essential for
    nonlinear DPG. At all-one weights it equals the ordinary learner objective.
    """
    if weights is None:
        return (logsumexp(gamma * losses) - jnp.log(losses.size)) / gamma
    return (
        logsumexp(gamma * losses + jnp.log(weights))
        - jnp.log(jnp.sum(weights))
    ) / gamma


def probabilities(logits, prior, epsilon=.05):
    return (1. - epsilon) * jax.nn.softmax(logits) + epsilon * prior


def init_generator(prior, optimizer):
    """prior must be strictly positive and normalized; baseline is independent."""
    logits = jnp.log(prior)
    return Generator(logits, optimizer.init(logits))


def weighted_block(student, batches, weights, losses_fn, optimizer, gamma=10.,
                   rematerialize=True):
    """K persistent clipped-Adam updates; weights shape [K,B].

    student=(Equinox model, actual Optax state). losses_fn(model,batch) must
    return B terminal losses. Optimizer can include clipping; nothing detached
    inside the block, so DPG differentiates moments and clipping as implemented.
    """
    model, opt_state = student
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def step(carry, inputs):
        p, state = carry
        batch, w = inputs

        def objective(candidate):
            return entropic(losses_fn(eqx.combine(candidate, static), batch), gamma, w)

        value, gradient = jax.value_and_grad(objective)(p)
        updates, state = optimizer.update(gradient, state, p)
        return (eqx.apply_updates(p, updates), state), value

    scan_step = jax.checkpoint(step) if rematerialize else step
    (params, opt_state), values = jax.lax.scan(
        scan_step, (params, opt_state), (batches, weights)
    )
    return (eqx.combine(params, static), opt_state), values


def make_target(losses_fn, hard, easy=None, scales=(1., 1.), gamma=10.):
    """Hard entropic, or balanced scaled easy/hard entropic objective.

    scales=(easy,hard) fixed positive constants frozen before adaptive training.
    Hard-only remains unscaled, as original entropic teacher implementation.
    """
    def objective(model):
        h = entropic(losses_fn(model, hard), gamma)
        if easy is None:
            return h
        e = entropic(losses_fn(model, easy), gamma)
        return .5 * (e / scales[0] + h / scales[1])
    return objective


def dpg_attributions(student, batches, block_fn, target_fn, weight_shape):
    """Return actual post-block state and raw negative weight derivatives.

    block_fn(student,batches,weights)->(post_student, learner_values).
    The incoming state is detached once; every update within K is differentiated.
    No differentiating the simulator, discrete regime IDs or carried past blocks.
    """
    student, batches = detached(student), detached(batches)

    def after(weights):
        post, values = block_fn(student, batches, weights)
        return target_fn(post[0]), (post, values)

    (target_after, (post, values)), derivative = eqx.filter_value_and_grad(
        after, has_aux=True
    )(jnp.ones(weight_shape))
    return detached(post), values, detached(-derivative), target_after


def soar_reward(pre_model, post_model, target_fn, baseline, decay=.9):
    before, after = target_fn(pre_model), target_fn(post_model)
    improvement = detached(before - after)
    advantage = detached(improvement - baseline)
    return advantage, decay * baseline + (1. - decay) * improvement, {
        'target_before': before, 'target_after': after,
        'improvement': improvement, 'ema_baseline': baseline,
    }


def find_adam_state(state):
    if hasattr(state, 'nu') and hasattr(state, 'count'):
        return state
    children = state.values() if isinstance(state, dict) else state
    if isinstance(state, (tuple, list, dict)):
        for child in children:
            found = find_adam_state(child)
            if found is not None:
                return found
    return None


def selfplay_rewards(model, historical, opt_state, batches, losses_fn,
                     learner_lr, outer_round, gamma=10., beta2=.999,
                     adam_epsilon=1e-8, adam_eps_root=0.):
    """Entropic-gradient intrinsic reward, one normalization per learner batch.

    historical must be pre-block snapshot floor(outer_round/2). Uses the actual
    learner Adam bias-corrected second moments. Historical direction is old-new.
    No target-domain batch or risk score is accepted by this function.
    """
    adam = find_adam_state(opt_state)
    if adam is None:
        raise ValueError('Selfplay requires actual learner Adam second moments')
    params, static = eqx.partition(model, eqx.is_inexact_array)
    historical_leaves = jax.tree.leaves(eqx.filter(historical, eqx.is_inexact_array))
    current_leaves, parameter_structure = jax.tree.flatten(params)
    if len(historical_leaves) != len(current_leaves) or any(
        h.shape != p.shape or h.dtype != p.dtype
        for h, p in zip(historical_leaves, current_leaves)
    ):
        raise ValueError('Historical snapshot parameter layout changed')
    # Separate pickle loads can reconstruct Equinox's private Missing sentinels
    # with different identities in static metadata. The historical numerical
    # parameters enter this direction on the frozen current architecture.
    historical_params = jax.tree.unflatten(parameter_structure, historical_leaves)
    correction = 1. - beta2 ** jnp.maximum(adam.count, 1)
    preconditioner = jax.tree.map(
        lambda v: learner_lr / (jnp.sqrt(v / correction + adam_eps_root) + adam_epsilon), adam.nu
    )
    direction = jax.tree.map(
        lambda h, p, v: (h - p) * v, historical_params, params, preconditioner
    )

    def one(batch):
        def values(p):
            return losses_fn(eqx.combine(p, static), batch)
        losses, directional = eqx.filter_jvp(values, (params,), (direction,))
        # B*softmax is the per-example contribution to the directional
        # derivative of empirical entropic risk, expressed on a mean scale.
        return jnp.abs(losses.size * jax.nn.softmax(gamma * losses) * directional)

    rewards = jax.vmap(one)(batches)
    rewards = jnp.where(jnp.asarray(outer_round) > 0, rewards, jnp.zeros_like(rewards))
    return detached(rewards), {
        'parameter_movement_norm': optax.global_norm(jax.tree.map(
            lambda h, p: h - p, historical_params, params
        )),
        'historical_index': jnp.asarray(outer_round) // 2,
    }


def generator_step(generator, optimizer, ids, rewards, prior, selfplay=False,
                   epsilon=.05, kl_coefficient=.01, expert_coefficient=1.,
                   normalization_epsilon=1e-8, apply_update=True):
    """Summed raw scores for DPG/SOAR; mean PG + normalized EI for selfplay."""
    ids, rewards = detached(ids), detached(rewards)

    def objective(logits):
        q = probabilities(logits, prior, epsilon)
        logp = jnp.log(q)[ids]
        if selfplay:
            advantage = (rewards - jnp.mean(rewards)) / (
                jnp.std(rewards) + normalization_epsilon
            )
            policy = -jnp.mean(advantage * logp)
            expert = -jnp.sum(rewards * logp) / (
                jnp.sum(rewards) + normalization_epsilon
            )
        else:
            policy = -jnp.sum(rewards * logp)
            expert = jnp.array(0.)
        prior_penalty = kl_coefficient * jnp.sum(q * (jnp.log(q) - jnp.log(prior)))
        return policy + expert_coefficient * expert + prior_penalty, (
            policy, expert, prior_penalty
        )

    (loss, (policy, expert, penalty)), grad = jax.value_and_grad(
        objective, has_aux=True
    )(generator.logits)
    updates, state = optimizer.update(grad, generator.opt_state, generator.logits)
    candidate = Generator(optax.apply_updates(generator.logits, updates), state)
    result = jax.lax.cond(jnp.asarray(apply_update), lambda: candidate, lambda: generator)
    q = probabilities(result.logits, prior, epsilon)
    return result, {
        'teacher_loss': loss, 'policy_loss': policy,
        'expert_iteration_loss': expert, 'prior_penalty': penalty,
        'teacher_gradient_norm': optax.global_norm(grad),
        'probabilities': q, 'kl': jnp.sum(q * (jnp.log(q) - jnp.log(prior))),
        'entropy': -jnp.sum(q * jnp.log(q)),
        'reward_mean': jnp.mean(rewards), 'reward_std': jnp.std(rewards),
        'reward_min': jnp.min(rewards), 'reward_max': jnp.max(rewards),
        'reward_nonfinite': jnp.sum(~jnp.isfinite(rewards)),
        'generator_update_applied': jnp.asarray(apply_update),
    }
