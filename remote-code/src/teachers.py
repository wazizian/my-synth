"""Persistent-learner teacher mechanisms; callbacks make optimizer fidelity explicit.

DPG/SOAR use summed scores (not mean). DPG uses raw unscaled attribution.
Self-play uses mean reward-normalized expert iteration, no target-data argument.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from .generator import update_generator, GeneratorState, probabilities, distribution_metrics


def detached(tree):
    return jax.tree.map(lambda x: jax.lax.stop_gradient(x) if eqx.is_array(x) else x, tree)


def reward_metrics(rewards):
    return {'reward_mean': jnp.mean(rewards), 'reward_std': jnp.std(rewards),
            'reward_min': jnp.min(rewards), 'reward_max': jnp.max(rewards),
            'reward_nonfinite': jnp.sum(~jnp.isfinite(rewards)), 'reward_clipping_fraction': jnp.array(0.)}


def dpg_attributions(student_pre, batch, block_fn, target_loss_fn, weight_shape):
    """block_fn(state,batch,weights)->(state,losses), including carried Optax state."""
    start = detached(student_pre)
    batch = detached(batch)
    def weighted_target(weights):
        post, losses = block_fn(start, batch, weights)
        return target_loss_fn(post[0]), (post, losses)
    (target, (post, losses)), derivative = eqx.filter_value_and_grad(weighted_target, has_aux=True)(jnp.ones(weight_shape))
    return detached(post), losses, detached(-derivative), target


def dpg_step(generator, generator_optimizer, student_pre, batch, ids, block_fn, target_loss_fn, target_components_fn=None, **generator_kwargs):
    post, losses, rewards, after = dpg_attributions(student_pre, batch, block_fn, target_loss_fn, ids.shape)
    generator, metrics = update_generator(generator, generator_optimizer, ids, rewards, **generator_kwargs)
    metrics.update(reward_metrics(rewards))
    metrics.update(attribution_norm=optax.global_norm(rewards), attribution_sum=jnp.sum(rewards),
                   attribution_positive_fraction=jnp.mean(rewards>0),
                   attribution_negative_fraction=jnp.mean(rewards<0),
                   attribution_scaling=jnp.array(1.), target_after=after,
                   target_before=target_loss_fn(student_pre[0]))
    if target_components_fn is not None:
        metrics.update(component_metrics(student_pre[0], post[0], target_components_fn))
    metrics['component_attributions_available'] = jnp.array(False)
    return generator, post, losses, metrics


def soar_reward(pre_model, post_model, feedback, loss_fn, baseline, ema_decay=.9):
    before = jnp.mean(loss_fn(pre_model, feedback))
    after = jnp.mean(loss_fn(post_model, feedback))
    improvement = detached(before-after)
    advantage = detached(improvement-baseline)
    new_baseline = ema_decay*baseline+(1-ema_decay)*improvement
    return advantage, new_baseline, {'target_before':before, 'target_after':after,
        'raw_improvement':improvement, 'ema_baseline':baseline, 'advantage':advantage}


def soar_step(generator, generator_optimizer, student_pre, student_post, feedback, loss_fn, ids, baseline, ema_decay=.9, **generator_kwargs):
    advantage, baseline, metrics = soar_reward(student_pre[0], student_post[0], feedback, loss_fn, baseline, ema_decay)
    rewards = jnp.broadcast_to(advantage, ids.shape)
    generator, gm = update_generator(generator, generator_optimizer, ids, rewards, **generator_kwargs)
    metrics.update(gm)
    metrics.update(reward_metrics(rewards))
    return generator, baseline, metrics


def find_adam_state(opt_state):
    """Locate actual ScaleByAdamState inside arbitrary Optax chains."""
    if hasattr(opt_state, 'nu') and hasattr(opt_state, 'count'):
        return opt_state
    if isinstance(opt_state, (tuple, list)):
        for child in opt_state:
            found = find_adam_state(child)
            if found is not None:
                return found
    if isinstance(opt_state, dict):
        for child in opt_state.values():
            found = find_adam_state(child)
            if found is not None:
                return found
    return None


def selfplay_rewards(model, historical_model, opt_state, batch, loss_fn, learner_lr, outer_round, beta2=.999, adam_epsilon=1e-8):
    """Single filtered JVP of batched losses; excludes static/nonparameter leaves."""
    adam = find_adam_state(opt_state)
    if adam is None:
        raise ValueError('Self-play requires the actual learner Adam second moments')
    params, static = eqx.partition(model, eqx.is_inexact_array)
    historical = eqx.filter(historical_model, eqx.is_inexact_array)
    correction = 1-beta2**jnp.maximum(adam.count, 1)
    preconditioner = jax.tree.map(lambda v: learner_lr/(jnp.sqrt(v/correction)+adam_epsilon), adam.nu)
    direction = jax.tree.map(lambda h,p,v: (h-p)*v, historical, params, preconditioner)
    def losses(p):
        return loss_fn(eqx.combine(p, static), batch)
    _, directional = eqx.filter_jvp(losses, (params,), (direction,))
    rewards = jnp.where(jnp.asarray(outer_round)==0, jnp.zeros_like(directional), jnp.abs(directional))
    leaves = jax.tree.leaves(preconditioner)
    metrics = {'historical_index':jnp.asarray(outer_round)//2,
        'parameter_movement_norm':optax.global_norm(jax.tree.map(lambda h,p:h-p, historical, params)),
        'preconditioner_mean':sum(jnp.sum(x) for x in leaves)/sum(x.size for x in leaves),
        'preconditioner_max':jnp.max(jnp.stack([jnp.max(x) for x in leaves]))}
    return detached(rewards), metrics


def selfplay_step(generator, generator_optimizer, model, historical_model, opt_state, batch, loss_fn, ids, learner_lr, outer_round, normalization_epsilon=1e-8, expert_iteration_coefficient=1., **generator_kwargs):
    """Fresh on-policy PG + reward-weighted EI, exact categorical prior KL.

    Regime adaptation omits program mutation and off-policy replay. PG uses
    centered/std-normalized rewards; EI uses positive rewards normalized by sum.
    """
    rewards, metrics = selfplay_rewards(model, historical_model, opt_state, batch, loss_fn, learner_lr, outer_round)
    advantages = detached((rewards-jnp.mean(rewards))/(jnp.std(rewards)+normalization_epsilon))
    weights = detached(rewards/(jnp.sum(rewards)+normalization_epsilon))
    q0 = generator_kwargs.get('q0')
    epsilon = generator_kwargs.get('epsilon', .05)
    beta = generator_kwargs.get('kl_coefficient', .01)
    def objective(logits):
        q = probabilities(logits, q0, epsilon)
        log_probs = jnp.log(q)[detached(ids)]
        policy = -jnp.mean(advantages*log_probs)
        expert = -jnp.sum(weights*log_probs)
        prior = beta*distribution_metrics(logits, q0, epsilon)['kl']
        return policy+prior+expert_iteration_coefficient*expert, (policy,prior,expert)
    (loss, (policy,prior,expert)), grad = jax.value_and_grad(objective, has_aux=True)(generator.logits)
    updates, state = generator_optimizer.update(grad, generator.opt_state, generator.logits)
    candidate = GeneratorState(optax.apply_updates(generator.logits, updates), state)
    # Preserve logits AND optimizer at zero movement, including round zero.
    usable = (jnp.asarray(outer_round)>0) & (metrics['parameter_movement_norm']>0)
    new_generator = jax.lax.cond(usable, lambda _: candidate, lambda _: generator, operand=None)
    gm = distribution_metrics(new_generator.logits, q0, epsilon)
    gm.update(teacher_loss=loss, policy_loss=policy, prior_penalty=prior,
              expert_iteration_loss=expert, teacher_grad_norm=optax.global_norm(grad),
              distribution_movement_l1=jnp.sum(jnp.abs(gm['probabilities']-probabilities(generator.logits,q0,epsilon))))
    metrics.update(gm)
    metrics.update(reward_metrics(rewards))
    metrics['reward_normalizer'] = jnp.std(rewards)+normalization_epsilon
    metrics['expert_reward_normalizer'] = jnp.sum(rewards)+normalization_epsilon
    metrics['normalized_reward_mean'] = jnp.mean(advantages)
    metrics['normalized_reward_std'] = jnp.std(advantages)
    metrics['generator_update_applied'] = usable
    return new_generator, metrics


def make_target_objective(mode, loss_fn, feedback_hard, scales, feedback_easy=None):
    """Return scalar objective and raw/scaled components, closing only allowed data.

    Scales are frozen positive zero-holdings MSE, keyed easy/hard. loss_fn
    returns per-episode losses. Hard callers must not provide easy feedback.
    """
    if mode not in ('hard', 'mixed'):
        raise ValueError('Target mode must be hard or mixed')
    if mode == 'hard' and feedback_easy is not None:
        raise ValueError('Hard-only teachers may not receive easy feedback')
    if mode == 'mixed' and feedback_easy is None:
        raise ValueError('Mixed objective requires easy feedback')
    for domain in (('hard',) if mode == 'hard' else ('easy', 'hard')):
        if float(scales[domain]) <= 0:
            raise ValueError('Objective scales must be positive')
    def components(model):
        hard = jnp.mean(loss_fn(model, feedback_hard))
        result = {'hard_raw':hard, 'hard_scaled':hard/scales['hard']}
        if mode == 'mixed':
            easy = jnp.mean(loss_fn(model, feedback_easy))
            result.update(easy_raw=easy, easy_scaled=easy/scales['easy'], raw_average=.5*(easy+hard))
        return result
    def objective(model):
        c = components(model)
        return c['hard_scaled'] if mode == 'hard' else .5*(c['easy_scaled']+c['hard_scaled'])
    return objective, components


def component_metrics(pre, post, components_fn):
    before, after = components_fn(pre), components_fn(post)
    result = {}
    for name in before:
        result[name+'_before'] = before[name]
        result[name+'_after'] = after[name]
        result[name+'_improvement'] = before[name]-after[name]
    return result


def soar_objective_step(generator, generator_optimizer, student_pre, student_post, target_loss_fn, ids, baseline, ema_decay=.9, target_components_fn=None, **generator_kwargs):
    """One persistent update, one hard or balanced mixed scalar reward."""
    before, after = target_loss_fn(student_pre[0]), target_loss_fn(student_post[0])
    improvement = detached(before-after)
    advantage = detached(improvement-baseline)
    next_baseline = ema_decay*baseline+(1-ema_decay)*improvement
    rewards = jnp.broadcast_to(advantage, ids.shape)
    generator, metrics = update_generator(generator, generator_optimizer, ids, rewards, **generator_kwargs)
    metrics.update(reward_metrics(rewards))
    metrics.update(target_before=before, target_after=after, raw_improvement=improvement,
                   ema_baseline=baseline, advantage=advantage)
    if target_components_fn is not None:
        metrics.update(component_metrics(student_pre[0], student_post[0], target_components_fn))
    return generator, next_baseline, metrics
