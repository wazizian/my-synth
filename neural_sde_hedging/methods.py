"""Paper mechanisms as pure JAX callbacks; no simulation, IO, or implicit resets.

Student states are arbitrary array PyTrees including optimizer moments. Callers
own sampling, fixed held-out batches, checkpointing, and optimizer updates.
"""

import chex
import jax
import jax.numpy as jnp


def pathwise_meta_learning(teacher, student, noise, key, simulate_fn,
                           block_fn, target_fn):
    """Return downstream risk change and trained student/paths for one unroll.

    Inputs: teacher/student PyTrees, fixed primitive noise and minibatch key,
    simulate_fn(teacher,noise)->paths, block_fn(student,states,key)->student,
    target_fn(student)->scalar risk on independent target paths. Outputs are
    scalar risk change and (trained student, generated paths, baseline risk).
    The teacher, paths, student updates and optimizer moments remain connected.
    Random draws and the incoming student are outer-round boundary conditions;
    callers differentiate this function with respect to the teacher only.
    """
    paths = simulate_fn(teacher, noise)
    batch, n_time, state_dim = paths.states.shape
    chex.assert_shape(paths.states, (batch, n_time, state_dim))
    before = target_fn(student)
    chex.assert_shape(before, ())
    trained = block_fn(student, paths.states, key)
    after = target_fn(trained)
    chex.assert_shape(after, ())
    change = after - before
    chex.assert_shape(change, ())
    return change, (trained, paths, before)


def dpg_attributions(initial_student, datasets, block_fn, target_fn,
                     weight_shape, cross_group_batching=True):
    """Attribute grouped synthetic datasets through reset inner learning.

    Inputs: fixed initial student PyTree, dataset PyTree with leading group axis,
    block_fn(student,data,weights)->student, target_fn(student)->scalar risk,
    weight_shape for each group's examples. Outputs: post-student PyTree(s),
    signed rewards [group,*weight_shape], and scalar target risk (summed across
    groups when batching is disabled).
    The fixed initialization includes optimizer state; no state persists here.
    """
    group = jax.tree.leaves(datasets)[0].shape[0]
    weights = jnp.ones((group, *weight_shape))
    chex.assert_shape(weights, (group, *weight_shape))
    initial = jax.tree.map(jax.lax.stop_gradient, initial_student)

    def attributed(w):
        """Run reset inner block(s) at weights w and return target risk/state."""
        if cross_group_batching:
            student = block_fn(initial, datasets, w)
            risk = target_fn(student)
            chex.assert_shape(risk, ())
        else:
            student = jax.vmap(lambda data, wi: block_fn(initial, data, wi))(
                datasets, w)
            risks = jax.vmap(target_fn)(student)
            chex.assert_shape(risks, (group,))
            risk = jnp.sum(risks)
            chex.assert_shape(risk, ())
        return risk, student

    (risk, students), derivatives = jax.value_and_grad(
        attributed, has_aux=True)(weights)
    chex.assert_shape(derivatives, (group, *weight_shape))
    rewards = jax.lax.stop_gradient(-derivatives)
    chex.assert_shape(rewards, (group, *weight_shape))
    return students, rewards, risk


def grpo_objective(current_logp, behavior_logp, rewards, prior_logp,
                   clip_epsilon=.2, beta=0., epsilon=1e-8):
    """Return clipped GRPO surrogate and metrics from [group,prompt] arrays.

    Rewards are signed DPG attribution rewards. Normalize across candidate
    groups for each prompt; old policy, rewards, and advantages are detached.
    Continuous SDE path densities replace language-model sequence probabilities.
    The sampled GRPO KL surrogate is not an exact continuous-law KL gradient.
    With one update starting on-policy, ratio clipping is initially inactive.
    """
    group, prompts = rewards.shape
    chex.assert_shape((current_logp, behavior_logp, rewards, prior_logp),
                      (group, prompts))
    centered = rewards - jnp.mean(rewards, axis=0, keepdims=True)
    chex.assert_shape(centered, (group, prompts))
    advantage = jax.lax.stop_gradient(centered / (
        jnp.std(rewards, axis=0, keepdims=True) + epsilon))
    chex.assert_shape(advantage, (group, prompts))
    ratio = jnp.exp(current_logp - jax.lax.stop_gradient(behavior_logp))
    chex.assert_shape(ratio, (group, prompts))
    clipped = jnp.clip(ratio, 1. - clip_epsilon, 1. + clip_epsilon)
    chex.assert_shape(clipped, (group, prompts))
    surrogate = jnp.minimum(ratio * advantage, clipped * advantage)
    chex.assert_shape(surrogate, (group, prompts))
    delta = prior_logp - current_logp
    chex.assert_shape(delta, (group, prompts))
    objective = -jnp.mean(surrogate)
    kl_penalty = jnp.array(0., dtype=current_logp.dtype)
    chex.assert_shape(kl_penalty, ())
    if beta != 0.:
        sampled_kl = jnp.exp(delta) - delta - 1.
        chex.assert_shape(sampled_kl, (group, prompts))
        kl_penalty = beta * jnp.mean(sampled_kl)
        chex.assert_shape(kl_penalty, ())
        objective = objective + kl_penalty
    chex.assert_shape(objective, ())
    return objective, {"advantage_std": jnp.std(advantage),
                       "kl_penalty": kl_penalty,
                       "prior_log_ratio_mean": -jnp.mean(delta)}


def soar_branches(baseline, datasets, keys, block_fn, target_fn):
    """Return SOAR branch states and risk improvements [group,repeats].

    Every block_fn(baseline,dataset,key) starts from the identical full baseline.
    target_fn closes over one shared held-out set; target risk is minimized.
    keys has leading [group,repeats], conventionally repeats=4. Branch randomness
    is algorithmic repeat variance reduction, not independent initialization.
    """
    group, repeats = keys.shape[:2]
    chex.assert_shape(keys, (group, repeats, *keys.shape[2:]))
    initial = jax.tree.map(jax.lax.stop_gradient, baseline)
    before = target_fn(initial)
    chex.assert_shape(before, ())

    def one_dataset(data, repeat_keys):
        """Train repeated branches from the same baseline on one dataset."""
        return jax.vmap(lambda key: block_fn(initial, data, key))(repeat_keys)

    branches = jax.vmap(one_dataset)(datasets, keys)
    after = jax.vmap(jax.vmap(target_fn))(branches)
    chex.assert_shape(after, (group, repeats))
    improvements = before - after
    chex.assert_shape(improvements, (group, repeats))
    return branches, jax.lax.stop_gradient(improvements)


def soar_teacher_objective(logp, rewards, beta=0., prior_logp=None):
    """Return dataset RLOO teacher loss from logp [group,items], rewards [G,R].

    Repeat rewards are averaged first; each generated row in a dataset receives
    that dataset's leave-one-out advantage. Optional prior KL uses sampled
    log-density ratios on fresh teacher samples, with a score-function correction
    for sampling-law dependence. This is a local on-policy KL gradient estimator,
    not an exact off-policy objective for repeated optimization on stale rows.
    """
    group, items = logp.shape
    if group < 2:
        raise ValueError("SOAR RLOO requires at least two candidate datasets")
    chex.assert_shape(rewards, (group, rewards.shape[1]))
    mean_rewards = jnp.mean(rewards, axis=1)
    chex.assert_shape(mean_rewards, (group,))
    other_mean = (jnp.sum(mean_rewards) - mean_rewards) / (group - 1)
    chex.assert_shape(other_mean, (group,))
    advantages = jax.lax.stop_gradient(mean_rewards - other_mean)
    chex.assert_shape(advantages, (group,))
    dataset_logp = jnp.sum(logp, axis=1)
    chex.assert_shape(dataset_logp, (group,))
    loss = -jnp.mean(advantages * dataset_logp)
    chex.assert_shape(loss, ())
    if prior_logp is not None and beta != 0.:
        chex.assert_shape(prior_logp, (group, items))
        delta = logp - jax.lax.stop_gradient(prior_logp)
        chex.assert_shape(delta, (group, items))
        score_correction = jnp.mean(jax.lax.stop_gradient(delta) * logp)
        chex.assert_shape(score_correction, ())
        penalty = (jnp.mean(delta) + score_correction
                   - jax.lax.stop_gradient(score_correction))
        chex.assert_shape(penalty, ())
        loss = loss + beta * penalty
        chex.assert_shape(loss, ())
    return loss, advantages


def promote_soar(baseline, branches, rewards, previous_ema,
                  threshold, decay=2. / 3.):
    """Return selected baseline, smoothing state, promotion flag and [g,r].

    A positive threshold is measured in declared target-risk improvement units.
    Best mean-reward dataset wins; its median-ranked repeat is promoted only
    when EMA exceeds threshold. Callers archive the winning dataset iff promoted.
    """
    group, repeats = rewards.shape
    mean_rewards = jnp.mean(rewards, axis=1)
    chex.assert_shape(mean_rewards, (group,))
    ema = decay * previous_ema + (1. - decay) * jnp.mean(mean_rewards)
    chex.assert_shape(ema, ())
    best_group = jnp.argmax(mean_rewards)
    chex.assert_shape(best_group, ())
    ranked = jnp.argsort(rewards[best_group])
    chex.assert_shape(ranked, (repeats,))
    median_repeat = ranked[(repeats - 1) // 2]
    chex.assert_shape(median_repeat, ())
    promote = ema > threshold
    chex.assert_shape(promote, ())
    candidate = jax.tree.map(lambda x: x[best_group, median_repeat], branches)
    selected = jax.tree.map(lambda old, new: jnp.where(promote, new, old),
                            baseline, candidate)
    return selected, ema, promote, (best_group, median_repeat)


def selfplay_rewards(params, historical_params, adam_nu, adam_count,
                     learning_rate, losses_fn, beta2=.999, epsilon=1e-8,
                     entropic_gamma=None, epsilon_root=0.):
    """Return absolute preconditioned historical-alignment rewards [batch].

    Inputs are matching parameter PyTrees; adam_nu is the uncorrected second
    moment. historical_params must be the pre-round floor(round/2) snapshot.
    losses_fn(params)->[batch] provides raw per-example learner losses.
    entropic_gamma=None follows the additive paper objective; specifying gamma
    explicitly adopts batch*softmax(gamma*loss) weighted loss-gradient alignment
    for empirical entropic risk. We do not differentiate those weighting factors.
    epsilon_root must match the actual learner Adam eps_root inside its square
    root; epsilon matches the outside-square-root epsilon.
    """
    correction = 1. - beta2 ** jnp.maximum(adam_count, 1)
    chex.assert_shape(correction, ())
    direction = jax.tree.map(
        lambda old, now, nu: (old - now) * learning_rate /
        (jnp.sqrt(nu / correction + epsilon_root) + epsilon),
        historical_params, params, adam_nu)
    losses, directional = jax.jvp(losses_fn, (params,), (direction,))
    batch = losses.shape[0]
    chex.assert_shape((losses, directional), (batch,))
    if entropic_gamma is not None:
        factors = batch * jax.nn.softmax(entropic_gamma * losses)
        chex.assert_shape(factors, (batch,))
        directional = directional * jax.lax.stop_gradient(factors)
        chex.assert_shape(directional, (batch,))
    rewards = jax.lax.stop_gradient(jnp.abs(directional))
    chex.assert_shape(rewards, (batch,))
    return rewards


def selfplay_objective(current_logp, behavior_logp, prior_logp, rewards,
                       mutation_mask, beta=.01, expert_coefficient=1.,
                       epsilon=1e-8):
    """Return paper Eq4+Eq5 self-play objective and scalar metrics.

    All inputs have shape [batch]. Replay densities refer to fresh reexecuted
    paths under their stored proposal snapshots. Mutations train the learner
    and expert iteration but are excluded from policy gradient. Rewards must
    be recomputed for the current learner, never reused from archive ranking.
    """
    batch = rewards.shape[0]
    chex.assert_shape((current_logp, behavior_logp, prior_logp, rewards,
                       mutation_mask), (batch,))
    stopped_rewards = jax.lax.stop_gradient(rewards)
    chex.assert_shape(stopped_rewards, (batch,))
    standardized = (stopped_rewards - jnp.mean(stopped_rewards)) / (
        jnp.std(stopped_rewards) + epsilon)
    chex.assert_shape(standardized, (batch,))
    advantage = jax.lax.stop_gradient(
        standardized - beta * (current_logp - prior_logp))
    chex.assert_shape(advantage, (batch,))
    log_ratio = jnp.clip(current_logp - behavior_logp, -20., 20.)
    chex.assert_shape(log_ratio, (batch,))
    ratio = jax.lax.stop_gradient(jnp.exp(log_ratio))
    chex.assert_shape(ratio, (batch,))
    pg_rows = jnp.logical_not(mutation_mask)
    chex.assert_shape(pg_rows, (batch,))
    weighted_score = jnp.where(pg_rows, ratio * current_logp * advantage, 0.)
    chex.assert_shape(weighted_score, (batch,))
    policy = -jnp.sum(weighted_score) / jnp.maximum(jnp.sum(pg_rows), 1)
    chex.assert_shape(policy, ())
    positive = jnp.maximum(stopped_rewards, 0.)
    chex.assert_shape(positive, (batch,))
    expert_weights = positive / jnp.maximum(jnp.sum(positive), epsilon)
    chex.assert_shape(expert_weights, (batch,))
    expert = -jnp.sum(expert_weights * current_logp)
    chex.assert_shape(expert, ())
    loss = policy + expert_coefficient * expert
    chex.assert_shape(loss, ())
    return loss, {"policy": policy, "expert": expert,
                  "reward_mean": jnp.mean(rewards),
                  "reward_std": jnp.std(rewards),
                  "ratio_max": jnp.max(ratio)}
