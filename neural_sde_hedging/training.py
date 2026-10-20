"""GPU-only, resumable neural-SDE curriculum training with paper mechanisms.

No work runs at import. Run as a module on an explicitly provisioned GPU.
The fixed neural-SDE prior supplies demonstration feedback; production studies
should provide independent target training and validation path files.
"""

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import time
from types import SimpleNamespace

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.special import logsumexp

from .checkpoint_codec import CheckpointPickler
from .config import SDEConfig
from .dynamics import NeuralSDE, path_log_prob, sample_noise, simulate
from .hedger import Contract, Hedger, terminal_losses
from .hard_regime import sample_hard
from .methods import (pathwise_meta_learning, dpg_attributions, grpo_objective, promote_soar,
                      selfplay_objective, selfplay_rewards, soar_branches,
                      soar_teacher_objective)
from .replay import ProgramBank, build_pool, niche_of_path
from .metrics import risk_metrics


from .runtime import require_enabled


def entropic(losses, weights, gamma):
    """Return weighted empirical entropic risk from losses/positive weights [B]."""
    batch = losses.shape[0]
    chex.assert_shape((losses, weights), (batch,))
    scaled = gamma * losses + jnp.log(weights)
    chex.assert_shape(scaled, (batch,))
    result = (logsumexp(scaled) - jnp.log(jnp.sum(weights))) / gamma
    chex.assert_shape(result, ())
    return result


def learner_block(student, states, weights, static, optimizer, contract,
                  gamma, steps, key, losses_fn=terminal_losses):
    """Return full student PyTree after differentiable clipped-Adam updates.

    Inputs: student=(array parameters, actual optimizer state), states with
    arbitrary example leading axes followed by [time,state], matching weights,
    shared static architecture and fixed hyperparameters. All optimizer moments
    and clipping derivatives stay inside the DPG differentiation graph.
    """
    n_time, dim = states.shape[-2:]
    flat_states = states.reshape((-1, n_time, dim))
    batch = flat_states.shape[0]
    chex.assert_shape(flat_states, (batch, n_time, dim))
    flat_weights = weights.reshape((batch,))
    chex.assert_shape(flat_weights, (batch,))

    def step(carry, subkey):
        """Apply one bootstrap minibatch update to parameters/moments."""
        params, optimizer_state = carry
        order = jax.random.randint(subkey, (batch,), 0, batch)
        chex.assert_shape(order, (batch,))

        def objective(candidate):
            """Return weighted risk of candidate on the fixed synthetic data."""
            observable = SimpleNamespace(log_spot=flat_states[order, :, 0],
                                         states=flat_states[order])
            losses = losses_fn(eqx.combine(candidate, static), observable, contract)
            chex.assert_shape(losses, (batch,))
            return entropic(losses, flat_weights[order], gamma)

        value, gradient = jax.value_and_grad(objective)(params)
        updates, optimizer_state = optimizer.update(gradient, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        return (params, optimizer_state), value

    result, _ = jax.lax.scan(jax.checkpoint(step), student, jax.random.split(key, steps))
    return result


def gaussian_fit_loss(params, static, states, config, density_fn=path_log_prob):
    """Return mean Gaussian transition NLL from parameters and full states [B,T+1,D].

    Normalize complete Euler path scores by the number of daily transitions;
    diffusion log determinants are included. Initial-state density is excluded.
    """
    batch = states.shape[0]
    chex.assert_shape(states, (batch, config.n_steps + 1, config.state_dim))
    scores = density_fn(eqx.combine(params, static), SimpleNamespace(states=states), config)
    chex.assert_shape(scores, (batch,))
    loss = -jnp.mean(scores) / config.n_steps
    chex.assert_shape(loss, ())
    return loss


def atomic_checkpoint(path, payload):
    """Write checkpoint payload atomically with fsync; inputs path/PyTree, output none."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        CheckpointPickler(stream, protocol=5).dump(jax.device_get(payload))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)



def load_checkpoint(path):
    """Load trusted local checkpoint to device arrays; input path, output PyTree."""
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    return jax.tree.map(lambda value: jnp.asarray(value) if isinstance(value, np.ndarray) else value, payload)


def find_adam(state):
    """Find actual Adam moment/count state in an Optax state tree; raise if absent."""
    if hasattr(state, "nu") and hasattr(state, "count"):
        return state
    if isinstance(state, (tuple, list)):
        for child in state:
            try:
                return find_adam(child)
            except ValueError:
                pass
    raise ValueError("Adam moments missing")


def restore_layout(loaded, template):
    """Restore saved array values using current template's static PyTree layout.

    Pickle may recreate private Equinox static sentinels with new identities.
    Matching every numerical shape/dtype preserves the intended architecture
    without relying on identity equality of those reconstructed sentinels.
    """
    values = jax.tree.leaves(loaded)
    expected, structure = jax.tree.flatten(template)
    if len(values) != len(expected) or any(
        old.shape != new.shape or old.dtype != new.dtype for old, new in zip(values, expected)
    ):
        raise RuntimeError("Checkpoint numerical layout differs from current architecture")
    return jax.tree.unflatten(structure, values)


def teacher_loss(params, static, pool, config, method, behavior, reference,
                 rewards, mutation_mask, density_fn=path_log_prob):
    """Return method-specific teacher loss/metrics on fixed sampled paths.

    Inputs include array parameters, static teacher architecture, joint paths,
    stopped behavior/prior densities/rewards and mutation mask. Output is a
    scalar objective and scalar diagnostic dictionary. Score only path density;
    never backpropagate into the sampled trajectory or learner reward.
    """
    logp = density_fn(eqx.combine(params, static), pool, config)
    if method == "selfplay":
        return selfplay_objective(logp, behavior, reference, rewards, mutation_mask)
    group, items = behavior.shape
    logp = logp.reshape((group, items))
    chex.assert_shape(logp, (group, items))
    if method == "dpg":
        return grpo_objective(logp, behavior, rewards, reference)
    loss, advantages = soar_teacher_objective(logp, rewards)
    return loss, {"advantage_sd": jnp.std(advantages)}


def run(args):
    """Run one method/initialization on GPU; resume every learner/teacher/RNG state."""
    if args.method != "hard-specialist":
        require_enabled()
    if jax.default_backend() != "gpu":
        raise RuntimeError("Only Ruff and py_compile are authorized locally; training requires GPU")
    args.target_regime = getattr(args, "target_regime", "hard")
    args.hard_pool_size = getattr(args, "hard_pool_size", 1024)
    if args.target_regime not in ("hard", "prior", "external"):
        raise ValueError("Unknown target regime")
    if args.method == "hard-specialist" and args.target_regime != "hard":
        raise ValueError("hard-specialist requires the hard target law")
    if args.target_regime == "hard" and getattr(args, "validation_paths", None):
        raise ValueError("Hard evaluation uses fresh simulations; omit --validation-paths")
    if args.target_regime == "external" and not (args.feedback_paths and args.validation_paths):
        raise ValueError("External targets require feedback and independent validation files")
    args.hard_paths = getattr(args, "hard_paths", None)
    args.fit_steps = getattr(args, "fit_steps", 1000)
    args.fit_batch_size = getattr(args, "fit_batch_size", 32)
    args.problem = getattr(args, "problem", "options")
    args.cds_days = getattr(args, "cds_days", 126)
    args.credit_grid_size = getattr(args, "credit_grid_size", 601)
    args.credit_simulation = getattr(args, "credit_simulation", "continuous")
    if args.credit_simulation not in ("continuous", "grid"):
        raise ValueError("Unknown credit simulation")
    if args.problem not in ("options", "credit"):
        raise ValueError("Unknown hedging problem")
    if args.problem == "credit" and args.credit_simulation == "grid" and args.method == "pathwise-meta-learning":
        raise ValueError("Credit grid transitions/defaults are discrete; pathwise gradients are invalid")
    if args.method == "fitted-sde":
        if not args.hard_paths or not getattr(args, "validation_paths", None):
            raise ValueError("fitted-sde requires --hard-paths and independent --validation-paths")
        if args.problem == "credit" and args.credit_simulation != "continuous":
            raise ValueError("Gaussian fitting requires continuous credit states")
        if not args.feedback_paths:
            args.feedback_paths = args.hard_paths
        if min(args.fit_steps, args.fit_batch_size) < 1:
            raise ValueError("Positive fit steps and batch size required")
    if args.target_regime == "hard" and args.problem == "credit" and args.credit_simulation != "continuous":
        raise ValueError("Hard credit target requires continuous credit simulation")
    if args.n_steps is None:
        args.n_steps = 1260 if args.problem == "credit" else 126
    # Defaults keep programmatic callers compatible; persist them in the fingerprint.
    args.validation_size = getattr(args, "validation_size", 2048)
    args.validation_every = getattr(args, "validation_every", 16)
    args.validation_paths = getattr(args, "validation_paths", None)
    args.entropic_ess_fraction = getattr(args, "entropic_ess_fraction", .1)
    if min(args.feedback_size, args.validation_size, args.validation_every) < 1:
        raise ValueError("Feedback/validation sizes and interval must be positive")
    if not 0 < args.entropic_ess_fraction <= 1:
        raise ValueError("Entropic ESS fraction must lie in (0,1]")
    if args.feedback_paths and not args.validation_paths and args.target_regime != "hard":
        raise ValueError("External feedback requires a separate --validation-paths file")
    began = time.monotonic()
    if args.seconds <= 0 or args.rounds < 1 or args.groups < 2 or args.items < 2 or args.inner_steps < 1:
        raise ValueError("Positive budget/steps and at least two groups/items required")
    if args.gamma <= 0 or min(args.student_lr, args.teacher_lr) <= 0:
        raise ValueError("Positive gamma and learning rates required")
    if args.warmup_steps < 0 or args.weight_decay < 0:
        raise ValueError("Warmup steps and weight decay must be nonnegative")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    simulator, density_fn, noise_fn = simulate, path_log_prob, sample_noise
    loss_fn = terminal_losses
    if args.problem == "credit":
        if args.credit_simulation == "continuous":
            from . import credit_continuous as credit
        else:
            from . import credit
        from credit_hedging.policy import Hedger as CreditHedger
        config = credit.CreditSDEConfig(n_steps=args.n_steps, cds_days=args.cds_days,
                                        grid_size=args.credit_grid_size)
        contract = credit.market_contract(config)
        learner = CreditHedger(jax.random.key(args.seed), n_vintages=config.cds_days)
        prior = credit.NeuralCredit(jax.random.key(args.seed + 1), config)
        tables = credit.pricing_tables(prior, config, contract)
        simulator, density_fn, noise_fn = credit.simulate, credit.path_log_prob, credit.sample_noise

        def loss_fn(policy, paths, unused_contract):
            """Return neural credit losses using common fixed-Q CDS pricing tables."""
            if args.credit_simulation == "continuous":
                return credit.terminal_losses(policy, paths, tables, contract, config, args.gamma)
            return credit.terminal_losses(policy, paths, tables, contract, config)
    else:
        config = SDEConfig(n_steps=args.n_steps)
        contract = Contract(payoff_cap=args.payoff_cap, premium=args.premium)
        learner = Hedger(jax.random.key(args.seed))
        prior = NeuralSDE(jax.random.key(args.seed + 1), config)
    configuration = vars(args) | {"sde": asdict(config), "contract": asdict(contract)}
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    if args.problem == "credit":
        source_hashes |= {"credit_hedging/" + p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__).parent.parent / "credit_hedging").glob("*.py")}
    configuration["source_hashes"] = source_hashes
    fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    if not (out / "configuration.json").exists():
        (out / "configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")
    initial_params, learner_static = eqx.partition(learner, eqx.is_inexact_array)
    teacher_params, teacher_static = eqx.partition(prior, eqx.is_inexact_array)
    def learner_rate(count):
        """Return current linear-warmup then constant learning rate from update count."""
        return args.student_lr * jnp.minimum((count + 1.) / max(args.warmup_steps, 1), 1.)

    learner_optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(
        learner_rate, eps_root=1e-16, weight_decay=args.weight_decay))
    teacher_optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(
        args.teacher_lr, weight_decay=args.weight_decay))
    initial_student = (initial_params, learner_optimizer.init(initial_params))
    state = dict(student=initial_student, teacher=teacher_params,
                 teacher_optimizer=teacher_optimizer.init(teacher_params),
                 key_data=jax.random.key_data(jax.random.key(args.seed + 2)),
                 round_index=0, seconds=0., ema=jnp.array(0.), bank=ProgramBank(),
                 soar_reward_window=[],
                 promoted_datasets=[], fingerprint=fingerprint)
    latest = out / "latest.pkl"
    if latest.exists():
        state = load_checkpoint(latest)
        if state["fingerprint"] != fingerprint:
            raise RuntimeError("Resume configuration or source changed")
        state["student"] = restore_layout(state["student"], initial_student)
        state["teacher"] = restore_layout(state["teacher"], teacher_params)
        state["teacher_optimizer"] = restore_layout(
            state["teacher_optimizer"], teacher_optimizer.init(teacher_params))
        restored_programs = {}
        entries = state["bank"].history + [entry for niche in state["bank"].archive.values() for entry in niche]
        for entry in entries:
            identity = id(entry.teacher)
            if identity not in restored_programs:
                restored_programs[identity] = eqx.combine(
                    restore_layout(eqx.filter(entry.teacher, eqx.is_inexact_array), teacher_params), teacher_static)
            entry.teacher = restored_programs[identity]
    history = out / "learner-history"
    history.mkdir(exist_ok=True)
    # One immutable hard training pool; per-round subsets and evaluation RNG are separate.
    compiled_hard = eqx.filter_jit(lambda key, batch: sample_hard(key, batch, config, args.problem))
    # Only target-training data may feed teacher rewards. Validation is disjoint.
    feedback_population = None
    feedback_hash = "fresh-neural-sde-prior-demonstration"
    if args.feedback_paths:
        feedback_file = Path(args.feedback_paths)
        feedback_hash = hashlib.sha256(feedback_file.read_bytes()).hexdigest()
        with np.load(feedback_file, allow_pickle=False) as data:
            feedback_population = jnp.asarray(data["states"])
        chex.assert_shape(feedback_population, (feedback_population.shape[0], config.n_steps + 1, config.state_dim))
        if feedback_population.shape[0] < args.feedback_size:
            raise ValueError("Feedback population smaller than the requested batch")
    if args.target_regime == "hard":
        if feedback_population is None:
            if args.hard_pool_size < max(args.feedback_size, args.groups*args.items):
                raise ValueError("Hard pool must cover feedback and specialist batches")
            feedback_population = compiled_hard(jax.random.key(args.seed + 101), args.hard_pool_size)
            feedback_hash = "fixed-hard-training-pool-v1"
        if feedback_population.shape[0] < max(args.feedback_size, args.groups*args.items):
            raise ValueError("Hard pool too small")
        if not bool(jnp.all(jnp.isfinite(feedback_population))):
            raise ValueError("Hard pool must be finite")
        validation_hash = "fresh-hard-target-v1"
        validation_states = None
    elif args.validation_paths:
        validation_file = Path(args.validation_paths)
        validation_hash = hashlib.sha256(validation_file.read_bytes()).hexdigest()
        if args.feedback_paths and validation_hash == feedback_hash:
            raise ValueError("Feedback and validation files must differ")
        with np.load(validation_file, allow_pickle=False) as data:
            validation_states = jnp.asarray(data["states"])
    else:
        validation_hash = "independent-neural-sde-prior-validation"
        validation_states = simulator(prior, noise_fn(jax.random.key(args.seed + 3), args.validation_size, config), config).states
    if validation_states is not None:
        chex.assert_shape(validation_states, (validation_states.shape[0], config.n_steps + 1, config.state_dim))
        if validation_states.shape[0] < 1:
            raise ValueError("Validation must contain at least one path")
    hard_population = None
    if args.method == "fitted-sde":
        hard_file = Path(args.hard_paths)
        hard_hash = hashlib.sha256(hard_file.read_bytes()).hexdigest()
        if hard_hash == validation_hash:
            raise ValueError("Hard fitting and validation files must differ")
        if state.get("hard_hash", hard_hash) != hard_hash:
            raise RuntimeError("Hard fitting dataset changed during resume")
        with np.load(hard_file, allow_pickle=False) as data:
            hard_population = jnp.asarray(data["states"])
        chex.assert_shape(hard_population, (hard_population.shape[0], config.n_steps + 1, config.state_dim))
        if hard_population.shape[0] < 1 or not bool(jnp.all(jnp.isfinite(hard_population))):
            raise ValueError("Hard fitting states must be nonempty and finite")
        initial = simulator(prior, noise_fn(jax.random.key(args.seed + 4), 1, config), config).states[0, 0]
        if not bool(jnp.all(jnp.isclose(hard_population[:, 0], initial))):
            raise ValueError("Hard fitting paths must use the simulator's fixed initial state")
        state["hard_hash"] = hard_hash
        state.setdefault("fit_step", 0)

    for name, identity in (("feedback_hash", feedback_hash), ("validation_hash", validation_hash)):
        if name in state and state[name] != identity:
            raise RuntimeError("Target dataset changed during resume")
        state[name] = identity

    def target(student, target_states):
        """Return entropic risk on an explicit shared target batch [B,T,D]."""
        losses = loss_fn(eqx.combine(student[0], learner_static),
                                 SimpleNamespace(log_spot=target_states[..., 0], states=target_states), contract)
        return entropic(losses, jnp.ones_like(losses), args.gamma)

    def feedback_metrics(student, target_states):
        """Return target risks and diagnostics of finite-sample tail concentration."""
        policy = eqx.combine(student[0], learner_static)
        if args.problem == "credit" and args.credit_simulation == "continuous":
            losses, result = credit.feedback_metrics(policy,
                SimpleNamespace(states=target_states), tables, contract, config, args.gamma)
        elif args.problem == "credit":
            credit_result = credit.rollout(policy, credit.unpack_states(target_states, config), tables, contract)
            losses = credit_result['loss']
            result = risk_metrics(losses, args.gamma,
                                  daily_marked_errors=credit_result['records']['daily_loss'])
        else:
            losses = loss_fn(policy, SimpleNamespace(log_spot=target_states[..., 0], states=target_states), contract)
            result = risk_metrics(losses, args.gamma)
        batch = losses.shape[0]
        chex.assert_shape(losses, (batch,))
        weights = jax.nn.softmax(args.gamma * losses)
        chex.assert_shape(weights, (batch,))
        ess = 1. / jnp.sum(weights**2)
        chex.assert_shape(ess, ())
        result["entropic_ess"] = ess
        result["entropic_ess_fraction"] = ess / batch
        result["entropic_max_weight"] = jnp.max(weights)
        result["entropic_tail_underresolved"] = ess < args.entropic_ess_fraction * batch
        result["sample_size"] = jnp.asarray(batch)
        return result

    def block(student, states, weights, key):
        """Run a differentiable inner block using the common student architecture."""
        return learner_block(student, states, weights, learner_static,
                             learner_optimizer, contract, args.gamma, args.inner_steps, key, losses_fn=loss_fn)

    # Compilation occurs only at these top-level workflow boundaries.
    compiled_block = eqx.filter_jit(block)
    compiled_target = eqx.filter_jit(target)
    compiled_metrics = eqx.filter_jit(feedback_metrics)
    compiled_density = eqx.filter_jit(lambda teacher, paths: density_fn(teacher, paths, config))
    compiled_simulate = eqx.filter_jit(simulator)
    def adapter_teacher_loss(params, static, paths, sde_config, method, behavior, reference, rewards, mutation_mask):
        """Return teacher objective using this problem's complete path likelihood."""
        return teacher_loss(params, static, paths, sde_config, method, behavior,
                            reference, rewards, mutation_mask, density_fn=density_fn)

    compiled_teacher_gradient = eqx.filter_jit(jax.value_and_grad(adapter_teacher_loss, has_aux=True))

    def execute_adapter(program, noise, sde_config, reference_model):
        """Execute and score an archived neural generator for this hedging problem."""
        paths = simulator(program, noise, sde_config)
        return paths, density_fn(program, paths, sde_config), density_fn(reference_model, paths, sde_config)

    compiled_execute = eqx.filter_jit(execute_adapter)

    def attributed_round(datasets, key, target_states):
        """Run one reset DPG inner calculation on grouped datasets and RNG key."""
        return dpg_attributions(initial_student, datasets,
                                lambda s, d, w: block(s, d, w, key),
                                lambda s: target(s, target_states),
                                (args.items,), cross_group_batching=True)

    def branched_round(student, datasets, keys, target_states):
        """Train all SOAR repeats from one baseline on common per-round feedback."""
        return soar_branches(student, datasets, keys,
                             lambda s, d, k: block(s, d, jnp.ones((args.items,)), k),
                             lambda s: target(s, target_states))

    def aligned_rewards(student, historical, pool):
        """Compute full-pool self-play JVP reward with actual optimizer moments."""
        adam = find_adam(student[1])

        def values(candidate):
            """Return raw losses for current learner on the full mixed round pool."""
            return loss_fn(eqx.combine(candidate, learner_static), pool, contract)

        return selfplay_rewards(student[0], historical, adam.nu, adam.count,
                                learner_rate(adam.count), values, entropic_gamma=args.gamma,
                                epsilon_root=1e-16)

    compiled_attributions = eqx.filter_jit(attributed_round)
    compiled_branches = eqx.filter_jit(branched_round)
    compiled_alignment = eqx.filter_jit(aligned_rewards)

    def pathwise_round(params, student, noise, key, target_states):
        """Return exact one-block teacher objective and live student/path auxiliaries."""
        teacher = eqx.combine(params, teacher_static)
        return pathwise_meta_learning(
            teacher, student, noise, key,
            lambda model, z: simulator(model, z, config),
            lambda s, data, k: block(s, data, jnp.ones((data.shape[0],)), k),
            lambda s: target(s, target_states))

    compiled_pathwise_gradient = eqx.filter_jit(
        jax.value_and_grad(pathwise_round, has_aux=True))
    resumed_seconds = state["seconds"]
    # Fit only on hard training data; validation never enters the likelihood.
    fit_gradient = eqx.filter_jit(jax.value_and_grad(
        lambda params, states: gaussian_fit_loss(params, teacher_static, states, config, density_fn)))
    while (args.method == "fitted-sde" and state["fit_step"] < args.fit_steps
           and resumed_seconds + time.monotonic() - began < args.seconds):
        if shutil.disk_usage(out).free < args.min_free_gb * 1024**3:
            raise RuntimeError("Storage reserve reached during SDE fit")
        next_key, fit_key = jax.random.split(jax.random.wrap_key_data(state["key_data"]))
        indices = jax.random.randint(fit_key, (args.fit_batch_size,), 0, hard_population.shape[0])
        chex.assert_shape(indices, (args.fit_batch_size,))
        nll, gradient = fit_gradient(state["teacher"], hard_population[indices])
        updates, moments = teacher_optimizer.update(gradient, state["teacher_optimizer"], state["teacher"])
        candidate = optax.apply_updates(state["teacher"], updates)
        if any(not bool(jnp.all(jnp.isfinite(v))) for v in jax.tree.leaves((nll, gradient, candidate, moments))):
            raise RuntimeError("Nonfinite Gaussian fit; previous checkpoint preserved")
        state.update(teacher=candidate, teacher_optimizer=moments,
                     fit_step=state["fit_step"] + 1, key_data=jax.random.key_data(next_key),
                     seconds=resumed_seconds + time.monotonic() - began)
        atomic_checkpoint(latest, state)
        with (out / "fit-progress.jsonl").open("a") as stream:
            stream.write(json.dumps({"fit_step": state["fit_step"], "transition_nll": float(nll),
                                     "seconds": state["seconds"]}, allow_nan=False) + "\n")
    if args.method == "fitted-sde" and state["fit_step"] >= args.fit_steps:
        fitted_path = out / "fitted-generator.pkl"
        if not fitted_path.exists():
            atomic_checkpoint(fitted_path, eqx.combine(state["teacher"], teacher_static))
    while (args.method != "fitted-sde" or state["fit_step"] >= args.fit_steps) and state["round_index"] < args.rounds and resumed_seconds + time.monotonic() - began < args.seconds:
        if shutil.disk_usage(out).free < args.min_free_gb * 1024**3:
            raise RuntimeError("Storage reserve reached; preserve checkpoint and export before resuming")
        round_index = state["round_index"]
        key = jax.random.wrap_key_data(state["key_data"])
        next_key, draw_key, inner_key, feedback_key = jax.random.split(key, 4)
        # One target batch per round, independent of teacher-generated training data.
        if feedback_population is None:
            feedback_states = compiled_simulate(
                prior, noise_fn(feedback_key, args.feedback_size, config), config).states
        else:
            indices = jax.random.choice(feedback_key, feedback_population.shape[0],
                                        (args.feedback_size,), replace=False)
            chex.assert_shape(indices, (args.feedback_size,))
            feedback_states = feedback_population[indices]
        chex.assert_shape(feedback_states, (args.feedback_size, config.n_steps + 1, config.state_dim))
        teacher = eqx.combine(state["teacher"], teacher_static)
        student_before = state["student"]
        metrics = {}
        if args.method == "hard-specialist":
            batch = args.groups * args.items
            indices = jax.random.choice(draw_key, feedback_population.shape[0], (batch,), replace=False)
            chex.assert_shape(indices, (batch,))
            training_states = feedback_population[indices]
            chex.assert_shape(training_states, (batch, config.n_steps + 1, config.state_dim))
            pool = SimpleNamespace(states=training_states)
            student = compiled_block(student_before, training_states, jnp.ones((batch,)), inner_key)
            rewards = behavior = reference = jnp.zeros((batch,))
            mutation_mask = jnp.zeros((batch,), dtype=bool)
            apply_teacher = False
        elif args.method == "pathwise-meta-learning":
            noise = noise_fn(draw_key, args.groups * args.items, config)
            ((objective_value, (student, pool, before_risk)), gradient) = compiled_pathwise_gradient(
                state["teacher"], student_before, noise, inner_key, feedback_states)
            rewards = jnp.reshape(-objective_value, (1,))
            behavior = pool.log_density
            reference = compiled_density(prior, pool)
            teacher_metrics = {"target_before": before_risk,
                               "pathwise_risk_change": objective_value}
            apply_teacher = True
        elif args.method == "selfplay":
            snapshot_path = history / f"{round_index:08d}.pkl"
            atomic_checkpoint(snapshot_path, student_before[0])
            old_path = history / f"{round_index // 2:08d}.pkl"
            if not old_path.exists():
                raise RuntimeError("Required historical learner missing; refusing reset")
            old_params = restore_layout(load_checkpoint(old_path), initial_params)
            pool, behavior, reference, sources, programs = build_pool(
                teacher, prior, state["bank"], draw_key, config,
                args.groups * args.items, round_index, host_seed=args.seed,
                execute_fn=compiled_execute, noise_fn=noise_fn)
            rewards = compiled_alignment(student_before, old_params, pool)
            mutation_mask = jnp.array([source == "mutation" for source in sources])
            chex.assert_shape(mutation_mask, (args.groups * args.items,))

            student = compiled_block(student_before, pool.states, jnp.ones_like(rewards), inner_key)
            state["bank"].age()
            for index, program in enumerate(programs):
                state["bank"].add(program, float(jax.device_get(rewards[index])),
                                  niche_of_path(pool.states[index], config.dt), round_index, sources[index])
            apply_teacher = round_index > 0
        else:
            batch = args.groups * args.items
            pool = compiled_simulate(teacher, noise_fn(draw_key, batch, config), config)
            behavior = jax.lax.stop_gradient(pool.log_density).reshape((args.groups, args.items))
            reference = compiled_density(prior, pool).reshape((args.groups, args.items))
            datasets = jax.lax.stop_gradient(pool.states).reshape((args.groups, args.items, config.n_steps + 1, config.state_dim))
            chex.assert_shape(datasets, (args.groups, args.items, config.n_steps + 1, config.state_dim))
            if args.method in ("baseline", "fitted-sde"):
                student = compiled_block(student_before, pool.states,
                                         jnp.ones((batch,)), inner_key)
                rewards = jnp.zeros((batch,))
            elif args.method == "dpg":
                student, rewards, risk = compiled_attributions(datasets, inner_key, feedback_states)
                metrics["target_after"] = risk
            else:
                repeat_keys = jax.random.split(inner_key, args.groups * 4).reshape((args.groups, 4))
                branches, rewards = compiled_branches(student_before, datasets, repeat_keys, feedback_states)
                state["soar_reward_window"] = (
                    state["soar_reward_window"] + [jnp.mean(rewards)])[-3:]
                smoothed_reward = jnp.mean(jnp.stack(state["soar_reward_window"]))
                chex.assert_shape(smoothed_reward, ())
                student, state["ema"], promoted, indices = promote_soar(
                    student_before, branches, rewards, smoothed_reward,
                    args.promotion_threshold, decay=1.)
                if bool(jax.device_get(promoted)):
                    state["promoted_datasets"].append(jax.device_get(datasets[int(indices[0])]))
                metrics["promoted"] = promoted
                metrics["reward_mean"] = jnp.mean(rewards)
                metrics["promotion_smoothed_reward"] = state["ema"]

            apply_teacher = args.method not in ("baseline", "fitted-sde")
            mutation_mask = jnp.zeros((batch,), dtype=bool)
        if args.method in ("baseline", "fitted-sde", "hard-specialist"):
            objective_value, teacher_metrics, gradient = jnp.array(0.), {}, ()
            candidate_teacher = state["teacher"]
            new_teacher_optimizer = state["teacher_optimizer"]
        elif args.method != "pathwise-meta-learning":
            (objective_value, teacher_metrics), gradient = compiled_teacher_gradient(
                state["teacher"], teacher_static, pool, config, args.method,
                behavior, reference, rewards, mutation_mask)
            teacher_updates, new_teacher_optimizer = teacher_optimizer.update(gradient, state["teacher_optimizer"], state["teacher"])
            candidate_teacher = optax.apply_updates(state["teacher"], teacher_updates)
        else:
            teacher_updates, new_teacher_optimizer = teacher_optimizer.update(
                gradient, state["teacher_optimizer"], state["teacher"])
            candidate_teacher = optax.apply_updates(state["teacher"], teacher_updates)
        # Fail before checkpoint commit. No hidden replacement of nonfinite
        # losses/rewards; the previous atomic checkpoint remains recoverable.
        target_risk = compiled_target(student, feedback_states)
        measured_feedback = compiled_metrics(student, feedback_states)
        measured_validation = {}
        elapsed = resumed_seconds + time.monotonic() - began
        if ((round_index + 1) % args.validation_every == 0
                or round_index + 1 >= args.rounds or elapsed >= args.seconds):
            if args.target_regime == "hard":
                evaluation_key = jax.random.fold_in(jax.random.key(args.seed + 102), round_index)
                fresh_validation = compiled_hard(evaluation_key, args.validation_size)
                state["last_validation_key_data"] = jax.random.key_data(evaluation_key)
                measured_validation = compiled_metrics(student, fresh_validation)
            else:
                measured_validation = compiled_metrics(student, validation_states)
        for value in jax.tree.leaves((student, candidate_teacher, new_teacher_optimizer, objective_value,
                                     teacher_metrics, gradient, rewards, behavior, reference, pool.states,
                                     target_risk, measured_feedback, measured_validation)):
            if not bool(jax.device_get(jnp.all(jnp.isfinite(value)))):
                raise RuntimeError("Nonfinite update; previous checkpoint preserved")
        state["student"] = jax.tree.map(jax.lax.stop_gradient, student)
        if apply_teacher:
            state["teacher"] = candidate_teacher
            state["teacher_optimizer"] = new_teacher_optimizer
        state["key_data"] = jax.random.key_data(next_key)
        state["round_index"] += 1
        state["seconds"] = resumed_seconds + time.monotonic() - began
        metrics |= teacher_metrics | {"teacher_loss": objective_value, "target_risk": target_risk,
                                      "teacher_gradient_norm": optax.global_norm(gradient),
                                      "reward_mean": jnp.mean(rewards),
                                      "reward_std": jnp.std(rewards),
                                      "reward_max_abs": jnp.max(jnp.abs(rewards)),
                                      "round": state["round_index"], "seconds": state["seconds"]}
        metrics |= {"feedback_" + name: value for name, value in measured_feedback.items()}
        metrics |= {"validation_" + name: value for name, value in measured_validation.items()}
        atomic_checkpoint(latest, state)
        with (out / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps({name: float(jax.device_get(value)) for name, value in metrics.items()}, allow_nan=False) + "\n")
    atomic_checkpoint(latest, state)
    status = "complete" if state["round_index"] >= args.rounds else "budget_exhausted"
    (out / "status.json").write_text(json.dumps({"status": status, "rounds": state["round_index"], "seconds": state["seconds"]}) + "\n")


def main():
    """Parse one-seed GPU job configuration and run the selected teacher method."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("baseline", "dpg", "soar", "selfplay", "pathwise-meta-learning", "fitted-sde", "hard-specialist"), required=True)
    parser.add_argument("--problem", choices=("options", "credit"), default="options")
    parser.add_argument("--cds-days", type=int, default=126)
    parser.add_argument("--credit-grid-size", type=int, default=601)
    parser.add_argument("--credit-simulation", choices=("continuous", "grid"), default="continuous")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--items", type=int, default=32)
    parser.add_argument("--inner-steps", type=int, default=8)
    parser.add_argument("--n-steps", type=int, help="Default: 126 options; 1260 credit")
    parser.add_argument("--gamma", type=float, default=10.)
    parser.add_argument("--student-lr", type=float, default=0.0003)
    parser.add_argument("--teacher-lr", type=float, default=0.001)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--promotion-threshold", type=float, default=0.0001)
    parser.add_argument("--target-regime", choices=("hard", "prior", "external"), default="hard")
    parser.add_argument("--hard-pool-size", type=int, default=1024)
    parser.add_argument("--hard-paths", help="Full-state hard training NPZ; fitted-sde only")
    parser.add_argument("--fit-steps", type=int, default=1000)
    parser.add_argument("--fit-batch-size", type=int, default=32)
    parser.add_argument("--feedback-paths")
    parser.add_argument("--feedback-size", type=int, default=128)
    parser.add_argument("--validation-paths")
    parser.add_argument("--validation-size", type=int, default=2048)
    parser.add_argument("--validation-every", type=int, default=16)
    parser.add_argument("--entropic-ess-fraction", type=float, default=.1)
    parser.add_argument("--payoff-cap", type=float)
    parser.add_argument("--premium", type=float, default=0.)
    parser.add_argument("--min-free-gb", type=float, default=1.)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
