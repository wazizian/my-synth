"""GPU-only, resumable neural-SDE curriculum training with paper mechanisms.

No work runs at import. Run as a module on an explicitly provisioned GPU.
The fixed prior supplies demonstration feedback; production studies should
provide independently frozen target paths using --feedback-paths.
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
from .methods import (dpg_attributions, grpo_objective, promote_soar,
                      selfplay_objective, selfplay_rewards, soar_branches,
                      soar_teacher_objective)
from .replay import ProgramBank, build_pool, niche_of_path
from .metrics import risk_metrics


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
                  gamma, steps, key):
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
            observable = SimpleNamespace(log_spot=flat_states[order, :, 0])
            losses = terminal_losses(eqx.combine(candidate, static), observable, contract)
            chex.assert_shape(losses, (batch,))
            return entropic(losses, flat_weights[order], gamma)

        value, gradient = jax.value_and_grad(objective)(params)
        updates, optimizer_state = optimizer.update(gradient, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        return (params, optimizer_state), value

    result, _ = jax.lax.scan(jax.checkpoint(step), student, jax.random.split(key, steps))
    return result


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
                 rewards, mutation_mask):
    """Return method-specific teacher loss/metrics on fixed sampled paths.

    Inputs include array parameters, static teacher architecture, joint paths,
    stopped behavior/prior densities/rewards and mutation mask. Output is a
    scalar objective and scalar diagnostic dictionary. Score only path density;
    never backpropagate into the sampled trajectory or learner reward.
    """
    logp = path_log_prob(eqx.combine(params, static), pool, config)
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
    if jax.default_backend() != "gpu":
        raise RuntimeError("Only Ruff and py_compile are authorized locally; training requires GPU")
    began = time.monotonic()
    if args.seconds <= 0 or args.rounds < 1 or args.groups < 2 or args.items < 2 or args.inner_steps < 1:
        raise ValueError("Positive budget/steps and at least two groups/items required")
    if args.gamma <= 0 or min(args.student_lr, args.teacher_lr) <= 0:
        raise ValueError("Positive gamma and learning rates required")
    if args.warmup_steps < 0 or args.weight_decay < 0:
        raise ValueError("Warmup steps and weight decay must be nonnegative")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = SDEConfig(n_steps=args.n_steps)
    contract = Contract(payoff_cap=args.payoff_cap, premium=args.premium)
    configuration = vars(args) | {"sde": asdict(config), "contract": asdict(contract)}
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    configuration["source_hashes"] = source_hashes
    fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    if not (out / "configuration.json").exists():
        (out / "configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")
    learner = Hedger(jax.random.key(args.seed))
    initial_params, learner_static = eqx.partition(learner, eqx.is_inexact_array)
    prior = NeuralSDE(jax.random.key(args.seed + 1), config)
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
    # Feedback is frozen outside every teacher update, never regenerated from
    # the learned teacher. Its data hash becomes part of checkpoint identity.
    if args.feedback_paths:
        feedback_file = Path(args.feedback_paths)
        feedback_hash = hashlib.sha256(feedback_file.read_bytes()).hexdigest()
        with np.load(feedback_file, allow_pickle=False) as data:
            feedback_states = jnp.asarray(data["states"])
    else:
        feedback_hash = "fixed-prior-demonstration-only"
        feedback_states = simulate(prior, sample_noise(jax.random.key(args.seed + 3), args.feedback_size, config), config).states
    chex.assert_shape(feedback_states, (feedback_states.shape[0], config.n_steps + 1, config.state_dim))
    if "feedback_hash" in state and state["feedback_hash"] != feedback_hash:
        raise RuntimeError("Feedback dataset changed during resume")
    state["feedback_hash"] = feedback_hash

    def target(student):
        """Return risk on the same fixed feedback paths for every compared branch."""
        losses = terminal_losses(eqx.combine(student[0], learner_static),
                                 SimpleNamespace(log_spot=feedback_states[..., 0]), contract)
        return entropic(losses, jnp.ones_like(losses), args.gamma)

    def feedback_metrics(student):
        """Return empirical risk, fractional CVaR95 and std on frozen feedback."""
        losses = terminal_losses(eqx.combine(student[0], learner_static),
                                 SimpleNamespace(log_spot=feedback_states[..., 0]), contract)
        result = risk_metrics(losses, args.gamma)
        weights = jax.nn.softmax(args.gamma * losses)
        chex.assert_shape(weights, losses.shape)
        result["entropic_ess"] = 1. / jnp.sum(weights**2)
        return result

    def block(student, states, weights, key):
        """Run a differentiable inner block using the common student architecture."""
        return learner_block(student, states, weights, learner_static,
                             learner_optimizer, contract, args.gamma, args.inner_steps, key)

    # Compilation occurs only at these top-level workflow boundaries.
    compiled_block = eqx.filter_jit(block)
    compiled_target = eqx.filter_jit(target)
    compiled_metrics = eqx.filter_jit(feedback_metrics)
    compiled_density = eqx.filter_jit(lambda teacher, paths: path_log_prob(teacher, paths, config))
    compiled_simulate = eqx.filter_jit(simulate)
    compiled_teacher_gradient = eqx.filter_jit(jax.value_and_grad(teacher_loss, has_aux=True))

    def attributed_round(datasets, key):
        """Run one reset DPG inner calculation on grouped datasets and RNG key."""
        return dpg_attributions(initial_student, datasets,
                                lambda s, d, w: block(s, d, w, key), target,
                                (args.items,), cross_group_batching=True)

    def branched_round(student, datasets, keys):
        """Train all SOAR repeats from one baseline on common fixed feedback."""
        return soar_branches(student, datasets, keys,
                             lambda s, d, k: block(s, d, jnp.ones((args.items,)), k), target)

    def aligned_rewards(student, historical, pool):
        """Compute full-pool self-play JVP reward with actual optimizer moments."""
        adam = find_adam(student[1])

        def values(candidate):
            """Return raw losses for current learner on the full mixed round pool."""
            return terminal_losses(eqx.combine(candidate, learner_static), pool, contract)

        return selfplay_rewards(student[0], historical, adam.nu, adam.count,
                                learner_rate(adam.count), values, entropic_gamma=args.gamma,
                                epsilon_root=1e-16)

    compiled_attributions = eqx.filter_jit(attributed_round)
    compiled_branches = eqx.filter_jit(branched_round)
    compiled_alignment = eqx.filter_jit(aligned_rewards)
    resumed_seconds = state["seconds"]
    while state["round_index"] < args.rounds and resumed_seconds + time.monotonic() - began < args.seconds:
        if shutil.disk_usage(out).free < args.min_free_gb * 1024**3:
            raise RuntimeError("Storage reserve reached; preserve checkpoint and export before resuming")
        round_index = state["round_index"]
        key = jax.random.wrap_key_data(state["key_data"])
        next_key, draw_key, inner_key = jax.random.split(key, 3)
        teacher = eqx.combine(state["teacher"], teacher_static)
        student_before = state["student"]
        metrics = {}
        if args.method == "selfplay":
            snapshot_path = history / f"{round_index:08d}.pkl"
            atomic_checkpoint(snapshot_path, student_before[0])
            old_path = history / f"{round_index // 2:08d}.pkl"
            if not old_path.exists():
                raise RuntimeError("Required historical learner missing; refusing reset")
            old_params = restore_layout(load_checkpoint(old_path), initial_params)
            pool, behavior, reference, sources, programs = build_pool(
                teacher, prior, state["bank"], draw_key, config,
                args.groups * args.items, round_index, host_seed=args.seed)
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
            pool = compiled_simulate(teacher, sample_noise(draw_key, batch, config), config)
            behavior = jax.lax.stop_gradient(pool.log_density).reshape((args.groups, args.items))
            reference = compiled_density(prior, pool).reshape((args.groups, args.items))
            datasets = jax.lax.stop_gradient(pool.states).reshape((args.groups, args.items, config.n_steps + 1, config.state_dim))
            chex.assert_shape(datasets, (args.groups, args.items, config.n_steps + 1, config.state_dim))
            if args.method == "baseline":
                student = compiled_block(student_before, pool.states,
                                         jnp.ones((batch,)), inner_key)
                rewards = jnp.zeros((batch,))
            elif args.method == "dpg":
                student, rewards, risk = compiled_attributions(datasets, inner_key)
                metrics["target_after"] = risk
            else:
                repeat_keys = jax.random.split(inner_key, args.groups * 4).reshape((args.groups, 4))
                branches, rewards = compiled_branches(student_before, datasets, repeat_keys)
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

            apply_teacher = args.method != "baseline"
            mutation_mask = jnp.zeros((batch,), dtype=bool)
        if args.method == "baseline":
            objective_value, teacher_metrics, gradient = jnp.array(0.), {}, ()
            candidate_teacher = state["teacher"]
            new_teacher_optimizer = state["teacher_optimizer"]
        else:
            (objective_value, teacher_metrics), gradient = compiled_teacher_gradient(
                state["teacher"], teacher_static, pool, config, args.method,
                behavior, reference, rewards, mutation_mask)
            teacher_updates, new_teacher_optimizer = teacher_optimizer.update(gradient, state["teacher_optimizer"], state["teacher"])
            candidate_teacher = optax.apply_updates(state["teacher"], teacher_updates)
        # Fail before checkpoint commit. No hidden replacement of nonfinite
        # losses/rewards; the previous atomic checkpoint remains recoverable.
        target_risk = compiled_target(student)
        measured_feedback = compiled_metrics(student)
        for value in jax.tree.leaves((student, candidate_teacher, new_teacher_optimizer, objective_value,
                                     teacher_metrics, gradient, rewards, behavior, reference, pool.states,
                                     target_risk, measured_feedback)):
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
        atomic_checkpoint(latest, state)
        with (out / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps({name: float(jax.device_get(value)) for name, value in metrics.items()}, allow_nan=False) + "\n")
    atomic_checkpoint(latest, state)
    status = "complete" if state["round_index"] >= args.rounds else "budget_exhausted"
    (out / "status.json").write_text(json.dumps({"status": status, "rounds": state["round_index"], "seconds": state["seconds"]}) + "\n")


def main():
    """Parse one-seed GPU job configuration and run the selected teacher method."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("baseline", "dpg", "soar", "selfplay"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--items", type=int, default=32)
    parser.add_argument("--inner-steps", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=126)
    parser.add_argument("--gamma", type=float, default=10.)
    parser.add_argument("--student-lr", type=float, default=0.0003)
    parser.add_argument("--teacher-lr", type=float, default=0.001)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--promotion-threshold", type=float, default=0.0001)
    parser.add_argument("--feedback-paths")
    parser.add_argument("--feedback-size", type=int, default=512)
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
