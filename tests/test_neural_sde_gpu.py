"""GPU-only integration checks; opt in with NEURAL_SDE_GPU_TESTS=1.

Never run on the local one-vCPU control plane. This module skips before JAX
import unless explicitly enabled, and rejects execution without a GPU backend.
"""
# ruff: noqa: E402
import os
import json
from types import SimpleNamespace

import numpy as np
import pytest

if os.environ.get("NEURAL_SDE_GPU_TESTS") != "1":
    pytest.skip("Explicit GPU validation opt-in required", allow_module_level=True)

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

if jax.default_backend() != "gpu":
    raise RuntimeError("Explicit neural-SDE validation requires a GPU backend")

from neural_sde_hedging.config import SDEConfig
from neural_sde_hedging.dynamics import (NeuralSDE, SDEPaths, path_log_prob,
                                         sample_noise, simulate,
                                         transition_coefficients,
                                         transition_log_prob)
from neural_sde_hedging.hedger import Contract, Hedger, terminal_losses
from neural_sde_hedging.methods import (dpg_attributions, selfplay_objective,
                                       selfplay_rewards, soar_branches)
from neural_sde_hedging.replay import ProgramBank, build_pool
from neural_sde_hedging.training import atomic_checkpoint, learner_block, load_checkpoint, run


@pytest.fixture
def small_model():
    """Return a tiny static configuration and seeded neural teacher for GPU checks."""
    config = SDEConfig(n_steps=3, n_latent=2)
    teacher = NeuralSDE(jax.random.key(11), config, width=4, depth=1)
    return config, teacher


def test_joint_shapes_score_and_diffusion_gradient(small_model):
    """Check simulator shapes, exact rescoring, and finite nonzero diffusion score gradients."""
    config, teacher = small_model
    noise = sample_noise(jax.random.key(12), 3, config)
    paths = simulate(teacher, noise, config)
    chex.assert_shape(paths.states, (3, 4, 3))
    chex.assert_shape(paths.log_spot, (3, 4))
    chex.assert_shape(paths.latent, (3, 4, 2))
    np.testing.assert_allclose(path_log_prob(teacher, paths, config), paths.log_density, rtol=2e-5, atol=2e-5)

    def score(candidate):
        """Return summed fixed-sample log likelihood for a candidate teacher."""
        return jnp.sum(path_log_prob(candidate, paths, config))

    gradients = eqx.filter_grad(score)(teacher)
    for leaf in jax.tree.leaves(gradients):
        assert np.isfinite(np.asarray(leaf)).all()
    magnitude = sum(float(jnp.sum(jnp.abs(leaf))) for leaf in jax.tree.leaves(gradients.diffusion_network))
    assert magnitude > 1e-6

    def coordinate_score(states):
        """Return teacher likelihood when supplied states are treated as score samples."""
        replaced = eqx.tree_at(lambda p: p.states, paths, states)
        return jnp.sum(path_log_prob(teacher, replaced, config))

    np.testing.assert_array_equal(jax.grad(coordinate_score)(paths.states), jnp.zeros_like(paths.states))


def test_euler_density_independent_gaussian_oracle(small_model):
    """Compare full joint transition density with an independent NumPy Gaussian calculation."""
    config, teacher = small_model
    paths = simulate(teacher, sample_noise(jax.random.key(13), 2, config), config)
    before, after = paths.states[:, 0], paths.states[:, 1]
    drift, diffusion = transition_coefficients(teacher, before, jnp.array(0.), config)
    actual = transition_log_prob(after, before, jnp.array(0.), teacher, config)
    expected = []
    for index in range(2):
        covariance = config.dt * np.asarray(diffusion[index]) @ np.asarray(diffusion[index]).T
        residual = np.asarray(after[index] - before[index] - config.dt * drift[index])
        determinant_sign, logdet = np.linalg.slogdet(covariance)
        assert determinant_sign > 0
        expected.append(-0.5 * (config.state_dim * np.log(2 * np.pi) + logdet + residual @ np.linalg.solve(covariance, residual)))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_reexecution_uses_new_noise_and_actual_behavior_density(small_model):
    """Check replay reexecutes historical programs and recomputes their new-path likelihoods."""
    config, teacher = small_model
    bank = ProgramBank()
    bank.add(teacher, 1., (0, 0), 0, "fresh")
    first = build_pool(teacher, teacher, bank, jax.random.key(14), config, 2, 1,
                       mutation_fraction=0., replay_fraction=.5)
    second = build_pool(teacher, teacher, bank, jax.random.key(15), config, 2, 1,
                        mutation_fraction=0., replay_fraction=.5)
    pool, behavior, _, sources, programs = first
    assert sources == ["replay", "fresh"]
    assert not np.array_equal(pool.innovations[0], second[0].innovations[0])
    for index, program in enumerate(programs):
        row = SDEPaths(pool.states[index:index + 1], pool.innovations[index:index + 1],
                       pool.log_density[index:index + 1], pool.times)
        np.testing.assert_allclose(behavior[index], path_log_prob(program, row, config)[0], rtol=2e-5)


def test_history_only_hedger_and_capped_loss_accounting(small_model):
    """Prove latent invariance numerically and challenge the analytic bound with extreme stock paths."""
    config, teacher = small_model
    paths = simulate(teacher, sample_noise(jax.random.key(16), 2, config), config)
    model = Hedger(jax.random.key(17), hidden_size=4)
    contract = Contract(payoff_cap=.5)
    hidden_changed = paths.states.at[..., 1:].set(1e6)
    altered = eqx.tree_at(lambda p: p.states, paths, hidden_changed)
    np.testing.assert_array_equal(terminal_losses(model, paths, contract), terminal_losses(model, altered, contract))
    extreme = SimpleNamespace(log_spot=jnp.array([[0., -20., 20., 40.], [0., 20., -20., -40.]]))
    losses = terminal_losses(model, extreme, contract)
    assert np.isfinite(np.asarray(losses)).all()
    assert bool(jnp.all(losses <= contract.loss_upper_bound(config.n_steps) + 1e-5))
    assert np.isinf(Contract().loss_upper_bound(config.n_steps))
    with pytest.raises(ValueError):
        Contract(cost=1.)


def test_checkpoint_preserves_generator_bank_rng_and_learner(tmp_path, small_model):
    """Round-trip full bank, learner history, optimizer-like state, and RNG without resets."""
    _, teacher = small_model
    bank = ProgramBank()
    bank.add(teacher, 2., (1, 2), 4, "fresh")
    bank.age()
    payload = dict(bank=bank, student={"weights": jnp.array([1., 2.]), "count": jnp.array(7)},
                   key_data=jax.random.key_data(jax.random.key(18)), round_index=5,
                   historical_params={"weights": jnp.array([.5, 1.])}, ema=jnp.array(.3))
    destination = tmp_path / "latest.pkl"
    atomic_checkpoint(destination, payload)
    loaded = load_checkpoint(destination)
    assert loaded["round_index"] == 5
    np.testing.assert_array_equal(loaded["student"]["count"], 7)
    np.testing.assert_array_equal(loaded["key_data"], payload["key_data"])
    np.testing.assert_array_equal(loaded["historical_params"]["weights"], [.5, 1.])
    assert len(loaded["bank"].history) == 1
    assert loaded["bank"].history[0].reward == 2.
    assert loaded["bank"].archive[(1, 2)][0].reward == pytest.approx(1.94)
    for old, new in zip(jax.tree.leaves(teacher), jax.tree.leaves(loaded["bank"].history[0].teacher)):
        np.testing.assert_array_equal(old, new)


def test_soar_common_baseline_and_dpg_fixed_reset():
    """Check SOAR branches share full optimizer state and DPG restarts its supplied initialization."""
    initial = {"weight": jnp.array(2.), "moment": jnp.array(3.), "count": jnp.array(7)}
    datasets = jnp.array([[.1, .2], [.3, .4]])
    keys = jax.random.split(jax.random.key(19), 8).reshape((2, 4))

    def block(student, data, key):
        """Advance a toy full learner state once; ignore key to expose branch initialization."""
        del key
        return dict(weight=student["weight"] + jnp.mean(data),
                    moment=student["moment"] + 1., count=student["count"] + 1)

    def target(student):
        """Return toy squared-weight target risk from full learner state."""
        return student["weight"] ** 2

    branches, improvements = soar_branches(initial, datasets, keys, block, target)
    chex.assert_shape(improvements, (2, 4))
    np.testing.assert_array_equal(branches["count"], np.full((2, 4), 8))
    np.testing.assert_array_equal(branches["moment"], np.full((2, 4), 4.))

    def weighted_block(student, data, weights):
        """Advance a toy reset learner with differentiable unnormalized example weights."""
        return dict(weight=student["weight"] + jnp.mean(data * weights),
                    moment=student["moment"] + 1., count=student["count"] + 1)

    post, rewards, risk = dpg_attributions(initial, datasets, weighted_block, target, (2,))
    again, repeated_rewards, repeated_risk = dpg_attributions(initial, datasets, weighted_block, target, (2,))
    assert int(post["count"]) == 8
    np.testing.assert_array_equal(post["weight"], again["weight"])
    np.testing.assert_array_equal(rewards, repeated_rewards)
    np.testing.assert_array_equal(risk, repeated_risk)


def test_actual_hedger_adam_dpg_metagradients_finite(small_model):
    """Differentiate through an actual GRU/clipped-Adam block and verify finite attribution rewards."""
    config, teacher = small_model
    paths = simulate(teacher, sample_noise(jax.random.key(20), 4, config), config)
    model = Hedger(jax.random.key(21), hidden_size=4)
    params, static = eqx.partition(model, eqx.is_inexact_array)
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(3e-4, eps_root=1e-16))
    initial = (params, optimizer.init(params))
    contract = Contract(payoff_cap=.5)
    datasets = paths.states.reshape((2, 2, config.n_steps + 1, config.state_dim))
    chex.assert_shape(datasets, (2, 2, 4, 3))

    def block(student, states, weights):
        """Return actual full learner state after one differentiable Adam update."""
        return learner_block(student, states, weights, static, optimizer, contract,
                             10., 1, jax.random.key(22))

    def target(student):
        """Return mean held-out loss under a frozen shared set of states."""
        return jnp.mean(terminal_losses(eqx.combine(student[0], static), paths, contract))

    compiled = eqx.filter_jit(lambda data: dpg_attributions(initial, data, block, target, (2,)))
    post, rewards, risk = compiled(datasets)
    chex.assert_shape(rewards, (2, 2))
    for leaf in jax.tree.leaves((post, rewards, risk)):
        assert np.isfinite(np.asarray(leaf)).all()


def test_selfplay_historical_alignment_and_mutation_exclusion():
    """Check zero historical movement and that mutation samples have no policy-gradient term."""
    params = {"weight": jnp.array([1., 2.])}
    moments = {"weight": jnp.array([0., 0.])}

    def losses(candidate):
        """Return additive example losses for checking the historical JVP reward."""
        return candidate["weight"] ** 2

    reward = selfplay_rewards(params, params, moments, jnp.array(0), 3e-4,
                             losses, epsilon_root=1e-16)
    np.testing.assert_array_equal(reward, [0., 0.])
    logp = jnp.array([-1., -2.])
    mutation_mask = jnp.array([True, True])

    def objective(current):
        """Return teacher loss with expert iteration disabled and every row marked mutation."""
        return selfplay_objective(current, logp, logp, jnp.array([1., 2.]),
                                  mutation_mask, expert_coefficient=0.)[0]

    np.testing.assert_array_equal(jax.grad(objective)(logp), [0., 0.])


@pytest.mark.parametrize("method", ("baseline", "dpg", "soar", "selfplay"))
def test_complete_training_and_resume(tmp_path, method):
    """Run two tiny real GPU rounds per method and verify exact no-reset resume."""
    args = SimpleNamespace(method=method, output=str(tmp_path / method), seconds=600.,
                           rounds=2, seed=20261017, groups=2, items=2, inner_steps=1,
                           n_steps=3, gamma=10., student_lr=3e-4, teacher_lr=1e-3,
                           warmup_steps=2, weight_decay=.01,
                           promotion_threshold=1e-4, feedback_paths=None,
                           feedback_size=4, payoff_cap=.5, premium=0., min_free_gb=.01)
    run(args)
    directory = tmp_path / method
    with (directory / "status.json").open() as stream:
        status = json.load(stream)
    assert status["status"] == "complete"
    before = load_checkpoint(directory / "latest.pkl")
    assert before["round_index"] == 2
    assert np.isfinite(before["seconds"])
    if method == "baseline":
        prior = NeuralSDE(jax.random.key(args.seed + 1), SDEConfig(n_steps=args.n_steps))
        for saved, original in zip(jax.tree.leaves(before["teacher"]),
                                   jax.tree.leaves(eqx.filter(prior, eqx.is_inexact_array))):
            np.testing.assert_array_equal(saved, original)
    if method == "selfplay":
        assert len(before["bank"].history) > 0
        assert (directory / "learner-history" / "00000000.pkl").exists()
    run(args)
    after = load_checkpoint(directory / "latest.pkl")
    assert after["round_index"] == before["round_index"]
    np.testing.assert_array_equal(after["key_data"], before["key_data"])
    for old, new in zip(jax.tree.leaves(before["student"]), jax.tree.leaves(after["student"])):
        np.testing.assert_array_equal(old, new)
