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
from neural_sde_hedging import training as training_module
from neural_sde_hedging import credit, credit_continuous
from neural_sde_hedging.dynamics import (NeuralSDE, SDEPaths, path_log_prob,
                                         sample_noise, simulate,
                                         transition_coefficients,
                                         transition_log_prob)
from neural_sde_hedging.hedger import Contract, Hedger, terminal_losses
from neural_sde_hedging.hard_regime import sample_hard, sample_regime
from neural_sde_hedging.methods import (pathwise_meta_learning, dpg_attributions, selfplay_objective,
                                       selfplay_rewards, soar_branches)
from neural_sde_hedging.replay import ProgramBank, build_pool
from neural_sde_hedging.training import atomic_checkpoint, gaussian_fit_loss, learner_block, load_checkpoint, run


@pytest.fixture
def small_model():
    """Return a tiny static configuration and seeded neural teacher for GPU checks."""
    config = SDEConfig(n_steps=3, n_latent=2)
    teacher = NeuralSDE(jax.random.key(11), config, width=4, depth=1)
    return config, teacher


def test_pathwise_meta_learning_matches_finite_difference(small_model):
    """Check full SDE-to-Adam-to-target derivative against fixed-noise differences."""
    config, teacher = small_model
    params, static = eqx.partition(teacher, eqx.is_inexact_array)
    learner = Hedger(jax.random.key(81), hidden_size=4)
    learner_params, learner_static = eqx.partition(learner, eqx.is_inexact_array)
    optimizer = optax.adam(1e-3, eps_root=1e-8)
    student = (learner_params, optimizer.init(learner_params))
    contract = Contract(payoff_cap=1.)
    noise = sample_noise(jax.random.key(82), 8, config)
    feedback = simulate(teacher, sample_noise(jax.random.key(83), 8, config), config)

    def target(state):
        """Return independent mean squared terminal error of student state."""
        losses = terminal_losses(eqx.combine(state[0], learner_static), feedback, contract)
        return jnp.mean(losses**2)

    def objective(scale):
        """Perturb teacher diffusion biases and differentiate a two-step Adam unroll."""
        candidate = eqx.tree_at(lambda p: p.diffusion_network.layers[-1].bias,
                                params, params.diffusion_network.layers[-1].bias + scale)
        value, _ = pathwise_meta_learning(
            eqx.combine(candidate, static), student, noise, jax.random.key(84),
            lambda model, z: simulate(model, z, config),
            lambda s, data, key: learner_block(s, data, jnp.ones((8,)),
                learner_static, optimizer, contract, 10., 2, key), target)
        return value

    compiled = jax.jit(jax.value_and_grad(objective))
    value, derivative = compiled(jnp.array(0.))
    epsilon = 2e-3
    plus, _ = compiled(jnp.array(epsilon))
    minus, _ = compiled(jnp.array(-epsilon))
    difference = (plus - minus) / (2 * epsilon)
    assert np.isfinite(float(value)) and np.isfinite(float(derivative))
    assert abs(float(derivative)) > 1e-9
    np.testing.assert_allclose(derivative, difference, rtol=.15, atol=2e-7)


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


@pytest.mark.parametrize("problem,method,credit_simulation", [
    ("options", method, "continuous") for method in ("baseline", "dpg", "soar", "selfplay", "pathwise-meta-learning", "fitted-sde")
] + [("credit", method, "grid") for method in ("baseline", "dpg", "soar", "selfplay")]
  + [("credit", method, "continuous") for method in ("baseline", "dpg", "soar", "selfplay", "pathwise-meta-learning", "fitted-sde")])
def test_complete_training_and_resume(tmp_path, problem, method, credit_simulation, monkeypatch):
    """Run two tiny real GPU rounds per method and verify exact no-reset resume."""
    monkeypatch.setenv("NEURAL_SDE_ENABLED", "1")
    args = SimpleNamespace(method=method, target_regime="prior", problem=problem, credit_simulation=credit_simulation,
                           cds_days=2, credit_grid_size=301,
                           output=str(tmp_path / method), seconds=600.,
                           rounds=2, seed=20261017, groups=2, items=2, inner_steps=1,
                           n_steps=3, gamma=10., student_lr=3e-4, teacher_lr=1e-3,
                           warmup_steps=2, weight_decay=.01,
                           promotion_threshold=1e-4, feedback_paths=None,
                           feedback_size=4, validation_size=8, validation_every=2,
                           validation_paths=None, entropic_ess_fraction=.1, payoff_cap=.5, premium=0., min_free_gb=.01)
    if method == "fitted-sde":
        if problem == "credit":
            cfg = credit_continuous.CreditSDEConfig(n_steps=3, cds_days=2, grid_size=301)
            model = credit_continuous.NeuralCredit(jax.random.key(args.seed + 1), cfg)
            simulator, noise_fn = credit_continuous.simulate, credit_continuous.sample_noise
        else:
            cfg = SDEConfig(n_steps=3)
            model = NeuralSDE(jax.random.key(args.seed + 1), cfg)
            simulator, noise_fn = simulate, sample_noise
        for name, count, seed in (("hard", 8, 951), ("validation", 8, 952)):
            states = simulator(model, noise_fn(jax.random.key(seed), count, cfg), cfg).states
            np.savez(tmp_path / f"{name}.npz", states=np.asarray(states))
        args.target_regime = "external"
        args.feedback_paths = str(tmp_path / "hard.npz")
        args.hard_paths = str(tmp_path / "hard.npz")
        args.validation_paths = str(tmp_path / "validation.npz")
        args.fit_steps, args.fit_batch_size = 2, 4
    run(args)
    directory = tmp_path / method
    with (directory / "status.json").open() as stream:
        status = json.load(stream)
    assert status["status"] == "complete"
    before = load_checkpoint(directory / "latest.pkl")
    assert before["round_index"] == 2
    rows = [json.loads(line) for line in (directory / "progress.jsonl").read_text().splitlines()]
    assert "validation_entropic" not in rows[0]
    assert "validation_entropic" in rows[1]
    assert rows[0]["feedback_sample_size"] == 4
    assert rows[1]["validation_sample_size"] == 8
    assert 0 < rows[0]["feedback_entropic_ess_fraction"] <= 1.00001
    assert 0 < rows[0]["feedback_entropic_max_weight"] <= 1
    expected_key = jax.random.key(args.seed + 2)
    if method == "fitted-sde":
        for _ in range(args.fit_steps):
            expected_key = jax.random.split(expected_key)[0]
        assert before["fit_step"] == args.fit_steps
        frozen = load_checkpoint(directory / "fitted-generator.pkl")
        for fitted, current in zip(jax.tree.leaves(eqx.filter(frozen, eqx.is_inexact_array)),
                                   jax.tree.leaves(before["teacher"])):
            np.testing.assert_array_equal(fitted, current)
        fit_rows = (directory / "fit-progress.jsonl").read_text().splitlines()
        assert len(fit_rows) == args.fit_steps
        assert all(np.isfinite(json.loads(row)["transition_nll"]) for row in fit_rows)
    for _ in range(2):
        expected_key = jax.random.split(expected_key, 4)[0]
    np.testing.assert_array_equal(before["key_data"], jax.random.key_data(expected_key))
    assert np.isfinite(before["seconds"])
    if method == "baseline":
        if problem == "credit":
            backend = credit_continuous if credit_simulation == "continuous" else credit
            prior = backend.NeuralCredit(jax.random.key(args.seed + 1),
                                          backend.CreditSDEConfig(n_steps=args.n_steps))
        else:
            prior = NeuralSDE(jax.random.key(args.seed + 1), SDEConfig(n_steps=args.n_steps))
        for saved, original in zip(jax.tree.leaves(before["teacher"]),
                                   jax.tree.leaves(eqx.filter(prior, eqx.is_inexact_array))):
            np.testing.assert_array_equal(saved, original)
    if problem == "credit":
        assert np.isfinite(rows[1]["validation_daily_error_std"])
    if method == "selfplay":
        assert len(before["bank"].history) > 0
        assert (directory / "learner-history" / "00000000.pkl").exists()
    run(args)
    after = load_checkpoint(directory / "latest.pkl")
    assert after["round_index"] == before["round_index"]
    np.testing.assert_array_equal(after["key_data"], before["key_data"])
    if method == "fitted-sde":
        assert after["fit_step"] == args.fit_steps
        assert len((directory / "fit-progress.jsonl").read_text().splitlines()) == args.fit_steps
        for old, new in zip(jax.tree.leaves(before["teacher"]), jax.tree.leaves(after["teacher"])):
            np.testing.assert_array_equal(old, new)
    for old, new in zip(jax.tree.leaves(before["student"]), jax.tree.leaves(after["student"])):
        np.testing.assert_array_equal(old, new)


def test_neural_credit_pricing_density_default_and_checkpoint(tmp_path):
    """Check neural credit law against independent cashflow/likelihood recursions."""
    config = credit.CreditSDEConfig(n_steps=5, cds_days=2, grid_size=31,
                                    grid_min=-3.8, grid_max=-3.2)
    teacher = credit.NeuralCredit(jax.random.key(91), config, width=4, depth=1)
    contract = credit.market_contract(config)
    tables = credit.pricing_tables(teacher, config, contract)
    grid, q_transition = credit.transition_kernel(teacher, config, "Q")
    np.testing.assert_allclose(np.sum(np.asarray(q_transition), axis=1), 1., atol=2e-6)
    assert np.all(np.asarray(tables['hazard']) > 0)
    survival = np.exp(-np.asarray(tables['hazard']) * config.dt)
    discount = np.exp(-contract.rate * config.dt)
    bond = np.ones(config.grid_size)
    for remaining in range(1, config.n_steps + 1):
        bond = discount * ((1-survival)*contract.recovery + survival*(np.asarray(q_transition) @ bond))
        np.testing.assert_allclose(tables['bond'][remaining], bond, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(tables['protection'][config.cds_days] -
        tables['par_spread'] * tables['annuity'][config.cds_days], 0., atol=1e-8)
    noise = credit.sample_noise(jax.random.key(92), 32, config)
    paths = credit.simulate(teacher, noise, config)
    np.testing.assert_allclose(credit.path_log_prob(teacher, paths, config), paths.log_density, atol=2e-5)
    _, p_transition = credit.transition_kernel(teacher, config)
    unpacked = credit.unpack_states(paths.states, config)
    states = np.asarray(unpacked['state'])
    alive = np.asarray(unpacked['alive'])
    defaults = np.asarray(unpacked['default'])
    oracle = np.zeros(32)
    for day in range(config.n_steps):
        before, after = states[:, day], states[:, day+1]
        row = np.where(defaults[:, day], np.log1p(-survival[before]),
                       np.log(survival[before]) + np.log(np.asarray(p_transition)[before, after]))
        oracle += np.where(alive[:, day], row, 0.)
    np.testing.assert_allclose(paths.log_density, oracle, atol=3e-5)

    def score(candidate):
        """Return fixed-path mean likelihood for the neural credit risk network."""
        return jnp.mean(credit.path_log_prob(candidate, paths, config))

    gradients = eqx.filter_grad(score)(teacher)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(gradients))
    assert sum(float(jnp.sum(jnp.abs(x))) for x in jax.tree.leaves(gradients)) > 1e-8
    forced = credit.simulate(teacher, jnp.ones_like(noise), config)
    forced_paths = credit.unpack_states(forced.states, config)
    assert np.asarray(forced_paths['default'][:, 0]).all()
    assert not np.asarray(forced_paths['alive'][:, 1:]).any()
    result = credit.rollout(None, forced_paths, tables, contract)
    np.testing.assert_allclose(result['loss'], result['initial_price'] -
                               discount*contract.recovery, atol=2e-6)
    np.testing.assert_allclose(jnp.sum(result['records']['daily_loss'], axis=1), result['loss'], atol=2e-6)
    checkpoint = tmp_path / "credit-generator.pkl"
    atomic_checkpoint(checkpoint, teacher)
    restored = load_checkpoint(checkpoint)
    np.testing.assert_array_equal(credit.path_log_prob(restored, paths, config),
                                  credit.path_log_prob(teacher, paths, config))


def test_neural_credit_rejects_invalid_pathwise_training(tmp_path, monkeypatch):
    """Reject unsupported gradients through discrete defaults before training starts."""
    monkeypatch.setenv("NEURAL_SDE_ENABLED", "1")
    args = SimpleNamespace(problem="credit", credit_simulation="grid", method="pathwise-meta-learning",
                           output=str(tmp_path), n_steps=3)
    with pytest.raises(ValueError, match="discrete"):
        run(args)


def test_continuous_credit_default_marginalization_and_metrics(tmp_path):
    """Compare every default branch to the original ledger and weighted raw metrics."""
    from credit_hedging.policy import Hedger as CreditHedger

    config = credit_continuous.CreditSDEConfig(n_steps=3, cds_days=2, grid_size=101,
                                             grid_min=-4., grid_max=-3.)
    teacher = credit_continuous.NeuralCredit(jax.random.key(101), config, width=4, depth=1)
    contract = credit_continuous.market_contract(config)
    tables = credit_continuous.pricing_tables(teacher, config, contract)
    policy = CreditHedger(jax.random.key(102), n_vintages=2, hidden_size=4)
    indices = jnp.array([[50, 51, 49, 52], [50, 48, 51, 50]])
    states = tables['grid'][indices][..., None]
    paths = SimpleNamespace(states=states)
    result = credit_continuous.default_branches(policy, paths, tables, contract, config)
    log_probability = np.asarray(result['log_probability'])
    factors = np.asarray(states[..., 0])
    survival = np.exp(-np.logaddexp(0., factors[:, :-1])*config.dt)
    expected_weights = np.empty((2, 4))
    preceding = np.ones(2)
    for day in range(3):
        expected_weights[:, day] = preceding*(1-survival[:, day])
        preceding *= survival[:, day]
    expected_weights[:, -1] = preceding
    np.testing.assert_allclose(np.exp(log_probability), expected_weights, atol=3e-7)
    np.testing.assert_allclose(np.exp(log_probability).sum(axis=1), 1., atol=2e-6)
    daily = []
    for event in range(4):
        alive = jnp.broadcast_to(jnp.arange(4)[None, :] <= event, (2, 4))
        defaults = jnp.broadcast_to(jnp.arange(3)[None, :] == event, (2, 3))
        ledger = credit.rollout(policy, dict(state=indices, alive=alive, default=defaults), tables, contract)
        np.testing.assert_allclose(result['event_loss'][:, event], ledger['loss'], atol=3e-6)
        np.testing.assert_allclose(jnp.sum(ledger['records']['daily_loss'], axis=1), ledger['loss'], atol=3e-6)
        daily.append(np.asarray(ledger['records']['daily_loss']))
    daily = np.stack(daily, axis=1)
    mean = np.sum(expected_weights[..., None]*daily)/(2*3)
    second = np.sum(expected_weights[..., None]*daily**2)/(2*3)
    np.testing.assert_allclose(result['daily_error_std'], np.sqrt(max(second-mean**2, 0)), atol=3e-6)
    conditional, metrics = credit_continuous.feedback_metrics(policy, paths, tables, contract, config)
    raw = np.asarray(result['event_loss'])
    expected_entropic = np.log(np.sum(expected_weights*np.exp(10*raw))/2)/10
    np.testing.assert_allclose(metrics['entropic'], expected_entropic, atol=3e-6)
    raw_mean = np.sum(expected_weights*raw)/2
    np.testing.assert_allclose(metrics['terminal_std'], np.sqrt(np.sum(expected_weights*(raw-raw_mean)**2)/2), atol=3e-6)
    order = np.argsort(raw.reshape(-1))[::-1]
    weights = expected_weights.reshape(-1)[order]/2
    tail = np.minimum(weights, np.maximum(.05-(np.cumsum(weights)-weights), 0.))
    np.testing.assert_allclose(metrics['cvar95'], np.sum(tail*raw.reshape(-1)[order])/.05, atol=3e-6)
    np.testing.assert_allclose(conditional, credit_continuous.terminal_losses(policy, paths, tables, contract, config), atol=1e-7)
    generated = credit_continuous.simulate(teacher, credit_continuous.sample_noise(jax.random.key(103), 8, config), config)
    np.testing.assert_allclose(credit_continuous.path_log_prob(teacher, generated, config), generated.log_density, atol=3e-5)

    def score(candidate):
        """Return mean stopped-path likelihood for the full continuous teacher."""
        return jnp.mean(credit_continuous.path_log_prob(candidate, generated, config))

    gradients = eqx.filter_grad(score)(teacher)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(gradients))
    assert sum(float(jnp.sum(jnp.abs(x))) for x in jax.tree.leaves(gradients.diffusion_network)) > 1e-7
    checkpoint = tmp_path / "continuous-credit.pkl"
    atomic_checkpoint(checkpoint, teacher)
    restored = load_checkpoint(checkpoint)
    np.testing.assert_array_equal(credit_continuous.path_log_prob(restored, generated, config),
                                  credit_continuous.path_log_prob(teacher, generated, config))


def test_continuous_credit_pathwise_gradient_through_default_and_adam():
    """Check diffusion->default-integrated credit loss->Adam->target derivative by FD."""
    config = credit_continuous.CreditSDEConfig(n_steps=5, cds_days=2, grid_size=101,
                                             grid_min=-4., grid_max=-3.)
    teacher = credit_continuous.NeuralCredit(jax.random.key(111), config, width=4, depth=1)
    params, static = eqx.partition(teacher, eqx.is_inexact_array)
    contract = credit_continuous.market_contract(config)
    tables = credit_continuous.pricing_tables(teacher, config, contract)
    noise = credit_continuous.sample_noise(jax.random.key(112), 8, config)
    optimizer = optax.adam(.1, eps_root=.1)
    initial = jnp.array(0.)
    student = (initial, optimizer.init(initial))

    def block(state, states, unused_key):
        """Fit a scalar student to default-integrated losses with two Adam updates."""
        losses = credit_continuous.terminal_losses(None, SimpleNamespace(states=states), tables, contract, config)

        def update(carry, unused):
            """Apply Adam to squared distance from the conditional credit targets."""
            value, moments = carry
            gradient = 2*(value-jnp.mean(losses))
            chex.assert_shape(gradient, ())
            updates, moments = optimizer.update(gradient, moments, value)
            value = optax.apply_updates(value, updates)
            chex.assert_shape(value, ())
            return (value, moments), None

        trained, _ = jax.lax.scan(update, state, None, length=2)
        return trained

    def objective(shift):
        """Return target improvement after perturbing the neural diffusion bias."""
        candidate = eqx.tree_at(lambda p: p.diffusion_network.layers[-1].bias,
            params, params.diffusion_network.layers[-1].bias+shift)
        loss, _ = pathwise_meta_learning(eqx.combine(candidate, static), student, noise,
            jax.random.key(113), lambda model, z: credit_continuous.simulate(model, z, config),
            block, lambda state: (state[0]-.1)**2)
        return loss

    compiled = jax.jit(jax.value_and_grad(objective))
    value, derivative = compiled(jnp.array(0.))
    plus, _ = compiled(jnp.array(.01))
    minus, _ = compiled(jnp.array(-.01))
    assert np.isfinite(float(value)) and np.isfinite(float(derivative))
    assert abs(float(derivative)) > 1e-10
    np.testing.assert_allclose(derivative, (plus-minus)/.02, rtol=.15, atol=3e-7)


def test_gaussian_fit_loss_uses_every_transition_and_log_determinant(small_model):
    """Compare fitting NLL against an independent Gaussian calculation on every step."""
    config, model = small_model
    paths = simulate(model, sample_noise(jax.random.key(947), 4, config), config)
    params, static = eqx.partition(model, eqx.is_inexact_array)
    expected = []
    for time_index in range(config.n_steps):
        before, after = paths.states[:, time_index], paths.states[:, time_index + 1]
        drift, lower = transition_coefficients(model, before, jnp.asarray(time_index * config.dt), config)
        covariance = np.asarray(lower) @ np.swapaxes(np.asarray(lower), -1, -2) * config.dt
        residual = np.asarray(after - before - config.dt * drift)
        quadratic = np.einsum("bi,bij,bj->b", residual, np.linalg.inv(covariance), residual)
        _, logdet = np.linalg.slogdet(covariance)
        expected.append(.5 * (config.state_dim * np.log(2*np.pi) + logdet + quadratic))
    actual, gradient = jax.value_and_grad(gaussian_fit_loss)(params, static, paths.states, config)
    np.testing.assert_allclose(actual, np.mean(expected), rtol=2e-5, atol=2e-5)
    assert np.isfinite(float(optax.global_norm(gradient)))
    assert float(optax.global_norm(gradient.diffusion_network)) > 0


@pytest.mark.parametrize("problem", ["options", "credit"])
@pytest.mark.parametrize("regime", ["easy", "hard"])
def test_hard_sampler_matches_reference(problem, regime):
    """Check the fixed hard law, persistent day-35 shock and fresh independent draws."""
    config = SDEConfig(n_steps=40) if problem == "options" else credit_continuous.CreditSDEConfig(n_steps=40)
    key, batch = jax.random.key(210), 4
    actual = sample_regime(key, batch, config, problem, regime)
    noise = np.asarray(jax.random.normal(key, (config.n_steps, batch, config.state_dim)))
    decay = np.exp(-np.log(2.)/np.asarray((40., 3.) if problem == "options" else (40.,)))
    if regime == "easy":
        decay = np.exp(-np.log(2.)/np.full_like(decay, 3.))
    sd = .35*np.sqrt(config.dt*(1-decay**2)/(-2*np.log(decay)))
    state = np.zeros((batch, config.state_dim))
    if problem == "credit":
        state[:, 0] = config.initial_latent
    expected = [state.copy()]
    for day, z in enumerate(noise):
        shock = np.log(2.) if day == 35 and regime == "hard" else 0.
        if problem == "options":
            factors = state[:, 1:] + shock
            variance = .04*np.exp(np.sum(factors, axis=1))
            spot = state[:, 0] - .5*variance*config.dt + np.sqrt(variance*config.dt)*z[:, 0]
            state = np.concatenate((spot[:, None], factors*decay + sd*z[:, 1:]), axis=1)
        else:
            state = config.initial_latent + (state-config.initial_latent+shock)*decay + sd*z
        expected.append(state.copy())
    np.testing.assert_allclose(actual, np.stack(expected, axis=1), rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(actual, sample_regime(key, batch, config, problem, regime))
    if regime == "hard":
        np.testing.assert_array_equal(actual, sample_hard(key, batch, config, problem))
    fresh = sample_regime(jax.random.fold_in(key, 1), batch, config, problem, regime)
    assert not np.array_equal(actual[:, 1:], fresh[:, 1:])


@pytest.mark.parametrize("problem", ["options", "credit"])
@pytest.mark.parametrize("method", ["dpg", "soar", "hard-specialist", "easy-specialist"])
def test_hard_target_pool_fresh_evaluation_and_specialist(tmp_path, monkeypatch, problem, method):
    """Exercise shared fixed-pool targets, fresh hard evaluation and resumed RNG on GPU."""
    if method.endswith("-specialist"):
        monkeypatch.delenv("NEURAL_SDE_ENABLED", raising=False)
    else:
        monkeypatch.setenv("NEURAL_SDE_ENABLED", "1")
    regime = "easy" if method == "easy-specialist" else "hard"
    args = SimpleNamespace(method=method, target_regime=None, hard_pool_size=8,
                           problem=problem, credit_simulation="continuous", cds_days=2,
                           credit_grid_size=31, output=str(tmp_path / method), seconds=600.,
                           rounds=2, seed=20261017, groups=2, items=2, inner_steps=1,
                           n_steps=3, gamma=10., student_lr=3e-4, teacher_lr=1e-3,
                           warmup_steps=2, weight_decay=.01, promotion_threshold=1e-4,
                           feedback_paths=None, feedback_size=4, validation_paths=None,
                           validation_size=8, validation_every=1, entropic_ess_fraction=.1,
                           payoff_cap=.5, premium=0., min_free_gb=.01)
    run(args)
    out = tmp_path / method
    state = load_checkpoint(out / "latest.pkl")
    assert state["feedback_hash"] == f"fixed-{regime}-training-pool-v1"
    assert state["validation_hash"] == f"fresh-{regime}-target-v1"
    expected = jax.random.key_data(jax.random.fold_in(jax.random.key(args.seed+(102 if regime == "hard" else 104)), 1))
    np.testing.assert_array_equal(state["last_validation_key_data"], expected)
    rows = [json.loads(row) for row in (out / "progress.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["validation_sample_size"] == 8 for row in rows)
    for evaluated, offset in (("hard", 102), ("easy", 104)):
        assert all(row[f"evaluation_{evaluated}_sample_size"] == 8 for row in rows)
        first_key = jax.random.key_data(jax.random.fold_in(jax.random.key(args.seed+offset), 0))
        last_key = jax.random.key_data(jax.random.fold_in(jax.random.key(args.seed+offset), 1))
        assert not np.array_equal(first_key, last_key)
        np.testing.assert_array_equal(state[f"last_{evaluated}_validation_key_data"], last_key)
    assert all(row["validation_entropic"] == row[f"evaluation_{regime}_entropic"] for row in rows)
    if method.endswith("-specialist"):
        assert all(row["teacher_gradient_norm"] == 0 for row in rows)
        if problem == "options":
            initial = Hedger(jax.random.key(args.seed))
        else:
            from credit_hedging.policy import Hedger as CreditHedger
            initial = CreditHedger(jax.random.key(args.seed), n_vintages=args.cds_days)
        changed = [not np.array_equal(before, after)
                   for before, after in zip(jax.tree.leaves(eqx.filter(initial, eqx.is_inexact_array)),
                                             jax.tree.leaves(state["student"][0]))]
        assert any(changed)
    run(args)
    resumed = load_checkpoint(out / "latest.pkl")
    np.testing.assert_array_equal(state["key_data"], resumed["key_data"])
    np.testing.assert_array_equal(state["last_validation_key_data"], resumed["last_validation_key_data"])


@pytest.mark.parametrize("method", ["easy-specialist", "hard-specialist"])
def test_specialist_rejects_external_pool(tmp_path, method, monkeypatch):
    """Reject relabeling arbitrary supplied paths as a built-in specialist regime."""
    monkeypatch.delenv("NEURAL_SDE_ENABLED", raising=False)
    args = SimpleNamespace(method=method, target_regime=None, feedback_paths="arbitrary.npz")
    with pytest.raises(ValueError, match="generated regime pool"):
        run(args)


@pytest.mark.parametrize("method", ["easy-specialist", "hard-specialist"])
def test_specialist_interrupted_resume_matches_uninterrupted(tmp_path, monkeypatch, method):
    """Reconstruct the fixed regime pool after interruption and reproduce the final learner."""
    monkeypatch.delenv("NEURAL_SDE_ENABLED", raising=False)
    args = SimpleNamespace(method=method, target_regime=None, hard_pool_size=8,
                           problem="options", credit_simulation="continuous", cds_days=2,
                           credit_grid_size=31, output=str(tmp_path / "interrupted"), seconds=600.,
                           rounds=2, seed=20261017, groups=2, items=2, inner_steps=1,
                           n_steps=3, gamma=10., student_lr=3e-4, teacher_lr=1e-3,
                           warmup_steps=2, weight_decay=.01, promotion_threshold=1e-4,
                           feedback_paths=None, feedback_size=4, validation_paths=None,
                           validation_size=8, validation_every=1, entropic_ess_fraction=.1,
                           payoff_cap=.5, premium=0., min_free_gb=.01)
    original_save = training_module.atomic_checkpoint

    def save_then_interrupt(path, payload):
        """Persist one completed round, then simulate a process failure before its log append."""
        original_save(path, payload)
        if path.name == "latest.pkl" and payload.get("round_index") == 1:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(training_module, "atomic_checkpoint", save_then_interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run(args)
    interrupted = load_checkpoint(tmp_path / "interrupted" / "latest.pkl")
    assert interrupted["round_index"] == 1
    monkeypatch.setattr(training_module, "atomic_checkpoint", original_save)
    run(args)
    resumed = load_checkpoint(tmp_path / "interrupted" / "latest.pkl")
    args.output = str(tmp_path / "uninterrupted")
    run(args)
    reference = load_checkpoint(tmp_path / "uninterrupted" / "latest.pkl")
    assert resumed["round_index"] == reference["round_index"] == 2
    assert resumed["feedback_hash"] == reference["feedback_hash"]
    np.testing.assert_array_equal(resumed["key_data"], reference["key_data"])
    for regime in ("easy", "hard"):
        np.testing.assert_array_equal(resumed[f"last_{regime}_validation_key_data"],
                                      reference[f"last_{regime}_validation_key_data"])
    for actual, expected in zip(jax.tree.leaves(resumed["student"]), jax.tree.leaves(reference["student"])):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
