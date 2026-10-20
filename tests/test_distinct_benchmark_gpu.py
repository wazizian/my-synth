"""GPU-only equivalence checks against the archived distinct-market implementation."""
# ruff: noqa: E402
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

if os.environ.get("NEURAL_SDE_GPU_TESTS") != "1":
    pytest.skip("Explicit GPU validation opt-in required", allow_module_level=True)

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

if jax.default_backend() != "gpu":
    raise RuntimeError("Distinct benchmark numerical validation requires GPU")
jax.config.update("jax_enable_x64", True)

from credit_hedging.curve_market import CurveConfig, build_curve_tables, simulate_curve_paths
from credit_hedging.learned_policy import TargetHedger
from neural_sde_hedging import credit_benchmark, distinct, option_benchmark
from neural_sde_hedging import training as training_module
from neural_sde_hedging.config import SDEConfig
from neural_sde_hedging.training import atomic_checkpoint, load_checkpoint, run


def reference(name):
    """Load an immutable small archived reference module; return it without invoking any runner."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent/'references'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("domain", ["easy", "base", "hard"])
def test_option_market_and_hedger_match_archived_code(domain):
    """Compare P/Q states, quotes, IV, policy parameters and full ledger to the previous model."""
    old = reference('options_20261018')
    config = SDEConfig(n_steps=40)
    for measure in ("P", "Q"):
        key = jax.random.key(172)
        actual = option_benchmark.sample_states(key, 16, config, domain, measure)
        expected = old.simulate(key, domain, 16, measure=measure, days=config.n_steps, burn=0)
        np.testing.assert_allclose(actual[..., 0], expected['logs'].T, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual[..., 1:], np.swapaxes(expected['states'], 0, 1), rtol=1e-12, atol=1e-12)
    states = option_benchmark.sample_states(jax.random.key(177), 4, config, domain)
    factors = states[:, 37, 1:]
    spot = jnp.exp(states[:, 37, 0])
    key = jax.random.key(178)
    quote, se = option_benchmark.price(key, factors, spot, 1., domain, 3, 3, 32)
    expected = old.price(key, factors, spot, jnp.ones(4), domain, 3, 32)
    np.testing.assert_allclose(quote, expected, rtol=1e-12, atol=1e-12)
    assert np.all(np.asarray(se) >= 0)
    iv, _ = option_benchmark.atm_iv(key, factors, domain, 3, 3, 32)
    old_iv, _ = old.atm_iv(key, factors, domain, 3, 32)
    np.testing.assert_allclose(iv, old_iv, rtol=1e-10, atol=1e-10)
    policy, previous = option_benchmark.Hedger(jax.random.key(20261017)), old.Hedger(jax.random.key(20261017))
    for actual, expected in zip(jax.tree.leaves(policy), jax.tree.leaves(previous)):
        np.testing.assert_array_equal(actual, expected)
    actual = option_benchmark.rollout(policy, states[..., 0], .0003)
    # Reference rollout uses its global horizon, so set it to this small fixture's horizon.
    old.HORIZON = config.n_steps
    expected = old.rollout(previous, states[..., 0].T, .0003)
    np.testing.assert_allclose(actual['loss'], expected['loss'], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual['holdings'], expected['holdings'].T, rtol=1e-12, atol=1e-12)
    assert float(option_benchmark.variance(jnp.array([[100., 100.]]))[0]) == 1.


def test_credit_tables_sampler_and_policy_match_archived_code():
    """Verify original two-factor grid pricing, default sampling and target-retention initialization."""
    old = reference('credit_curve_20261018')
    previous_policy = reference('credit_policy_20261018')
    values = dict(bond_days=4, cds_days=2, grid_size=9, fast_grid_size=7,
                  mean_reversion=.25, latent_vol=.25, fast_vol=1.224744871391589,
                  grid_min=-6.5, grid_max=-.5)
    config, before = CurveConfig(**values), old.CurveConfig(**values)
    tables, previous = build_curve_tables(config), old.build_curve_tables(before)
    for key in tables:
        np.testing.assert_allclose(tables[key], previous[key], rtol=1e-12, atol=1e-12)
    paths = simulate_curve_paths(jax.random.key(814), tables, config, 16)
    archived = old.simulate_curve_paths(jax.random.key(814), previous, before, 16)
    for key in paths:
        np.testing.assert_array_equal(paths[key], archived[key])
    policy, original = TargetHedger(jax.random.key(20261017), n_vintages=2), previous_policy.TargetHedger(jax.random.key(20261017), n_vintages=2)
    for actual, expected in zip(jax.tree.leaves(policy), jax.tree.leaves(original)):
        np.testing.assert_array_equal(actual, expected)
    hidden, action = policy(jnp.zeros(10), jnp.zeros(32))
    np.testing.assert_array_equal(action['retain'], np.ones(2))
    assert float(action['buy']) == 1.
    assert np.all(np.isfinite(hidden))


def test_two_factor_grid_neural_score_codec_and_prior():
    """Compare zero-correction proposal rows with actual hard grid rows and round-trip a generator."""
    cfg = CurveConfig(bond_days=3, cds_days=2, grid_size=11, fast_grid_size=7,
                      grid_min=-6.5, grid_max=-.5, mean_reversion=.25, latent_vol=.25,
                      fast_vol=1.224744871391589)
    config = credit_benchmark.CreditGridConfig(cfg)
    model = credit_benchmark.NeuralCredit(jax.random.key(92), config, width=4, depth=1)
    tables = build_curve_tables(cfg)
    indices = jnp.array([0, 12, 42, 76])
    px, py = credit_benchmark.transition_rows(model, indices, config)
    np.testing.assert_allclose(px, tables['transition_x'][indices//cfg.fast_grid_size], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(py, tables['transition_y'][indices%cfg.fast_grid_size], rtol=1e-12, atol=1e-14)
    paths = credit_benchmark.simulate(model, credit_benchmark.sample_noise(jax.random.key(93), 8, config), config)
    np.testing.assert_allclose(paths.log_density, credit_benchmark.path_log_prob(model, paths, config), atol=1e-12)
    params, static = eqx.partition(model, eqx.is_inexact_array)
    gradient = jax.grad(lambda p: jnp.mean(credit_benchmark.path_log_prob(eqx.combine(p, static), paths, config)))(params)
    assert all(np.all(np.isfinite(leaf)) for leaf in jax.tree.leaves(gradient))
    assert any(np.any(np.asarray(leaf) != 0) for leaf in jax.tree.leaves(gradient))


def test_two_factor_grid_codec(tmp_path):
    """Load a complete archived-style neural grid generator without pickling activations."""
    config = credit_benchmark.CreditGridConfig(CurveConfig(bond_days=2, cds_days=1, grid_size=5, fast_grid_size=5))
    teacher = credit_benchmark.NeuralCredit(jax.random.key(942), config, width=4, depth=1)
    atomic_checkpoint(tmp_path/'teacher.pkl', teacher)
    loaded = load_checkpoint(tmp_path/'teacher.pkl')
    for before, after in zip(jax.tree.leaves(teacher), jax.tree.leaves(loaded)):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("problem", ["options", "credit"])
@pytest.mark.parametrize("method", ["baseline", "easy-specialist", "hard-specialist", "dpg", "soar", "selfplay"])
def test_restored_distinct_training_and_fresh_evaluation(tmp_path, monkeypatch, problem, method):
    """Exercise exact-market arms and adaptive callbacks on small explicit GPU smoke grids."""
    monkeypatch.setenv('NEURAL_SDE_ENABLED', '1')
    args = SimpleNamespace(method=method, market='distinct', problem=problem, target_regime=None,
        hard_pool_size=8, credit_simulation='continuous', cds_days=2, credit_grid_size=9,
        credit_fast_grid_size=7, option_cost=.0003, option_quote_paths=4, option_daily_paths=2,
        output=str(tmp_path/method), seconds=600., rounds=2, seed=20261017, groups=2,
        items=2, inner_steps=2, n_steps=3, gamma=10., student_lr=3e-4, teacher_lr=1e-3,
        warmup_steps=0, weight_decay=0., promotion_threshold=1e-4, feedback_paths=None,
        feedback_size=4, validation_paths=None, validation_size=4, validation_every=1,
        entropic_ess_fraction=.1, payoff_cap=None, premium=0., min_free_gb=.01)
    run(args)
    state = load_checkpoint(tmp_path/method/'latest.pkl')
    assert state['round_index'] == 2
    rows = [json.loads(row) for row in (tmp_path/method/'progress.jsonl').read_text().splitlines()]
    assert state['feedback_hash'].endswith('distinct-v1')
    for row in rows:
        for domain in distinct.DOMAINS:
            assert row[f'evaluation_{domain}_sample_size'] == 4
            assert np.isfinite(row[f'evaluation_{domain}_daily_error_std'])
        if method in ('baseline', 'easy-specialist', 'hard-specialist'):
            assert row['teacher_gradient_norm'] == 0.
    # Completed-run resume must preserve parameters and every RNG stream.
    run(args)
    resumed = load_checkpoint(tmp_path/method/'latest.pkl')
    np.testing.assert_array_equal(state['key_data'], resumed['key_data'])
    for before, after in zip(jax.tree.leaves(state['student']), jax.tree.leaves(resumed['student'])):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("method,domain", [("baseline", "base"), ("easy-specialist", "easy"), ("hard-specialist", "hard")])
def test_fresh_per_optimizer_update_matches_direct_reference(tmp_path, monkeypatch, method, domain):
    """Compare the actual learner with two independently drawn, unbootstrapped own-domain updates."""
    monkeypatch.delenv('NEURAL_SDE_ENABLED', raising=False)
    args = SimpleNamespace(method=method, market='distinct', problem='options', target_regime=None,
        hard_pool_size=8, credit_simulation='continuous', cds_days=2, credit_grid_size=9,
        credit_fast_grid_size=7, option_cost=.0003, option_quote_paths=4, option_daily_paths=0,
        output=str(tmp_path/method), seconds=600., rounds=1, seed=20261017, groups=2,
        items=2, inner_steps=2, n_steps=3, gamma=10., student_lr=3e-4, teacher_lr=1e-3,
        warmup_steps=0, weight_decay=0., promotion_threshold=1e-4, feedback_paths=None,
        feedback_size=4, validation_paths=None, validation_size=4, validation_every=1,
        entropic_ess_fraction=.1, payoff_cap=None, premium=0., min_free_gb=.01)
    run(args)
    state = load_checkpoint(tmp_path/method/'latest.pkl')
    model = option_benchmark.Hedger(jax.random.key(args.seed))
    params, static = eqx.partition(model, eqx.is_inexact_array)
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(args.student_lr, eps_root=1e-16))
    moments = optimizer.init(params)
    inner_key = jax.random.split(jax.random.key(args.seed+2), 4)[2]
    seen = []
    for subkey in jax.random.split(inner_key, 2):
        draw, _ = jax.random.split(subkey)
        states = option_benchmark.sample_states(draw, 4, SDEConfig(n_steps=3), domain)
        seen.append(np.asarray(states))

        def objective(candidate):
            """Return original full-batch entropic objective, with no bootstrap or fixed-pool reuse."""
            losses = option_benchmark.rollout(eqx.combine(candidate, static), states[..., 0], args.option_cost)['loss']
            return (jax.scipy.special.logsumexp(args.gamma*losses)-jnp.log(4))/args.gamma

        _, gradient = jax.value_and_grad(objective)(params)
        updates, moments = optimizer.update(gradient, moments, params)
        params = optax.apply_updates(params, updates)
    assert not np.array_equal(seen[0], seen[1])
    for actual, expected in zip(jax.tree.leaves(state['student']), jax.tree.leaves((params, moments))):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize("problem", ["options", "credit"])
def test_distinct_interrupted_resume_reproduces_learner(tmp_path, monkeypatch, problem):
    """Rebuild actual pricing tables and fixed hard feedback after interruption without changing updates."""
    args = SimpleNamespace(method='hard-specialist', market='distinct', problem=problem, target_regime=None,
        hard_pool_size=8, credit_simulation='continuous', cds_days=2, credit_grid_size=9,
        credit_fast_grid_size=7, option_cost=.0003, option_quote_paths=4, option_daily_paths=0,
        output=str(tmp_path/'interrupted'), seconds=600., rounds=2, seed=20261017, groups=2,
        items=2, inner_steps=2, n_steps=3, gamma=10., student_lr=3e-4, teacher_lr=1e-3,
        warmup_steps=0, weight_decay=0., promotion_threshold=1e-4, feedback_paths=None,
        feedback_size=4, validation_paths=None, validation_size=4, validation_every=1,
        entropic_ess_fraction=.1, payoff_cap=None, premium=0., min_free_gb=.01)
    original_save = training_module.atomic_checkpoint

    def save_then_stop(path, payload):
        """Commit round one, then interrupt before the next optimizer block."""
        original_save(path, payload)
        if path.name == 'latest.pkl' and payload.get('round_index') == 1:
            raise RuntimeError('simulated interruption')

    monkeypatch.setattr(training_module, 'atomic_checkpoint', save_then_stop)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        run(args)
    monkeypatch.setattr(training_module, 'atomic_checkpoint', original_save)
    run(args)
    resumed = load_checkpoint(tmp_path/'interrupted'/'latest.pkl')
    args.output = str(tmp_path/'reference')
    run(args)
    expected = load_checkpoint(tmp_path/'reference'/'latest.pkl')
    assert resumed['round_index'] == expected['round_index'] == 2
    np.testing.assert_array_equal(resumed['key_data'], expected['key_data'])
    for actual, reference_value in zip(jax.tree.leaves(resumed['student']), jax.tree.leaves(expected['student'])):
        np.testing.assert_allclose(actual, reference_value, rtol=2e-6, atol=2e-8)
