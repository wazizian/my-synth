"""GPU campaign checks for the tensor credit model; no local numerical run."""
import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from credit_hedging.market import MarketConfig, build_tables
from credit_hedging.curve_market import CurveConfig, build_curve_tables, simulate_curve_paths, continuation
from credit_hedging.environment import rollout


def small_config(**kwargs):
    return CurveConfig(bond_days=12, cds_days=4, grid_size=17,
                       fast_grid_size=9, **kwargs)


def test_tensor_expectation_matches_explicit_joint_law():
    cfg = small_config()
    t = build_curve_tables(cfg)
    values = jnp.arange(cfg.grid_size*cfg.fast_grid_size*3, dtype=float).reshape(cfg.grid_size, cfg.fast_grid_size, 3)
    expected = np.kron(np.asarray(t['transition_x']), np.asarray(t['transition_y'])) @ np.asarray(values).reshape(-1, 3)
    np.testing.assert_allclose(continuation(values, t['transition_x'], t['transition_y']).reshape(-1, 3), expected, atol=1e-10, rtol=1e-12)


def test_constant_fast_factor_recovers_audited_one_factor():
    cfg = CurveConfig(bond_days=12, cds_days=4, grid_size=17,
                      fast_grid_size=1, fast_vol=0, fast_initial=0, fast_mean=0)
    one = MarketConfig(**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(MarketConfig)})
    reference, actual = build_tables(one), build_curve_tables(cfg)
    for name in ('bond', 'protection', 'annuity', 'survival', 'par_spread'):
        expected = reference[name]
        if name in ('protection', 'annuity'):
            expected = expected[:cfg.cds_days+1]
        np.testing.assert_allclose(actual[name], expected, atol=1e-12, rtol=1e-12)


def test_joint_pricing_recursion_and_par_upfront():
    cfg = small_config()
    t = build_curve_tables(cfg)
    joint = np.kron(np.asarray(t['transition_x']), np.asarray(t['transition_y']))
    s = np.asarray(t['survival'])
    discount = np.exp(-cfg.rate/cfg.trading_days)
    for day in range(1, cfg.bond_days+1):
        expected = discount*((1-s)*cfg.recovery+s*(joint@np.asarray(t['bond'][day-1])))
        np.testing.assert_allclose(t['bond'][day], expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(t['protection'][cfg.cds_days]-t['par_spread']*t['annuity'][cfg.cds_days], 0, atol=1e-12)


def test_absorption_and_unhedged_accounting():
    cfg = small_config()
    t = build_curve_tables(cfg)
    paths = simulate_curve_paths(jax.random.key(7), t, cfg, 128)
    result = rollout(None, paths, t, cfg)
    np.testing.assert_allclose(result['records']['daily_loss'].sum(axis=1), result['loss'], atol=1e-12)
    assert np.all(np.asarray(paths['default']).sum(axis=1) <= 1)
    assert np.all(np.asarray(result['records']['daily_loss'])[~np.asarray(paths['alive'][:, :-1])] == 0)


def test_observable_rules_obey_actions_and_cash_accounting():
    from credit_hedging.diagnostic_policies import FixedProtection, ObservableSensitivity
    cfg = CurveConfig(bond_days=12, cds_days=4, grid_size=31,
                      fast_grid_size=1, fast_vol=0)
    t = build_curve_tables(cfg)
    paths = simulate_curve_paths(jax.random.key(19), t, cfg, 32)
    assert np.all(np.diff(np.asarray(t['par_spread'])) > 0)
    for policy in [FixedProtection(1., False, 4), FixedProtection(1., True, 4), ObservableSensitivity(t, cfg)]:
        result = rollout(policy, paths, t, cfg, record_inventory=True)
        r = result['records']
        np.testing.assert_allclose(r['daily_loss'].sum(axis=1), result['loss'], atol=1e-12)
        assert np.all(np.asarray(r['notional']) >= 0)
        assert np.all(np.asarray(r['notional']) <= cfg.max_protection+1e-12)
        assert np.isfinite(np.asarray(result['loss'])).all()


def test_independent_full_calendar_reference_matches_production_tables():
    from credit_hedging.numerical_validation import sampled_prices
    cfg = CurveConfig(bond_days=12, cds_days=4, grid_size=13,
                      grid_min=-6.5, grid_max=-.5, fast_grid_size=21)
    actual = build_curve_tables(cfg)
    ref = sampled_prices(cfg)
    x, y = np.asarray(actual['x_grid']), np.asarray(actual['y_grid'])
    ix = np.abs(x[:,None]-np.asarray(ref['x'])).argmin(axis=0)
    iy = np.abs(y[:,None]-np.asarray(ref['y'])).argmin(axis=0)
    indices = ix[:,None]*len(y)+iy[None,:]
    for k,name in enumerate(('bond','protection','annuity')):
        days = cfg.bond_days if k==0 else cfg.cds_days
        np.testing.assert_allclose(actual[name][1:days+1,indices],ref['prices'][:days,k],atol=1e-12,rtol=1e-12)


def test_two_factor_mixed_coupon_book_is_self_financing_in_expectation():
    from credit_hedging.environment import book_step
    cfg = small_config(transaction_cost=0.)
    t = build_curve_tables(cfg)
    n = cfg.grid_size*cfg.fast_grid_size
    state = int(t['initial_state'])
    joint = np.kron(np.asarray(t['transition_x']), np.asarray(t['transition_y']))[state]
    survival = float(t['survival'][state])
    weights = np.concatenate((survival*joint,(1-survival)*joint))
    book = dict(notional=jnp.tile(jnp.array([0.,.2,.4,.1]),(2*n,1)),
                coupon=jnp.tile(jnp.array([0.,.02,.015,.01]),(2*n,1)))
    action = dict(buy=jnp.full(2*n,.3),retain=jnp.full((2*n,4),.5))
    market = dict(day=jnp.array(2),state=jnp.full(2*n,state,dtype=jnp.int32),
                  next_state=jnp.tile(jnp.arange(n),2),alive=jnp.ones(2*n,dtype=bool),
                  default=jnp.arange(2*n)>=n)
    _,records=book_step(book,action,market,t,cfg)
    np.testing.assert_allclose(weights@np.asarray(records['daily_loss']),0,atol=1e-12)


@pytest.mark.parametrize('kwargs', [dict(fast_vol=-1), dict(fast_reversion=.5), dict(fast_grid_size=1), dict(fast_vol=float('nan'))])
def test_invalid_fast_factor(kwargs):
    with pytest.raises(ValueError):
        CurveConfig(**kwargs)
