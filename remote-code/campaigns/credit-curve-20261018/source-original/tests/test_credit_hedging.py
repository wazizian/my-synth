"""Accounting, vintage lifecycle, pricing, causality and differentiation checks."""
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from credit_hedging.environment import book_step, rollout
from credit_hedging.market import MarketConfig, build_tables, simulate_paths
from credit_hedging.metrics import entropic_risk, evaluate_metrics
from credit_hedging.policy import Hedger
from credit_hedging.train import train_chunk

jax.config.update('jax_enable_x64', True)


@pytest.fixture(scope='module')
def setup():
    """Return a small market config/table pair for fast deterministic lifecycle checks."""
    config = MarketConfig(bond_days=8, cds_days=3, grid_size=31, grid_min=-5, grid_max=-2)
    return config, build_tables(config)


def test_constant_hazard_prices_have_closed_form():
    """Check bond and CDS legs against constant-intensity geometric-series prices."""
    config = MarketConfig(bond_days=12, cds_days=4, grid_size=3, grid_min=-4,
                          grid_max=-3, latent_vol=1e-8)
    tables = build_tables(config)
    # At the central mean the grid chain stays put, leaving a constant intensity.
    idx = 1
    q = float(tables['survival'][idx])
    d = np.exp(-config.rate/config.trading_days)
    for n in [1, 4, 12]:
        geometric = (1-(d*q)**n)/(1-d*q)
        expected_bond = (d*q)**n + d*config.recovery*(1-q)*geometric
        expected_protection = d*(1-config.recovery)*(1-q)*geometric
        expected_annuity = d/config.trading_days*geometric
        np.testing.assert_allclose(tables['bond'][n,idx], expected_bond, atol=1e-12)
        np.testing.assert_allclose(tables['protection'][n,idx], expected_protection, atol=1e-12)
        np.testing.assert_allclose(tables['annuity'][n,idx], expected_annuity, atol=1e-12)
    np.testing.assert_allclose(tables['protection'][4]-tables['par_spread']*tables['annuity'][4], 0, atol=1e-14)


def test_vintage_coupon_retained_sale_realizes_market_value(setup):
    """Buy, age, then sell half after a spread change; compare coupon and cashflow independently."""
    c, t = setup
    idx = int(t['initial_state'])
    book = dict(notional=jnp.zeros((1,3)), coupon=jnp.zeros((1,3)))
    market = dict(day=jnp.array(0), state=jnp.array([idx]), next_state=jnp.array([idx]),
                  alive=jnp.array([True]), default=jnp.array([False]))
    action = dict(buy=jnp.array([1.0]), retain=jnp.ones((1,3)))
    book, r = book_step(book, action, market, t, c)
    coupon = float(t['par_spread'][idx])
    np.testing.assert_allclose(book['notional'], [[0,1,0]])
    np.testing.assert_allclose(book['coupon'], [[0,coupon,0]])
    np.testing.assert_allclose(r['premium'], [coupon/c.trading_days])
    np.testing.assert_allclose(r['discounted_cost'], [c.transaction_cost])
    higher = idx+3
    market = dict(market, day=jnp.array(1), state=jnp.array([higher]), next_state=jnp.array([higher]))
    action = dict(buy=jnp.array([0.0]), retain=jnp.array([[1.0,.5,1.0]]))
    book, r = book_step(book, action, market, t, c)
    value = float(t['protection'][2,higher]-coupon*t['annuity'][2,higher])
    np.testing.assert_allclose(r['sale_cash'], [.5*value])
    np.testing.assert_allclose(book['notional'], [[0,0,.5]])
    np.testing.assert_allclose(book['coupon'], [[0,0,coupon]])
    np.testing.assert_allclose(r['premium'], [.5*coupon/c.trading_days])


def test_expiry_default_and_long_only_constraints(setup):
    """Check natural expiry, default recovery/protection and clipping of infeasible orders."""
    c,t = setup
    idx = int(t['initial_state'])
    market = dict(day=jnp.array(2), state=jnp.array([idx]), next_state=jnp.array([idx]),
                  alive=jnp.array([True]), default=jnp.array([False]))
    book = dict(notional=jnp.array([[0.,0.,1.]]), coupon=jnp.array([[0.,0.,.02]]))
    action = dict(buy=jnp.array([-1.]), retain=jnp.full((1,3),2.))
    nxt,r = book_step(book, action, market, t, c)
    np.testing.assert_allclose(nxt['notional'], 0)
    np.testing.assert_allclose(r['discounted_cost'], 0)
    np.testing.assert_allclose(r['premium'], [.02/c.trading_days])
    action = dict(buy=jnp.array([10.]), retain=jnp.ones((1,3)))
    nxt,r = book_step(book, action, dict(market, default=jnp.array([True])), t, c)
    np.testing.assert_allclose(r['buy'], [1.])  # old protection uses half the cap
    np.testing.assert_allclose(r['protection_cash'], [2*(1-c.recovery)])
    np.testing.assert_allclose(r['bond_cash'], [c.recovery])
    np.testing.assert_allclose(nxt['notional'], 0)
    nxt,after = book_step(nxt, action, dict(market, day=jnp.array(3), alive=jnp.array([False])), t, c)
    np.testing.assert_allclose(after['discounted_cashflow'], 0)
    np.testing.assert_allclose(after['daily_loss'], 0)


def test_bond_maturity_liquidates_unexpired_cds(setup):
    """Verify surviving protection is sold at market at bond maturity, with a close cost."""
    c,t = setup
    idx = int(t['initial_state'])
    market = dict(day=jnp.array(c.bond_days-1), state=jnp.array([idx]), next_state=jnp.array([idx+2]),
                  alive=jnp.array([True]), default=jnp.array([False]))
    book = dict(notional=jnp.zeros((1,3)), coupon=jnp.zeros((1,3)))
    action = dict(buy=jnp.array([1.]), retain=jnp.ones((1,3)))
    nxt,r = book_step(book, action, market, t, c)
    coupon = float(t['par_spread'][idx])
    value = float(t['protection'][2,idx+2]-coupon*t['annuity'][2,idx+2])
    np.testing.assert_allclose(r['final_sale'], [value])
    np.testing.assert_allclose(r['bond_cash'], [1.])
    np.testing.assert_allclose(nxt['notional'], 0)
    expected_cost = c.transaction_cost*(np.exp(-c.rate*(c.bond_days-1)/c.trading_days)+np.exp(-c.rate*c.bond_days/c.trading_days))
    np.testing.assert_allclose(r['discounted_cost'], [expected_cost])


def test_discounted_accounting_and_unhedged_payoff(setup):
    """Check telescoping and independent long-bond loss on known default/no-default paths."""
    c,t = setup
    idx = int(t['initial_state'])
    states = jnp.full((2,c.bond_days+1),idx)
    alive = jnp.ones((2,c.bond_days+1),dtype=bool).at[1,3:].set(False)
    defaults = jnp.zeros((2,c.bond_days),dtype=bool).at[1,2].set(True)
    paths = dict(state=states,alive=alive,default=defaults)
    result = rollout(None,paths,t,c)
    expected = float(t['bond'][c.bond_days,idx])-np.array([
        np.exp(-c.rate*c.bond_days/c.trading_days),c.recovery*np.exp(-c.rate*3/c.trading_days)])
    np.testing.assert_allclose(result['loss'],expected,atol=1e-12)
    policy = Hedger(jax.random.key(0),c.cds_days,8,c.max_protection)
    result = eqx.filter_jit(rollout)(policy,paths,t,c,True)
    np.testing.assert_allclose(jnp.sum(result['records']['daily_loss'],axis=1),result['loss'],atol=1e-12)
    assert float(jnp.max(result['records']['notional'])) <= c.max_protection
    np.testing.assert_allclose(result['records']['inventory'][:,-1],0)
    assert float(evaluate_metrics(result)['max_accounting_error']) < 1e-12


def test_cash_translation_and_fractional_cvar():
    """Check entropic translation and fractional empirical tail mass against explicit numbers."""
    loss=jnp.array([0.,1.,2.,3.])
    np.testing.assert_allclose(entropic_risk(loss-.3),entropic_risk(loss)-.3,atol=1e-12)
    daily=loss[:,None]
    result=dict(loss=loss,defaulted=jnp.zeros(4,dtype=bool),
                records=dict(daily_loss=daily,discounted_cost=jnp.zeros_like(daily)))
    metrics=evaluate_metrics(result,alpha=.625)
    np.testing.assert_allclose(metrics['cvar'],(3+.5*2)/1.5)
    np.testing.assert_allclose(metrics['daily_error_std'],np.std(np.asarray(loss)))


def test_simulator_default_absorbing_and_pricing_martingale(setup):
    """Check simulated default is absorbing and bond discounted expected gains vanish under its law."""
    c,t=setup
    paths=simulate_paths(jax.random.key(5),t,c,16)
    np.testing.assert_array_equal(paths['alive'][:,1:],paths['alive'][:,:-1]&~paths['default'])
    assert np.all(np.asarray(paths['default']).sum(axis=1)<=1)
    n=5
    q=np.asarray(t['survival'])
    continuation=np.asarray(t['transition'])@np.asarray(t['bond'][n-1])
    expected=np.exp(-c.rate/c.trading_days)*(q*continuation+(1-q)*c.recovery)
    np.testing.assert_allclose(t['bond'][n],expected,atol=1e-12)


def test_policy_is_causal(setup):
    """Perturb future market states/defaults; past executed actions must remain unchanged."""
    c,t=setup
    paths=simulate_paths(jax.random.key(5),t,c,2)
    # Supply two deterministic, alive paths to isolate dependence on future quotes.
    paths=dict(state=jnp.full_like(paths['state'],t['initial_state']),
               alive=jnp.ones_like(paths['alive']),default=jnp.zeros_like(paths['default']))
    changed=dict(paths,state=paths['state'].at[:,4:].set(c.grid_size-1))
    model=Hedger(jax.random.key(4),c.cds_days,8)
    first=rollout(model,paths,t,c)
    second=rollout(model,changed,t,c)
    for name in ['buy','sold','notional']:
        np.testing.assert_allclose(first['records'][name][:,:4],second['records'][name][:,:4],atol=1e-12)


def test_finite_gradient_and_exact_checkpoint_resume(setup,tmp_path):
    """Compare continuous two-update training with one update, serialize/restore, then next update."""
    c,t=setup
    model=Hedger(jax.random.key(7),c.cds_days,8)
    opt=optax.chain(optax.clip_by_global_norm(1.),optax.adam(1e-3))
    state=opt.init(eqx.filter(model,eqx.is_inexact_array))
    key=jax.random.key(8)
    full,full_state,full_key,logs=train_chunk(model,state,key,t,c,opt,2,4,10.)
    first,first_state,first_key,_=train_chunk(model,state,key,t,c,opt,1,4,10.)
    skeleton=(first,first_state,jax.random.key_data(first_key))
    eqx.tree_serialise_leaves(tmp_path/'checkpoint.eqx',skeleton)
    restored,restored_state,key_data=eqx.tree_deserialise_leaves(tmp_path/'checkpoint.eqx',skeleton)
    resumed,resumed_state,resumed_key,_=train_chunk(restored,restored_state,jax.random.wrap_key_data(key_data),t,c,opt,1,4,10.)
    assert np.isfinite(np.asarray(logs['gradient_norm'])).all()
    assert np.any(np.asarray(logs['gradient_norm'])>0)
    for a,b in zip(jax.tree.leaves((full,full_state)),jax.tree.leaves((resumed,resumed_state))):
        if eqx.is_array(a): np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(jax.random.key_data(full_key),jax.random.key_data(resumed_key))


def test_bad_config_rejected():
    """Reject impossible recovery, invalid calendar and negative transaction cost inputs."""
    for kwargs in [dict(recovery=1.2),dict(cds_days=0),dict(transaction_cost=-.1)]:
        with pytest.raises(ValueError): MarketConfig(**kwargs)
