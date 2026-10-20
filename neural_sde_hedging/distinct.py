"""Faithful distinct-market adapters, with tables passed as dynamic JIT arguments."""
from dataclasses import replace
import json
from pathlib import Path

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from credit_hedging.curve_market import CurveConfig, build_curve_tables, simulate_curve_paths
from credit_hedging.environment import rollout as credit_rollout
from . import option_benchmark as options
from .metrics import risk_metrics

DOMAINS = ("easy", "base", "hard")


class Market(eqx.Module):
    """Dynamic price tables/premiums plus immutable per-domain benchmark configurations."""
    tables: tuple
    premiums: jax.Array
    premium_se: jax.Array
    configs: tuple = eqx.field(static=True)
    problem: str = eqx.field(static=True)
    cost: float = eqx.field(static=True)
    quote_paths: int = eqx.field(static=True)
    daily_paths: int = eqx.field(static=True)


def build_market(args, option_config):
    """Build historical production markets; grid overrides are explicit smoke approximations."""
    if args.problem == "options":
        compiled_price = eqx.filter_jit(options.price)
        premiums, errors = [], []
        for index, domain in enumerate(DOMAINS):
            value, error = compiled_price(jax.random.key(args.seed+500+index), jnp.zeros((1, 2)),
                                         jnp.ones(1), 1., domain, option_config.n_steps,
                                         option_config.n_steps, args.option_quote_paths, option_config.dt)
            premiums.append(value[0])
            errors.append(error[0])
        return Market((), jnp.stack(premiums), jnp.stack(errors), (option_config,),
                      "options", args.option_cost, args.option_quote_paths, args.option_daily_paths)
    source = Path(__file__).parent.parent/"credit_hedging"/"configs"/"distinct_regimes.json"
    saved = json.loads(source.read_text())
    configs, tables = [], []
    compiled_tables = eqx.filter_jit(build_curve_tables)
    for domain in DOMAINS:
        config = CurveConfig(**saved[domain])
        config = replace(config, bond_days=args.n_steps, cds_days=args.cds_days)
        if args.credit_grid_size is not None:
            config = replace(config, grid_size=args.credit_grid_size)
        if args.credit_fast_grid_size is not None and config.fast_grid_size > 1:
            config = replace(config, fast_grid_size=args.credit_fast_grid_size)
        configs.append(config)
        tables.append(compiled_tables(config))
    return Market(tuple(tables), jnp.zeros(3), jnp.zeros(3), tuple(configs),
                  "credit", 0., args.option_quote_paths, args.option_daily_paths)


def sample_states(key, batch, config, regime, market):
    """Draw fresh historical regime paths [B,T+1,3]; credit packs state/alive/default."""
    if market.problem == "options":
        return options.sample_states(key, batch, config, regime)
    index = DOMAINS.index(regime)
    paths = simulate_curve_paths(key, market.tables[index], market.configs[index], batch)
    default = jnp.concatenate((jnp.zeros((batch, 1)), paths['default']), axis=1)
    chex.assert_shape(default, (batch, config.n_steps+1))
    packed = jnp.stack((paths['state'], paths['alive'], default), axis=2)
    chex.assert_shape(packed, (batch, config.n_steps+1, 3))
    return packed


def unpack_states(states):
    """Decode credit packed [B,T+1,3] into original grid/default ledger paths."""
    batch, n_dates, dim = states.shape
    chex.assert_shape(states, (batch, n_dates, 3))
    return {"state": states[..., 0].astype(jnp.int32), "alive": states[..., 1] > .5,
            "default": states[:, 1:, 2] > .5}


def losses(policy, states, regime, market):
    """Return actual domain-priced terminal losses [B], without exposing regime or factors to policy."""
    index = DOMAINS.index(regime)
    if market.problem == "options":
        return options.rollout(policy, states[..., 0], market.cost, market.premiums[index])["loss"]
    return credit_rollout(policy, unpack_states(states), market.tables[index], market.configs[index])["loss"]


def metrics(policy, states, regime, market, gamma, daily=True):
    """Return terminal risks and daily marked-error SD using actual domain quotes and ledger."""
    index = DOMAINS.index(regime)
    if market.problem == "credit":
        result = credit_rollout(policy, unpack_states(states), market.tables[index], market.configs[index])
        loss = result['loss']
        return loss, risk_metrics(loss, gamma, daily_marked_errors=result['records']['daily_loss'])
    result = options.rollout(policy, states[..., 0], market.cost, market.premiums[index])
    loss = result['loss']
    output = risk_metrics(loss, gamma)
    output['initial_q_premium'] = market.premiums[index]
    output['initial_q_premium_mc_se'] = market.premium_se[index]
    if daily and market.daily_paths:
        count = min(states.shape[0], market.daily_paths)
        selected = states[:count]
        n_steps = selected.shape[1]-1
        config = market.configs[0]

        def mark_day(_, day):
            """Price the current-state European claim under Q, including future random arrivals."""
            marks, errors = options.price(jax.random.fold_in(jax.random.key(848), day),
                selected[:, day, 1:], jnp.exp(selected[:, day, 0]), 1., regime,
                n_steps-day, n_steps, market.quote_paths, config.dt)
            chex.assert_shape((marks, errors), (count,))
            return None, (marks, errors)

        _, (marks, errors) = jax.lax.scan(mark_day, None, jnp.arange(n_steps))
        terminal = jax.nn.relu(jnp.exp(selected[:, -1, 0])-1.)
        marks = jnp.concatenate((marks.T, terminal[:, None]), axis=1)
        chex.assert_shape(marks, (count, n_steps+1))
        ledger = options.rollout(policy, selected[..., 0], market.cost)
        daily = jnp.diff(marks, axis=1)-ledger['gains']+ledger['daily_cost']
        chex.assert_shape(daily, (count, n_steps))
        output['daily_error_std'] = jnp.std(daily)
        output['daily_error_sample_size'] = jnp.asarray(count)
        output['daily_quote_paths'] = jnp.asarray(market.quote_paths)
        output['daily_quote_max_mc_se'] = jnp.max(errors)
        expected = ledger['loss']-marks[:, 0]
        output['daily_accounting_max_error'] = jnp.max(jnp.abs(jnp.sum(daily, axis=1)-expected))
    return loss, output
