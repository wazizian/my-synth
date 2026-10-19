"""GPU-only credit bridge: exact per-path categorical scenario mixture.

Simulator states are used for market accounting only. No scenario identifier is
added to policy observations. Domains share all contract and trading conventions.
"""
from dataclasses import replace
import gc

import equinox as eqx
import jax
import jax.numpy as jnp

from credit_hedging.curve_market import build_curve_tables, simulate_curve_paths
from credit_hedging.environment import book_step
from credit_hedging.learned_run import configs

DOMAINS = ('easy', 'ordinary', 'hard')


class CreditAdapter(eqx.Module):
    configs: tuple = eqx.field(static=True)
    tables: dict
    quotes: dict
    initial_prices: object
    offsets: tuple = eqx.field(static=True)
    contract: object = eqx.field(static=True)

    def __init__(self, source, bp, audit_callback=None):
        original = configs(source)
        self.configs = tuple(replace(original[d], transaction_cost=bp / 10000) for d in DOMAINS)
        tables, quote_parts, initial_prices, offsets = {}, [], [], []
        offset = 0
        for d, cfg in zip(DOMAINS, self.configs):
            full = eqx.filter_jit(build_curve_tables)(cfg)
            jax.block_until_ready(full)
            if audit_callback is not None:
                audit_callback(d, cfg, full)
            # Training terminal cashflows need only the issuance bond price.
            # Do not retain the ~9GB hard full-maturity bond-mark tensor.
            initial = full['bond'][cfg.bond_days]
            jax.block_until_ready(initial)
            initial_prices.append(initial)
            quote_parts.append({n: full[n] for n in ('protection', 'annuity', 'par_spread')})
            tables[d] = {n: full[n] for n in ('transition_x', 'transition_y', 'survival', 'initial_state')}
            offsets.append(offset)
            offset += full['par_spread'].shape[0]
            del full
            jax.clear_caches()
            gc.collect()
        self.tables = tables
        self.quotes = {n: jnp.concatenate([t[n] for t in quote_parts], axis=-1)
                       for n in ('protection', 'annuity', 'par_spread')}
        self.initial_prices = jnp.concatenate(initial_prices)
        jax.block_until_ready((self.quotes, self.initial_prices))
        self.offsets = tuple(offsets)
        contract = self.configs[1]
        for cfg in self.configs:
            for name in ('bond_days', 'cds_days', 'trading_days', 'rate',
                         'recovery', 'max_protection', 'transaction_cost'):
                if getattr(cfg, name) != getattr(contract, name):
                    raise ValueError(f'Domain changes contract convention: {name}')
        self.contract = contract

    def losses(self, model, batch):
        return terminal_losses(model, batch, self.quotes, self.initial_prices, self.contract)

    def domain(self, key, domain, count):
        index = DOMAINS.index(domain)
        paths = simulate_curve_paths(key, self.tables[domain],
                                     self.configs[index], count)
        return dict(paths, state=paths['state'] + self.offsets[index])

    def draw(self, key, probabilities, block_size=8, batch_size=160):
        keys = jax.random.split(key, 4)
        count = block_size * batch_size
        ids = jax.random.categorical(keys[0], jnp.log(probabilities), shape=(count,))
        draws = [self.domain(keys[i + 1], d, count) for i, d in enumerate(DOMAINS)]
        # Independent candidate paths for every domain and row; selecting by a
        # categorical label therefore samples the exact declared mixture.
        paths = jax.tree.map(
            lambda a, b, c: jnp.where(ids[:, None] == 0, a,
                                     jnp.where(ids[:, None] == 1, b, c)),
            *draws,
        )
        return jax.tree.map(
            lambda x: x.reshape((block_size, batch_size) + x.shape[1:]), paths
        ), ids.reshape(block_size, batch_size)


def terminal_losses(policy, paths, quotes, initial_prices, config):
    """Exact original terminal ledger, excluding unused bond/daily mark tensors.

    book_step's dummy bond marks affect only discarded daily mark records;
    they do not enter actions, holdings, coupons or discounted_cashflow. The
    actual issuance price is supplied separately and subtracted exactly once.
    """
    count, n_dates = paths['state'].shape
    n_time, n_vintages = n_dates-1, config.cds_days
    book = dict(notional=jnp.zeros((count, n_vintages)), coupon=jnp.zeros((count, n_vintages)))
    hidden = jnp.zeros((count, policy.hidden_size))
    previous_spread = quotes['par_spread'][paths['state'][:, 0]]
    remaining = n_vintages-jnp.arange(n_vintages)
    # Original book_step retains mark records. These dummy marks are never used
    # by the terminal-only result; XLA can eliminate their entire calculation.
    ledger_quotes = dict(quotes, bond=jnp.zeros((1, 1)))

    def step(carry, day):
        book, hidden, previous_spread, cash = carry
        state = paths['state'][:, day]
        spread = quotes['par_spread'][state]
        marks = quotes['protection'][remaining[None, :], state[:, None]]-book['coupon']*quotes['annuity'][remaining[None, :], state[:, None]]
        marks = jnp.where(book['notional'] > 0, marks, 0)
        global_features = jnp.stack((100*spread, 100*(spread-previous_spread),
            jnp.full(count, (n_time-day)/n_time),
            jnp.sum(book['notional'], axis=1)/config.max_protection), axis=1)
        features = jnp.concatenate((global_features, book['notional']/config.max_protection,
                                   100*book['coupon'], 100*marks), axis=1)
        hidden, action = jax.vmap(policy)(features, hidden)
        market = dict(day=day, state=state, next_state=paths['state'][:, day+1],
                      alive=paths['alive'][:, day], default=paths['default'][:, day])
        book, records = book_step(book, action, market, ledger_quotes, config)
        return (book, hidden, spread, cash+records['discounted_cashflow']), None

    final, _ = jax.lax.scan(jax.checkpoint(step),
                           (book, hidden, previous_spread, jnp.zeros(count)),
                           jnp.arange(n_time))
    return initial_prices[paths['state'][:, 0]]-final[3]
