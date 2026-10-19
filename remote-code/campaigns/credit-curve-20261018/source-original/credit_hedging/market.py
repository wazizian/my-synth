"""Positive Vasicek-style intensity and consistently priced daily credit cashflows.

The latent OU process is approximated by a finite-state Gaussian transition
matrix. Simulation AND pricing use that matrix, so prices are exact for the
discrete research model, not an inconsistent continuous/discrete hybrid.
"""
from dataclasses import dataclass
import math
from numbers import Integral

import chex
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr


@dataclass(frozen=True)
class MarketConfig:
    """Static simulation/pricing inputs; daily counts determine array dimensions."""

    trading_days: int = 252
    bond_days: int = 1260
    cds_days: int = 126
    rate: float = 0.02
    recovery: float = 0.4
    mean_reversion: float = 1.0
    latent_mean: float = -3.5
    latent_vol: float = 0.5
    initial_latent: float = -3.5
    grid_min: float = -7.0
    grid_max: float = 0.0
    grid_size: int = 301
    transaction_cost: float = 0.0001
    max_protection: float = 2.0

    def __post_init__(self):
        """Validate scalar configuration inputs; raise ValueError on invalid conventions."""
        for name in ('trading_days', 'bond_days', 'cds_days', 'grid_size'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f'{name} must be an integer')
        for name in ('rate', 'recovery', 'mean_reversion', 'latent_mean', 'latent_vol',
                     'initial_latent', 'grid_min', 'grid_max', 'transaction_cost', 'max_protection'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        if min(self.trading_days, self.bond_days, self.cds_days) < 1:
            raise ValueError("Calendar and contract lengths must be positive")
        if self.grid_size < 3 or not self.grid_min < self.grid_max:
            raise ValueError("Need an ordered grid with at least three states")
        if not 0 <= self.recovery < 1:
            raise ValueError("Recovery must lie in [0, 1)")
        if self.mean_reversion <= 0 or self.latent_vol <= 0:
            raise ValueError("OU mean reversion and volatility must be positive")
        if not self.grid_min <= self.initial_latent <= self.grid_max:
            raise ValueError("Initial latent state must lie inside the grid")
        if self.transaction_cost < 0 or self.max_protection <= 0:
            raise ValueError("Invalid cost or position bound")


def build_tables(config: MarketConfig):
    """Map static config to a PyTree of transition, bond and CDS-leg price arrays.

    Hazard = softplus(latent OU); it is never a negative Gaussian intensity.
    Default occurs at the END of an interval, with recovery of bond face value.
    A CDS pays its full daily coupon on that interval, including default day.
    Returns prices indexed by [remaining_days, latent_state].
    """
    n_states = config.grid_size
    n_maturities = max(config.bond_days, config.cds_days)
    dt = 1.0 / config.trading_days
    grid = jnp.linspace(config.grid_min, config.grid_max, n_states)
    chex.assert_shape(grid, (n_states,))
    hazard = jax.nn.softplus(grid)
    chex.assert_shape(hazard, (n_states,))
    survival = jnp.exp(-hazard * dt)
    chex.assert_shape(survival, (n_states,))
    decay = jnp.exp(-config.mean_reversion * dt)
    chex.assert_shape(decay, ())
    means = config.latent_mean + decay * (grid - config.latent_mean)
    chex.assert_shape(means, (n_states,))
    # expm1 avoids cancellation and a spuriously frozen process for small kappa*dt.
    variance = config.latent_vol**2 * (-jnp.expm1(-2*config.mean_reversion*dt)) / (2 * config.mean_reversion)
    chex.assert_shape(variance, ())
    midpoints = (grid[:-1] + grid[1:]) / 2
    chex.assert_shape(midpoints, (n_states - 1,))
    edges = jnp.concatenate((jnp.array([-jnp.inf]), midpoints, jnp.array([jnp.inf])))
    chex.assert_shape(edges, (n_states + 1,))
    standardized = (edges[None, :] - means[:, None]) / jnp.sqrt(variance)
    chex.assert_shape(standardized, (n_states, n_states + 1))
    cdf = ndtr(standardized)
    chex.assert_shape(cdf, (n_states, n_states + 1))
    transition = jnp.maximum(jnp.diff(cdf, axis=1), 0)
    chex.assert_shape(transition, (n_states, n_states))
    row_sum = jnp.sum(transition, axis=1, keepdims=True)
    chex.assert_shape(row_sum, (n_states, 1))
    transition = transition / row_sum
    chex.assert_shape(transition, (n_states, n_states))
    discount = jnp.exp(-config.rate * dt)
    chex.assert_shape(discount, ())
    # Columns: bond, CDS protection leg, unit annual-coupon premium annuity.
    terminal = jnp.stack((jnp.ones(n_states), jnp.zeros(n_states), jnp.zeros(n_states)), axis=1)
    chex.assert_shape(terminal, (n_states, 3))
    default_cash = jnp.stack(((1-survival)*config.recovery,
                             (1-survival)*(1-config.recovery),
                             jnp.full(n_states, dt)), axis=1)
    chex.assert_shape(default_cash, (n_states, 3))

    def price_step(previous, unused):
        """Advance price maturity by one day; inputs previous legs/unused; returns legs twice."""
        continuation = transition @ previous
        chex.assert_shape(continuation, (n_states, 3))
        current = discount * (default_cash + survival[:, None] * continuation)
        chex.assert_shape(current, (n_states, 3))
        return current, current

    _, prices = jax.lax.scan(price_step, terminal, None, length=n_maturities)
    chex.assert_shape(prices, (n_maturities, n_states, 3))
    prices = jnp.concatenate((terminal[None, :, :], prices), axis=0)
    chex.assert_shape(prices, (n_maturities + 1, n_states, 3))
    bond, protection, annuity = prices[:, :, 0], prices[:, :, 1], prices[:, :, 2]
    chex.assert_shape((bond, protection, annuity), (n_maturities + 1, n_states))
    par_spread = protection[config.cds_days] / annuity[config.cds_days]
    chex.assert_shape(par_spread, (n_states,))
    initial_state = jnp.argmin(jnp.abs(grid - config.initial_latent))
    chex.assert_shape(initial_state, ())
    return dict(grid=grid, hazard=hazard, survival=survival, transition=transition,
                bond=bond, protection=protection, annuity=annuity,
                par_spread=par_spread, initial_state=initial_state)


def simulate_paths(key, tables, config: MarketConfig, batch_size: int):
    """Generate a PyTree of daily states/alive flags/defaults for a batch of bond lives.

    Inputs: PRNG key, pricing tables, static config and batch size. Outputs:
    state/alive [batch, bond_days+1], default [batch, bond_days]. States and
    default are sampled under the SAME finite-state law used by build_tables.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    batch, n_time, n_states = batch_size, config.bond_days, config.grid_size
    state = jnp.full((batch,), tables['initial_state'], dtype=jnp.int32)
    chex.assert_shape(state, (batch,))
    alive = jnp.ones((batch,), dtype=bool)
    chex.assert_shape(alive, (batch,))
    keys = jax.random.split(key, n_time)
    chex.assert_shape(keys, (n_time,))

    def step(carry, day_key):
        """Sample one daily transition for state/alive carry; return next state and event arrays."""
        state, alive = carry
        key_state, key_default = jax.random.split(day_key)
        probabilities = tables['transition'][state]
        chex.assert_shape(probabilities, (batch, n_states))
        logits = jnp.log(probabilities)
        chex.assert_shape(logits, (batch, n_states))
        next_state = jax.random.categorical(key_state, logits, axis=-1).astype(jnp.int32)
        chex.assert_shape(next_state, (batch,))
        uniforms = jax.random.uniform(key_default, (batch,))
        chex.assert_shape(uniforms, (batch,))
        default = alive & (uniforms >= tables['survival'][state])
        chex.assert_shape(default, (batch,))
        next_alive = alive & ~default
        chex.assert_shape(next_alive, (batch,))
        next_state = jnp.where(next_alive, next_state, state)
        chex.assert_shape(next_state, (batch,))
        return (next_state, next_alive), (next_state, next_alive, default)

    _, (states, alive_after, defaults) = jax.lax.scan(step, (state, alive), keys)
    chex.assert_shape((states, alive_after, defaults), (n_time, batch))
    states = jnp.concatenate((state[None], states), axis=0).T
    alive_path = jnp.concatenate((alive[None], alive_after), axis=0).T
    default_path = defaults.T
    chex.assert_shape((states, alive_path), (batch, n_time + 1))
    chex.assert_shape(default_path, (batch, n_time))
    return dict(state=states, alive=alive_path, default=default_path)
