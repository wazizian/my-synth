"""Independent two-factor positive-OU daily credit model.

The finite-state law is the scientific simulator. Independent OU transition
matrices act on a tensor grid without constructing a squared joint-state matrix.
An informed dealer publishes executable quotes; factors are not policy inputs.
Quotes, including old-vintage marks, may reveal state information. This is not
a claim of observational nonidentifiability or a posterior-filtered market.
"""
from dataclasses import dataclass
from numbers import Integral
import math

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr

from .market import MarketConfig


@dataclass(frozen=True)
class CurveConfig(MarketConfig):
    fast_reversion: float = 6.0
    fast_mean: float = 0.0
    fast_vol: float = 0.6
    fast_initial: float = 0.0
    fast_grid_min: float = -2.0
    fast_grid_max: float = 2.0
    fast_grid_size: int = 101

    def __post_init__(self):
        super().__post_init__()
        for name in ('fast_reversion', 'fast_mean', 'fast_vol', 'fast_initial',
                     'fast_grid_min', 'fast_grid_max'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        if isinstance(self.fast_grid_size, bool) or not isinstance(self.fast_grid_size, Integral):
            raise ValueError('fast_grid_size must be an integer')
        if self.fast_grid_size < 1 or self.fast_vol < 0:
            raise ValueError('Invalid fast grid or volatility')
        if self.fast_reversion <= self.mean_reversion:
            raise ValueError('Fast reversion must exceed slow reversion')
        if self.fast_grid_size == 1:
            if self.fast_vol != 0 or self.fast_initial != self.fast_mean:
                raise ValueError('Singleton fast state must be constant at its mean')
        elif not (self.fast_vol > 0 and self.fast_grid_min < self.fast_grid_max
                  and self.fast_grid_min <= self.fast_initial <= self.fast_grid_max):
            raise ValueError('Invalid stochastic fast factor')


def ou_transition(grid, kappa, theta, sigma, dt):
    """Gaussian cell probabilities with boundary tail mass retained."""
    if grid.shape[0] == 1:
        return jnp.ones((1, 1), dtype=grid.dtype)
    means = theta + jnp.exp(-kappa * dt) * (grid - theta)
    variance = sigma**2 * (-jnp.expm1(-2*kappa*dt)) / (2*kappa)
    edges = jnp.concatenate((jnp.array([-jnp.inf]), (grid[1:]+grid[:-1])/2,
                             jnp.array([jnp.inf])))
    probs = jnp.maximum(jnp.diff(ndtr((edges[None]-means[:, None])/jnp.sqrt(variance)), axis=1), 0)
    return probs / probs.sum(axis=1, keepdims=True)


def continuation(values, px, py):
    """Conditional expectation on independent factor transitions."""
    return jnp.einsum('ia,abk,jb->ijk', px, values, py, optimize=True)


def build_curve_tables(config: CurveConfig):
    dt = 1 / config.trading_days
    x = jnp.linspace(config.grid_min, config.grid_max, config.grid_size)
    y = (jnp.array([config.fast_mean]) if config.fast_grid_size == 1 else
         jnp.linspace(config.fast_grid_min, config.fast_grid_max, config.fast_grid_size))
    px = ou_transition(x, config.mean_reversion, config.latent_mean, config.latent_vol, dt)
    py = ou_transition(y, config.fast_reversion, config.fast_mean, config.fast_vol, dt)
    hazard = jax.nn.softplus(x[:, None] + y[None, :])
    survival = jnp.exp(-dt*hazard)
    discount = jnp.exp(-dt*config.rate)
    def price_leg(terminal, cash, days):
        # A scan followed by concatenating maturity zero temporarily retains
        # two complete five-year tensors. Write each maturity into one loop
        # buffer so the validated double-precision grid fits the worker.
        storage = jnp.zeros((days+1,)+terminal.shape, dtype=terminal.dtype)
        storage = storage.at[0].set(terminal)

        def step(day, prices):
            previous = jax.lax.dynamic_index_in_dim(prices, day-1, keepdims=False)
            current = discount*(cash + survival[..., None]*continuation(previous, px, py))
            return jax.lax.dynamic_update_index_in_dim(prices, current, day, axis=0)

        return jax.lax.fori_loop(1, days+1, step, storage)

    # CDS never has more than cds_days remaining. Do not retain five years of
    # unused CDS-leg maturities: that would triple the large refinement grid.
    bond_prices = price_leg(jnp.ones_like(hazard)[..., None],
                           ((1-survival)*config.recovery)[..., None], config.bond_days)
    cds_cash = jnp.stack(((1-survival)*(1-config.recovery), jnp.full_like(hazard, dt)), -1)
    cds_prices = price_leg(jnp.zeros_like(cds_cash), cds_cash, config.cds_days)
    bond = bond_prices[..., 0].reshape(config.bond_days+1, -1)
    protection = cds_prices[..., 0].reshape(config.cds_days+1, -1)
    annuity = cds_prices[..., 1].reshape(config.cds_days+1, -1)
    initial = jnp.argmin(jnp.abs(x-config.initial_latent))*len(y) + jnp.argmin(jnp.abs(y-config.fast_initial))
    return dict(x_grid=x, y_grid=y, transition_x=px, transition_y=py,
                hazard=hazard.ravel(), survival=survival.ravel(), bond=bond,
                protection=protection, annuity=annuity,
                par_spread=protection[config.cds_days]/annuity[config.cds_days],
                initial_state=initial)


def simulate_curve_paths(key, tables, config: CurveConfig, batch_size: int):
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    ny = config.fast_grid_size
    initial = jnp.full((batch_size,), tables['initial_state'], dtype=jnp.int32)
    alive = jnp.ones(batch_size, dtype=bool)
    cdf_x = jnp.cumsum(tables['transition_x'], axis=1).at[:, -1].set(1)
    cdf_y = jnp.cumsum(tables['transition_y'], axis=1).at[:, -1].set(1)

    def step(carry, key):
        state, alive = carry
        uniforms = jax.random.uniform(key, (3, batch_size))
        ix, iy = state // ny, state % ny
        # Inverse CDF avoids one Gumbel draw per transition destination.
        nx = jnp.sum(uniforms[0, :, None] >= cdf_x[ix], axis=1)
        ny_index = jnp.sum(uniforms[1, :, None] >= cdf_y[iy], axis=1)
        default = alive & (uniforms[2] >= tables['survival'][state])
        next_alive = alive & ~default
        next_state = jnp.where(next_alive, nx*ny+ny_index, state).astype(jnp.int32)
        return (next_state, next_alive), (next_state, next_alive, default)

    _, (states, alive_after, defaults) = jax.lax.scan(
        step, (initial, alive), jax.random.split(key, config.bond_days))
    return dict(state=jnp.concatenate((initial[None], states)).T,
                alive=jnp.concatenate((alive[None], alive_after)).T,
                default=defaults.T)
