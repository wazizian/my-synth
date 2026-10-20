"""Neural physical proposals on the historical two-factor credit grid.

Prices are the original hard-regime finite-grid prices. This proposal changes
physical transitions, not contractual quotes; the true target and its Q tables
remain fixed and independent of the teacher. No continuous-grid interpolation.
"""
from dataclasses import dataclass
import math

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr

from credit_hedging.curve_market import CurveConfig
from .dynamics import SDEPaths


@dataclass(frozen=True)
class CreditGridConfig:
    """Static hard-market grid and path dimensions for a two-factor neural proposal."""
    market: CurveConfig

    def __post_init__(self):
        """Reject singleton/degenerate fast grids: proposals are explicitly hard-market only."""
        if self.market.fast_grid_size < 3 or self.market.fast_vol <= 0:
            raise ValueError("Two-factor neural credit proposal requires a stochastic hard fast grid")

    @property
    def n_steps(self):
        """Return number of daily transitions."""
        return self.market.bond_days

    @property
    def state_dim(self):
        """Return packed index/alive/default coordinate count."""
        return 3

    @property
    def dt(self):
        """Return daily year fraction."""
        return 1/self.market.trading_days


class NeuralCredit(eqx.Module):
    """Train mean/volatility corrections of both grid factors; zero head starts at true hard law."""
    network: eqx.nn.MLP

    def __init__(self, key, config, width=32, depth=2):
        """Initialize a two-input four-output MLP; return zero corrections at all states."""
        self.network = eqx.nn.MLP(2, 4, width, depth, activation=jax.nn.tanh, key=key)
        last = self.network.layers[-1]
        self.network = eqx.tree_at(lambda m: (m.layers[-1].weight, m.layers[-1].bias), self.network,
                                   (jnp.zeros_like(last.weight), jnp.zeros_like(last.bias)))


def transition_rows(teacher, indices, config):
    """Return slow/fast row probabilities [B,Nx],[B,Ny] at packed grid indices [B]."""
    market = config.market
    batch = indices.shape[0]
    x = jnp.linspace(market.grid_min, market.grid_max, market.grid_size)
    y = jnp.linspace(market.fast_grid_min, market.fast_grid_max, market.fast_grid_size)
    chex.assert_shape(x, (market.grid_size,))
    chex.assert_shape(y, (market.fast_grid_size,))
    before = jnp.stack((x[indices//market.fast_grid_size], y[indices%market.fast_grid_size]), axis=1)
    chex.assert_shape(before, (batch, 2))
    raw = jax.vmap(teacher.network)(jnp.tanh(before/2.))
    chex.assert_shape(raw, (batch, 4))
    rows = []
    for factor, (grid, kappa, mean, sigma) in enumerate((
            (x, market.mean_reversion, market.latent_mean, market.latent_vol),
            (y, market.fast_reversion, market.fast_mean, market.fast_vol))):
        decay = math.exp(-kappa*config.dt)
        center = mean+decay*(before[:, factor]-mean)+.5*config.dt*jnp.tanh(raw[:, factor])
        sd = sigma*math.sqrt(-math.expm1(-2*kappa*config.dt)/(2*kappa))*jnp.exp(.5*jnp.tanh(raw[:, 2+factor]))
        chex.assert_shape((center, sd), (batch,))
        edges = jnp.concatenate((jnp.array([-jnp.inf]), (grid[1:]+grid[:-1])/2, jnp.array([jnp.inf])))
        chex.assert_shape(edges, (grid.shape[0]+1,))
        # Keep infinities outside the differentiable division (avoid inf*0 VJPs).
        middle = (edges[None, 1:-1]-center[:, None])/sd[:, None]
        cdf = jnp.concatenate((jnp.zeros((batch, 1)), ndtr(middle), jnp.ones((batch, 1))), axis=1)
        row = jnp.maximum(jnp.diff(cdf, axis=1), jnp.finfo(cdf.dtype).tiny)
        row = row/jnp.sum(row, axis=1, keepdims=True)
        chex.assert_shape(row, (batch, grid.shape[0]))
        rows.append(row)
    return tuple(rows)


def sample_noise(key, batch, config):
    """Return primitive uniforms [B,T,3] for inverse-CDF factor transitions and default."""
    result = jax.random.uniform(key, (batch, config.n_steps, 3))
    chex.assert_shape(result, (batch, config.n_steps, 3))
    return result


def simulate(teacher, innovations, config):
    """Return packed SDEPaths under the two-factor grid proposal, absorbed at daily default."""
    market = config.market
    batch = innovations.shape[0]
    chex.assert_shape(innovations, (batch, config.n_steps, 3))
    x = jnp.linspace(market.grid_min, market.grid_max, market.grid_size)
    y = jnp.linspace(market.fast_grid_min, market.fast_grid_max, market.fast_grid_size)
    index0 = jnp.argmin(jnp.abs(x-market.initial_latent))*market.fast_grid_size+jnp.argmin(jnp.abs(y-market.fast_initial))
    initial = jnp.stack((jnp.full(batch, index0), jnp.ones(batch), jnp.zeros(batch)), axis=1)
    chex.assert_shape(initial, (batch, 3))

    def step(state, u):
        """Sample factor cells and current-hazard default, then freeze the state on death."""
        index = state[:, 0].astype(jnp.int32)
        px, py = transition_rows(teacher, index, config)
        nx = jnp.sum(u[:, :1] >= jnp.cumsum(px, axis=1).at[:, -1].set(1.), axis=1)
        ny = jnp.sum(u[:, 1:2] >= jnp.cumsum(py, axis=1).at[:, -1].set(1.), axis=1)
        hazard = jax.nn.softplus(x[index//market.fast_grid_size]+y[index%market.fast_grid_size])
        survival = jnp.exp(-hazard*config.dt)
        alive = state[:, 1] > .5
        default = alive & (u[:, 2] >= survival)
        following_alive = alive & ~default
        following_index = jnp.where(following_alive, nx*market.fast_grid_size+ny, index)
        following = jnp.stack((following_index, following_alive, default), axis=1).astype(state.dtype)
        chex.assert_shape(following, (batch, 3))
        return following, following

    _, future = jax.lax.scan(step, initial, jnp.swapaxes(innovations, 0, 1))
    states = jnp.concatenate((initial[:, None], jnp.swapaxes(future, 0, 1)), axis=1)
    chex.assert_shape(states, (batch, config.n_steps+1, 3))
    paths = SDEPaths(states, innovations, jnp.zeros(batch), jnp.arange(config.n_steps+1)*config.dt)
    return eqx.tree_at(lambda p: p.log_density, paths, path_log_prob(teacher, paths, config))


def path_log_prob(teacher, paths, config):
    """Return exact stopped-path grid/default log scores [B], omitting post-default transitions."""
    market = config.market
    states = jax.lax.stop_gradient(paths.states)
    batch = states.shape[0]
    chex.assert_shape(states, (batch, config.n_steps+1, 3))
    x = jnp.linspace(market.grid_min, market.grid_max, market.grid_size)
    y = jnp.linspace(market.fast_grid_min, market.fast_grid_max, market.fast_grid_size)

    def step(total, pair):
        """Accumulate one grid probability and Cox survival/default likelihood."""
        before, after = pair
        index, following = before[:, 0].astype(jnp.int32), after[:, 0].astype(jnp.int32)
        px, py = transition_rows(teacher, index, config)
        row = jnp.arange(batch)
        log_transition = jnp.log(px[row, following//market.fast_grid_size])+jnp.log(py[row, following%market.fast_grid_size])
        hazard = jax.nn.softplus(x[index//market.fast_grid_size]+y[index%market.fast_grid_size])
        log_survive = -hazard*config.dt
        log_default = jnp.log(-jnp.expm1(log_survive))
        alive, default = before[:, 1] > .5, after[:, 2] > .5
        score = jnp.where(default, log_default, log_survive+log_transition)
        total = total+jnp.where(alive, score, 0.)
        chex.assert_shape(total, (batch,))
        return total, None

    result, _ = jax.lax.scan(step, jnp.zeros(batch),
                             (jnp.swapaxes(states[:, :-1], 0, 1), jnp.swapaxes(states[:, 1:], 0, 1)))
    chex.assert_shape(result, (batch,))
    return result
