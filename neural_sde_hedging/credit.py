"""Neural hazard SDE on a pricing-coherent finite-state Euler grid.

The target/pricing Q drift and volatility are fixed seeded neural networks.
The teacher learns a bounded neural market price of hazard risk, changing the
P drift. Thus P and Q share diffusion and the model retains a fixed economic
pricing reference. No categorical selection of market regimes is used.
"""

from dataclasses import dataclass
import math

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr

from credit_hedging.environment import rollout
from credit_hedging.market import MarketConfig, price_tables

from .dynamics import SDEPaths


@dataclass(frozen=True)
class CreditSDEConfig:
    """Static neural hazard grid, calendar and fixed neural-Q initialization."""

    n_steps: int = 1260
    cds_days: int = 126
    grid_size: int = 301
    pricing_seed: int = 20261117
    trading_days: int = 252
    grid_min: float = -7.
    grid_max: float = 0.
    initial_latent: float = -3.5
    drift_max: float = 2.
    volatility_min: float = .05
    volatility_max: float = .6
    risk_price_max: float = 2.

    def __post_init__(self):
        """Validate finite ordered grid, positive calendar and neural bounds."""
        if any(isinstance(x, bool) or not isinstance(x, int) for x in
               (self.n_steps, self.cds_days, self.grid_size, self.pricing_seed, self.trading_days)):
            raise ValueError("Credit dimensions and pricing seed must be integers")
        if min(self.n_steps, self.cds_days, self.trading_days) < 1 or self.grid_size < 3:
            raise ValueError("Credit calendar must be positive; grid needs at least 3 points")
        values = (self.grid_min, self.grid_max, self.initial_latent, self.drift_max,
                  self.volatility_min, self.volatility_max, self.risk_price_max)
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Credit SDE coefficients must be finite")
        if not self.grid_min < self.grid_max or not self.grid_min <= self.initial_latent <= self.grid_max:
            raise ValueError("Invalid credit grid/initial state")
        if min(self.drift_max, self.volatility_min, self.risk_price_max) <= 0 or self.volatility_max < self.volatility_min:
            raise ValueError("Invalid neural credit coefficient bounds")

    @property
    def state_dim(self):
        """Return packed path width: hazard latent, alive indicator, default event."""
        return 3

    @property
    def dt(self):
        """Return the daily integration interval in years."""
        return 1. / self.trading_days


class NeuralCredit(eqx.Module):
    """Trainable bounded market-price-of-risk network for the hazard SDE."""

    risk_price: eqx.nn.MLP

    def __init__(self, key, config, width=32, depth=2):
        """Build the teacher network from a key and static grid configuration."""
        self.risk_price = eqx.nn.MLP(1, 1, width, depth,
                                     activation=jax.nn.tanh, key=key)


def transition_kernel(teacher, config, measure="P"):
    """Return grid and exact binned Euler kernel [states,states] under P or Q.

    Q coefficients are fixed neural functions of the latent state. Under P the
    teacher adds sigma(x)*lambda_phi(x) to Q drift. Infinite end-bin edges collect
    boundary tails; the finite-grid chain, not the continuum, is the traded model.
    """
    if measure not in ("P", "Q"):
        raise ValueError("Credit measure must be P or Q")
    n_states = config.grid_size
    grid = jnp.linspace(config.grid_min, config.grid_max, n_states)
    chex.assert_shape(grid, (n_states,))
    features = jnp.tanh(grid[:, None] / 2.)
    chex.assert_shape(features, (n_states, 1))
    drift_key, volatility_key = jax.random.split(jax.random.key(config.pricing_seed))
    drift_model = eqx.nn.MLP(1, 1, 16, 1, activation=jax.nn.tanh, key=drift_key)
    volatility_model = eqx.nn.MLP(1, 1, 16, 1, activation=jax.nn.tanh, key=volatility_key)
    drift = config.drift_max * jnp.tanh(jax.vmap(drift_model)(features)[:, 0])
    chex.assert_shape(drift, (n_states,))
    volatility = config.volatility_min + (config.volatility_max - config.volatility_min) * jax.nn.sigmoid(jax.vmap(volatility_model)(features)[:, 0])
    chex.assert_shape(volatility, (n_states,))
    if measure == "P":
        risk_price = config.risk_price_max * jnp.tanh(jax.vmap(teacher.risk_price)(features)[:, 0])
        chex.assert_shape(risk_price, (n_states,))
        drift = drift + volatility * risk_price
        chex.assert_shape(drift, (n_states,))
    means = grid + config.dt * drift
    chex.assert_shape(means, (n_states,))
    midpoints = (grid[:-1] + grid[1:]) / 2.
    chex.assert_shape(midpoints, (n_states - 1,))
    interior = (midpoints[None, :] - means[:, None]) / (math.sqrt(config.dt) * volatility[:, None])
    chex.assert_shape(interior, (n_states, n_states - 1))
    standardized = jnp.concatenate((jnp.full((n_states, 1), -12.), interior,
                                     jnp.full((n_states, 1), 12.)), axis=1)
    chex.assert_shape(standardized, (n_states, n_states + 1))
    standardized = jnp.clip(standardized, -12., 12.)
    chex.assert_shape(standardized, (n_states, n_states + 1))
    probabilities = jnp.maximum(jnp.diff(ndtr(standardized), axis=1), 0.)
    chex.assert_shape(probabilities, (n_states, n_states))
    # Small common floor gives finite likelihoods for replay across generators.
    probabilities = jnp.maximum(probabilities, 1e-12)
    chex.assert_shape(probabilities, (n_states, n_states))
    probabilities = probabilities / jnp.sum(probabilities, axis=1, keepdims=True)
    chex.assert_shape(probabilities, (n_states, n_states))
    return grid, probabilities


def pricing_tables(teacher, config, contract):
    """Return coherent bond/CDS Q tables for a neural credit config and contract."""
    if (contract.grid_size != config.grid_size or contract.bond_days != config.n_steps
            or contract.cds_days != config.cds_days or contract.trading_days != config.trading_days
            or contract.grid_min != config.grid_min or contract.grid_max != config.grid_max
            or contract.initial_latent != config.initial_latent):
        raise ValueError("Credit pricing and simulation grid/calendar must match")
    grid, transition = transition_kernel(teacher, config, "Q")
    hazard = jax.nn.softplus(grid)
    chex.assert_shape(hazard, (config.grid_size,))
    survival = jnp.exp(-config.dt * hazard)
    chex.assert_shape(survival, (config.grid_size,))
    return price_tables(grid, hazard, survival, transition, contract)


def sample_noise(key, batch, config):
    """Return exogenous uniforms [batch,days,3] for grid transitions and default."""
    noise = jax.random.uniform(key, (batch, config.n_steps, config.state_dim))
    chex.assert_shape(noise, (batch, config.n_steps, config.state_dim))
    return noise


def simulate(teacher, innovations, config):
    """Return packed SDEPaths from the neural Euler-grid law including default.

    States [batch,days+1,3] contain latent grid value, alive and today's default
    event. Default is at day end, before transition of surviving names. Absorbed
    states freeze. Log density includes every alive survival/transition or default
    probability; innovations are sampling uniforms, not Gaussian density scores.
    """
    batch, n_steps, dim = innovations.shape
    chex.assert_shape(innovations, (batch, config.n_steps, config.state_dim))
    grid, transition = transition_kernel(teacher, config)
    cumulative = jnp.cumsum(transition, axis=1).at[:, -1].set(1.)
    chex.assert_shape(cumulative, (config.grid_size, config.grid_size))
    initial = jnp.argmin(jnp.abs(grid - config.initial_latent))
    chex.assert_shape(initial, ())
    state = jnp.full((batch,), initial, dtype=jnp.int32)
    alive = jnp.ones((batch,), dtype=bool)
    chex.assert_shape((state, alive), (batch,))

    def step(carry, uniforms):
        """Advance state/alive/score vectors using daily uniforms [batch,3]."""
        state, alive, score = carry
        chex.assert_shape(uniforms, (batch, dim))
        survival = jnp.exp(-config.dt * jax.nn.softplus(grid[state]))
        chex.assert_shape(survival, (batch,))
        default = alive & (uniforms[:, 1] >= survival)
        survive = alive & ~default
        chex.assert_shape((default, survive), (batch,))
        proposed = jnp.sum(uniforms[:, :1] >= cumulative[state], axis=1)
        chex.assert_shape(proposed, (batch,))
        proposed = jnp.minimum(proposed, config.grid_size - 1)
        chex.assert_shape(proposed, (batch,))
        next_state = jnp.where(survive, proposed, state)
        chex.assert_shape(next_state, (batch,))
        event_logp = jnp.where(default, jnp.log1p(-survival),
                              jnp.log(survival) + jnp.log(transition[state, next_state]))
        chex.assert_shape(event_logp, (batch,))
        score = score + jnp.where(alive, event_logp, 0.)
        chex.assert_shape(score, (batch,))
        packed = jnp.stack((grid[next_state], survive.astype(grid.dtype), default.astype(grid.dtype)), axis=1)
        chex.assert_shape(packed, (batch, dim))
        return (next_state, survive, score), packed

    scan_noise = jnp.swapaxes(innovations, 0, 1)
    chex.assert_shape(scan_noise, (n_steps, batch, dim))
    (_, _, score), future = jax.lax.scan(step, (state, alive, jnp.zeros((batch,))), scan_noise)
    chex.assert_shape(future, (n_steps, batch, dim))
    start = jnp.stack((grid[state], alive.astype(grid.dtype), jnp.zeros((batch,))), axis=1)
    chex.assert_shape(start, (batch, dim))
    states = jnp.concatenate((start[:, None, :], jnp.swapaxes(future, 0, 1)), axis=1)
    chex.assert_shape(states, (batch, n_steps + 1, dim))
    times = jnp.arange(n_steps + 1) * config.dt
    chex.assert_shape(times, (n_steps + 1,))
    return SDEPaths(states, innovations, score, times)


def unpack_states(states, config):
    """Convert packed [batch,days+1,3] neural credit paths to ledger path arrays."""
    batch, n_dates, dim = states.shape
    chex.assert_shape(states, (batch, config.n_steps + 1, config.state_dim))
    indices = jnp.rint((states[..., 0] - config.grid_min) *
                       (config.grid_size - 1) / (config.grid_max - config.grid_min)).astype(jnp.int32)
    chex.assert_shape(indices, (batch, n_dates))
    alive = states[..., 1] > .5
    chex.assert_shape(alive, (batch, n_dates))
    defaults = states[:, 1:, 2] > .5
    chex.assert_shape(defaults, (batch, n_dates - 1))
    return dict(state=indices, alive=alive, default=defaults)


def path_log_prob(teacher, paths, config):
    """Score fixed packed credit paths under a candidate neural hazard teacher."""
    states = jax.lax.stop_gradient(paths.states)
    batch = states.shape[0]
    unpacked = unpack_states(states, config)
    grid, transition = transition_kernel(teacher, config)
    before, after = unpacked['state'][:, :-1], unpacked['state'][:, 1:]
    chex.assert_shape((before, after), (batch, config.n_steps))
    survival = jnp.exp(-config.dt * jax.nn.softplus(grid[before]))
    chex.assert_shape(survival, (batch, config.n_steps))
    rows = jnp.where(unpacked['default'], jnp.log1p(-survival),
                     jnp.log(survival) + jnp.log(transition[before, after]))
    chex.assert_shape(rows, (batch, config.n_steps))
    score = jnp.sum(jnp.where(unpacked['alive'][:, :-1], rows, 0.), axis=1)
    chex.assert_shape(score, (batch,))
    return score


def terminal_losses(policy, paths, tables, contract, config):
    """Return long-ZCB hedging loss [batch] using the existing causal CDS ledger."""
    unpacked = unpack_states(paths.states, config)
    result = rollout(policy, unpacked, tables, contract)
    chex.assert_shape(result['loss'], (paths.states.shape[0],))
    return result['loss']


def market_contract(config):
    """Return existing credit ledger conventions aligned to a CreditSDEConfig."""
    return MarketConfig(bond_days=config.n_steps, cds_days=config.cds_days,
                        trading_days=config.trading_days, grid_size=config.grid_size,
                        grid_min=config.grid_min, grid_max=config.grid_max,
                        initial_latent=config.initial_latent)
