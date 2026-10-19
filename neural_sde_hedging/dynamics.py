"""Coupled neural latent/spot SDE with exact Euler transition scores and replay."""
import math

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp

from .config import SDEConfig


class NeuralSDE(eqx.Module):
    """Trainable joint drift/diffusion and market price of risk neural networks."""

    latent_q_drift: eqx.nn.MLP
    diffusion_network: eqx.nn.MLP
    risk_price: eqx.nn.MLP

    def __init__(self, key, config: SDEConfig, width=32, depth=2):
        """Initialize three networks from key/config/width/depth; return a teacher PyTree."""
        drift_key, diffusion_key, risk_key = jax.random.split(key, 3)
        dim = config.state_dim
        inputs = dim + 1
        self.latent_q_drift = eqx.nn.MLP(inputs, config.n_latent, width, depth,
                                        activation=jax.nn.tanh, key=drift_key)
        self.diffusion_network = eqx.nn.MLP(inputs, dim * (dim + 1) // 2, width, depth,
                                           activation=jax.nn.tanh, key=diffusion_key)
        self.risk_price = eqx.nn.MLP(inputs, dim, width, depth,
                                   activation=jax.nn.tanh, key=risk_key)


class SDEPaths(eqx.Module):
    """Joint states, primitive innovations, behavior density, and fixed time grid."""

    states: jax.Array
    innovations: jax.Array
    log_density: jax.Array
    times: jax.Array

    @property
    def log_spot(self):
        """Return observable log spot [batch,time], excluding all latent coordinates."""
        result = self.states[..., 0]
        chex.assert_shape(result, self.states.shape[:2])
        return result

    @property
    def latent(self):
        """Return hidden factors [batch,time,n_latent] for diagnostics only."""
        result = self.states[..., 1:]
        chex.assert_shape(result, (*self.states.shape[:2], self.states.shape[-1] - 1))
        return result


def transition_coefficients(teacher, state, time, config, measure="P"):
    """Map teacher, joint states [B,D], scalar time/config/measure to drift [B,D], L [B,D,D]."""
    batch, dim = state.shape
    chex.assert_shape(state, (batch, config.state_dim))
    chex.assert_shape(time, ())
    if measure not in ("P", "Q"):
        raise ValueError("measure must be P or Q")
    scaled_state = jnp.tanh(state / config.feature_scale)
    chex.assert_shape(scaled_state, (batch, dim))
    time_features = jnp.full((batch, 1), time / (config.n_steps * config.dt))
    chex.assert_shape(time_features, (batch, 1))
    features = jnp.concatenate((time_features, scaled_state), axis=-1)
    chex.assert_shape(features, (batch, dim + 1))
    packed = jax.vmap(teacher.diffusion_network)(features)
    chex.assert_shape(packed, (batch, dim * (dim + 1) // 2))
    row, col = jnp.tril_indices(dim)
    lower = config.off_diagonal_max * jnp.tanh(packed)
    chex.assert_shape(lower, packed.shape)
    diagonal = config.diagonal_min + (config.diagonal_max - config.diagonal_min) * jax.nn.sigmoid(packed)
    chex.assert_shape(diagonal, packed.shape)
    values = jnp.where(row == col, diagonal, lower)
    chex.assert_shape(values, packed.shape)
    diffusion = jnp.zeros((batch, dim, dim), dtype=state.dtype).at[:, row, col].set(values)
    chex.assert_shape(diffusion, (batch, dim, dim))
    latent_drift = config.latent_drift_max * jnp.tanh(jax.vmap(teacher.latent_q_drift)(features))
    chex.assert_shape(latent_drift, (batch, config.n_latent))
    spot_variance = jnp.sum(jnp.square(diffusion[:, 0, :]), axis=-1, keepdims=True)
    chex.assert_shape(spot_variance, (batch, 1))
    drift = jnp.concatenate((-0.5 * spot_variance, latent_drift), axis=-1)
    chex.assert_shape(drift, (batch, dim))
    if measure == "P":
        market_price = config.risk_price_max * jnp.tanh(jax.vmap(teacher.risk_price)(features))
        chex.assert_shape(market_price, (batch, dim))
        premium = jnp.einsum("bij,bj->bi", diffusion, market_price)
        chex.assert_shape(premium, (batch, dim))
        drift = drift + premium
        chex.assert_shape(drift, (batch, dim))
    return drift, diffusion


def transition_log_prob(next_state, state, time, teacher, config, measure="P"):
    """Evaluate exact joint Euler Gaussian log density [B] of next_state given state [B,D]."""
    batch, dim = state.shape
    chex.assert_shape(next_state, (batch, dim))
    drift, diffusion = transition_coefficients(teacher, state, time, config, measure)
    residual = (next_state - state - config.dt * drift) / math.sqrt(config.dt)
    chex.assert_shape(residual, (batch, dim))
    standardized = jax.vmap(lambda matrix, vector: jsp.linalg.solve_triangular(matrix, vector, lower=True))(diffusion, residual)
    chex.assert_shape(standardized, (batch, dim))
    diagonal = jnp.diagonal(diffusion, axis1=-2, axis2=-1)
    chex.assert_shape(diagonal, (batch, dim))
    log_determinant = jnp.sum(jnp.log(diagonal), axis=-1) + 0.5 * dim * math.log(config.dt)
    chex.assert_shape(log_determinant, (batch,))
    squared_norm = jnp.sum(jnp.square(standardized), axis=-1)
    chex.assert_shape(squared_norm, (batch,))
    result = -0.5 * (dim * math.log(2 * math.pi) + squared_norm) - log_determinant
    chex.assert_shape(result, (batch,))
    return result


def sample_noise(key, batch, config):
    """Draw standard independent Euler innovations [batch,n_steps,state_dim] from key/config."""
    innovations = jax.random.normal(key, (batch, config.n_steps, config.state_dim))
    chex.assert_shape(innovations, (batch, config.n_steps, config.state_dim))
    return innovations


def simulate(teacher, innovations, config, measure="P", initial_state=None):
    """Generate differentiable SDEPaths from teacher/noise/config and optional fixed initial [B,D]."""
    batch, n_steps, dim = innovations.shape
    chex.assert_shape(innovations, (batch, config.n_steps, config.state_dim))
    if initial_state is None:
        initial_state = jnp.zeros((batch, dim), dtype=innovations.dtype).at[:, 0].set(math.log(config.spot0))
    chex.assert_shape(initial_state, (batch, dim))
    times = jnp.arange(n_steps + 1, dtype=innovations.dtype) * config.dt
    chex.assert_shape(times, (n_steps + 1,))
    scan_noise = jnp.swapaxes(innovations, 0, 1)
    chex.assert_shape(scan_noise, (n_steps, batch, dim))

    def step(state, inputs):
        """Advance states [B,D] with (time scalar, noise [B,D]); return next state and density."""
        time, noise = inputs
        chex.assert_shape(noise, (batch, dim))
        drift, diffusion = transition_coefficients(teacher, state, time, config, measure)
        stochastic_increment = jnp.einsum("bij,bj->bi", diffusion, noise)
        chex.assert_shape(stochastic_increment, (batch, dim))
        next_state = state + config.dt * drift + math.sqrt(config.dt) * stochastic_increment
        chex.assert_shape(next_state, (batch, dim))
        log_density = transition_log_prob(next_state, state, time, teacher, config, measure)
        chex.assert_shape(log_density, (batch,))
        return next_state, (next_state, log_density)

    _, (future, step_log_density) = jax.lax.scan(step, initial_state, (times[:-1], scan_noise))
    chex.assert_shape(future, (n_steps, batch, dim))
    chex.assert_shape(step_log_density, (n_steps, batch))
    future = jnp.swapaxes(future, 0, 1)
    chex.assert_shape(future, (batch, n_steps, dim))
    states = jnp.concatenate((initial_state[:, None, :], future), axis=1)
    chex.assert_shape(states, (batch, n_steps + 1, dim))
    log_density = jnp.sum(step_log_density, axis=0)
    chex.assert_shape(log_density, (batch,))
    return SDEPaths(states, innovations, log_density, times)


def path_log_prob(teacher, paths, config, measure="P"):
    """Score fixed complete joint paths under current teacher; stop all sample-coordinate gradients."""
    states = jax.lax.stop_gradient(paths.states)
    batch, n_times, dim = states.shape
    chex.assert_shape(states, (batch, config.n_steps + 1, config.state_dim))
    times = jnp.arange(n_times - 1, dtype=states.dtype) * config.dt
    chex.assert_shape(times, (n_times - 1,))
    previous = jnp.swapaxes(states[:, :-1, :], 0, 1)
    chex.assert_shape(previous, (n_times - 1, batch, dim))
    following = jnp.swapaxes(states[:, 1:, :], 0, 1)
    chex.assert_shape(following, (n_times - 1, batch, dim))

    def accumulate(total, inputs):
        """Accumulate transition densities from total [B] and one fixed joint transition; return totals."""
        before, after, time = inputs
        increment = transition_log_prob(after, before, time, teacher, config, measure)
        chex.assert_shape(increment, (batch,))
        total = total + increment
        chex.assert_shape(total, (batch,))
        return total, None

    density, _ = jax.lax.scan(accumulate, jnp.zeros((batch,), dtype=states.dtype), (previous, following, times))
    chex.assert_shape(density, (batch,))
    return density
