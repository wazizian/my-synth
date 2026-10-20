"""Fixed non-neural easy/no-shock and hard/persistent-shock benchmark laws."""

import math

import chex
import jax
import jax.numpy as jnp


def sample_regime(key, batch, config, problem, regime):
    """Return regime states [B,T+1,D] from key, dimensions, problem and easy/hard label.

    Options restore the diagnostic's physical half-lives (40,3) days, eta .35,
    20% base volatility and two log(2) shocks at day 35. Credit uses a one-factor
    persistent positive-hazard analogue; it is not the historical two-factor
    credit-curve benchmark. Easy has no shock and three-day factor half-lives.
    The easy/hard economic difficulty ordering is not certified.
    """
    if regime not in ("easy", "hard"):
        raise ValueError("Regime must be easy or hard")
    n_steps, dim = config.n_steps, config.state_dim
    if problem not in ("options", "credit"):
        raise ValueError("Unknown hard target problem")
    expected = 3 if problem == "options" else 1
    if dim != expected:
        raise ValueError("Hard target requires options D=3 or continuous credit D=1")
    noise = jax.random.normal(key, (n_steps, batch, dim))
    chex.assert_shape(noise, (n_steps, batch, dim))
    n_factors = 2 if problem == "options" else 1
    half_lives = jnp.asarray((40., 3.) if problem == "options" else (40.,))
    if regime == "easy":
        half_lives = jnp.full((n_factors,), 3.)
    chex.assert_shape(half_lives, (n_factors,))
    decay = jnp.exp(-math.log(2.) / half_lives)
    chex.assert_shape(decay, (n_factors,))
    sd = .35 * jnp.sqrt(config.dt * (1-decay**2) / (-2*jnp.log(decay)))
    chex.assert_shape(sd, (n_factors,))
    initial = jnp.zeros((batch, dim))
    chex.assert_shape(initial, (batch, dim))
    if problem == "options":
        initial = initial.at[:, 0].set(math.log(config.spot0))
    else:
        initial = initial.at[:, 0].set(config.initial_latent)
    chex.assert_shape(initial, (batch, dim))

    def step(state, inputs):
        """Advance a daily state [B,D] with a fixed shock and independent Gaussian noise."""
        day, z = inputs
        shock = jnp.where((day == 35) & (regime == "hard"), math.log(2.), 0.)
        chex.assert_shape(shock, ())
        if problem == "options":
            factors = state[:, 1:] + shock
            chex.assert_shape(factors, (batch, 2))
            variance = .04 * jnp.exp(jnp.sum(factors, axis=1))
            chex.assert_shape(variance, (batch,))
            log_spot = state[:, 0] - .5*variance*config.dt + jnp.sqrt(variance*config.dt)*z[:, 0]
            chex.assert_shape(log_spot, (batch,))
            factors = factors*decay + sd*z[:, 1:]
            chex.assert_shape(factors, (batch, 2))
            following = jnp.concatenate((log_spot[:, None], factors), axis=1)
        else:
            centered = state - config.initial_latent + shock
            chex.assert_shape(centered, (batch, 1))
            following = config.initial_latent + centered*decay + sd*z
        chex.assert_shape(following, (batch, dim))
        return following, following

    _, future = jax.lax.scan(step, initial, (jnp.arange(n_steps), noise))
    chex.assert_shape(future, (n_steps, batch, dim))
    future = jnp.swapaxes(future, 0, 1)
    chex.assert_shape(future, (batch, n_steps, dim))
    result = jnp.concatenate((initial[:, None, :], future), axis=1)
    chex.assert_shape(result, (batch, n_steps+1, dim))
    return result


def sample_hard(key, batch, config, problem):
    """Return hard states [B,T+1,D] from key, batch, simulator config and problem."""
    return sample_regime(key, batch, config, problem, "hard")
