"""Explicit empirical risk metrics, in underlying currency units."""

import chex
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def risk_metrics(losses, gamma=10., alpha=.95, daily_marked_errors=None):
    """Return terminal entropic/CVaR/std/mean and optional pooled daily error std.

    losses is [episodes]; positive means loss. CVaR uses fractional mass in the
    worst 1-alpha tail; std is population ddof0. daily_marked_errors, if supplied,
    must be actual daily marked hedging errors [episodes,days], not terminal
    losses, returns or unmarked cash increments. No absent metric is fabricated.
    """
    batch = losses.shape[0]
    chex.assert_shape(losses, (batch,))
    if batch < 1 or not 0 < alpha < 1 or gamma <= 0:
        raise ValueError("Nonempty losses, positive gamma and 0<alpha<1 required")
    descending = jnp.sort(losses)[::-1]
    chex.assert_shape(descending, (batch,))
    mass = (1. - alpha) * batch
    weights = jnp.clip(mass - jnp.arange(batch), 0., 1.)
    chex.assert_shape(weights, (batch,))
    result = {"entropic": (logsumexp(gamma * losses) - jnp.log(batch)) / gamma,
              "cvar95" if alpha == .95 else "cvar": jnp.sum(descending * weights) / mass,
              "terminal_std": jnp.std(losses), "mean_loss": jnp.mean(losses)}
    if daily_marked_errors is not None:
        chex.assert_shape(daily_marked_errors, (batch, daily_marked_errors.shape[1]))
        result["daily_error_std"] = jnp.std(daily_marked_errors)
    for value in result.values():
        chex.assert_shape(value, ())
    return result
