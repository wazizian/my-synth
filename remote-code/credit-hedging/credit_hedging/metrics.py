"""Positive-loss risk metrics; all monetary values are per unit bond face value."""
import math

import chex
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def entropic_risk(loss, gamma=10.0):
    """Return stable empirical entropic risk from loss [batch] and positive static gamma."""
    batch = loss.shape[0]
    chex.assert_shape(loss, (batch,))
    if not math.isfinite(gamma) or gamma <= 0 or batch < 1:
        raise ValueError('Need finite positive gamma and nonempty loss batch')
    risk = (logsumexp(gamma*loss)-jnp.log(batch))/gamma
    chex.assert_shape(risk, ())
    return risk


def evaluate_metrics(result, gamma=10.0, alpha=0.95):
    """Map a rollout PyTree to entropic/CVaR/daily-SD metrics and an accounting residual.

    Daily SD is RMS across days of cross-episode standard deviations, including
    absorbing zero returns after default. CVaR uses fractional boundary mass.
    """
    loss = result['loss']
    batch, n_time = result['records']['daily_loss'].shape
    chex.assert_shape(loss, (batch,))
    chex.assert_shape(result['records']['discounted_cost'], (batch, n_time))
    if batch < 1 or n_time < 1:
        raise ValueError('Evaluation needs nonempty episode and time axes')
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie in (0, 1)')
    sorted_loss = jnp.sort(loss)[::-1]
    chex.assert_shape(sorted_loss, (batch,))
    weights = jnp.clip((1-alpha)*batch-jnp.arange(batch), 0, 1)
    chex.assert_shape(weights, (batch,))
    cvar = jnp.sum(weights*sorted_loss)/((1-alpha)*batch)
    chex.assert_shape(cvar, ())
    daily_mean = jnp.mean(result['records']['daily_loss'], axis=0)
    daily_std = jnp.std(result['records']['daily_loss'], axis=0)
    chex.assert_shape((daily_mean, daily_std), (n_time,))
    residual = jnp.sum(result['records']['daily_loss'], axis=1)-loss
    chex.assert_shape(residual, (batch,))
    chex.assert_shape(result['defaulted'], (batch,))
    default_fraction = jnp.mean(result['defaulted'].astype(loss.dtype))
    chex.assert_shape(default_fraction, ())
    centered_loss = loss-jnp.max(loss)
    chex.assert_shape(centered_loss, (batch,))
    exponential_weights = jnp.exp(gamma*centered_loss)
    chex.assert_shape(exponential_weights, (batch,))
    exponential_weights = exponential_weights/jnp.sum(exponential_weights)
    chex.assert_shape(exponential_weights, (batch,))
    exponential_ess = 1/jnp.sum(exponential_weights**2)
    chex.assert_shape(exponential_ess, ())
    return dict(entropic=entropic_risk(loss, gamma), cvar=cvar,
                mean=jnp.mean(loss), terminal_std=jnp.std(loss),
                default_fraction=default_fraction,
                sample_count=jnp.asarray(batch), tail_mass=jnp.asarray((1-alpha)*batch),
                exponential_ess=exponential_ess, max_exponential_weight=jnp.max(exponential_weights),
                daily_error_std=jnp.sqrt(jnp.mean(daily_std**2)),
                daily_mean=daily_mean, daily_std=daily_std,
                mean_cost=jnp.mean(jnp.sum(result['records']['discounted_cost'], axis=1)),
                max_accounting_error=jnp.max(jnp.abs(residual)))
