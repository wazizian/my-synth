"""Self financing gains, transaction accounting, and differentiable Adam blocks."""
import math
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from .hedger import actions_one

DEFAULT_TASK = dict(inception=64, intervals=20, spot_cost=0.0005, call_cost=0.001)
# Base long-run variance: 70% base-dominant regime plus 30% across remaining 11.
BASE_VARIANCE = 0.7 * 0.04 + 0.3 * ((5 * 0.04 + 6 * 0.09) / 11)


def base_premium_fraction(intervals=20):
    """Public ATM BS cash endowment as a fraction of inception spot."""
    sigma_sqrt_t = math.sqrt(BASE_VARIANCE * intervals / 252)
    return math.erf(sigma_sqrt_t / (2 * math.sqrt(2)))


def portfolio_accounting(prices, holdings, rates):
    """prices=(T+1,2), holdings=(T,2); include initial/final changes once."""
    gains = jnp.sum(holdings * jnp.diff(prices, axis=0))
    changes = jnp.diff(jnp.concatenate((jnp.zeros_like(holdings[:1]), holdings)), axis=0)
    trading_costs = jnp.sum(jnp.abs(changes) * prices[:-1] * rates, axis=0)
    liquidation = jnp.abs(holdings[-1]) * prices[-1] * rates
    turnover = jnp.sum(jnp.abs(changes), axis=0) + jnp.abs(holdings[-1])
    return gains, trading_costs + liquidation, liquidation, turnover


def rollout(hedger, episode_batch, task_config=None):
    config = {**DEFAULT_TASK, **(task_config or {})}
    inception, intervals = config['inception'], config['intervals']
    rates = jnp.array([config['spot_cost'], config['call_cost']])
    premium = config.get('premium_fraction', base_premium_fraction(intervals))

    def one(spot, call):
        holdings = actions_one(hedger, spot, call, inception, intervals)
        prices = jnp.stack((spot, call), axis=-1)[inception:inception + intervals + 1]
        gain, costs, liquidation, turnover = portfolio_accounting(prices, holdings, rates)
        initial = spot[inception]
        payoff = jnp.maximum(spot[inception + intervals] - initial, 0)
        shortfall = (payoff - premium * initial - gain + costs.sum()) / initial
        bound = getattr(hedger, 'position_bound', 2.0)
        return shortfall, dict(gains=gain, costs=costs, liquidation_costs=liquidation,
            turnover=turnover, holdings=holdings,
            mean_abs_holdings=jnp.mean(jnp.abs(holdings), axis=0),
            saturation_fraction=jnp.mean(jnp.abs(holdings) >= 0.99 * bound),
            payoff=payoff, premium=premium * initial, pnl=-shortfall)

    return jax.vmap(one)(episode_batch['spot'], episode_batch['call'])


def example_losses(hedger, batch, task_config=None):
    return jnp.square(rollout(hedger, batch, task_config)[0])


def make_optimizer(learning_rate=1e-3):
    # eps_root makes second derivatives through initially zero Adam moments finite.
    return optax.chain(optax.clip_by_global_norm(1.0),
                       optax.adam(learning_rate, b1=0.9, b2=0.999,
                                  eps=1e-8, eps_root=1e-12))


def student_block(hedger, optimizer_state, batch, weights, optimizer,
                  task_config=None, learning_rate=None, return_diagnostics=False):
    """K carried updates, differentiable in every example weight and Adam state.

    ``batch`` has leading (K,B) axes. Returned losses are the pre-update weighted
    objectives. ``learning_rate`` optionally scales updates from ``optimizer``;
    construct an optimizer with unit LR when passing an absolute dynamic LR.
    With ``return_diagnostics=True`` return a third per-step diagnostics dict.
    Clipping diagnostics refer to make_optimizer's fixed global norm cap of 1.
    """
    params, static = eqx.partition(hedger, eqx.is_inexact_array)
    if weights is None:
        weights = jnp.ones(batch['spot'].shape[:2])

    def step(carry, inputs):
        model_params, state = carry
        minibatch, weight = inputs
        def loss_fn(p):
            loss = example_losses(eqx.combine(p, static), minibatch, task_config)
            return jnp.sum(weight * loss) / jnp.sum(weight)
        loss, grads = jax.value_and_grad(loss_fn)(model_params)
        updates, state = optimizer.update(grads, state, model_params)
        if learning_rate is not None:
            updates = jax.tree.map(lambda u: u * learning_rate, updates)
        output = loss
        if return_diagnostics:
            gradient_norm = optax.global_norm(grads)
            nonfinite = sum(jnp.sum(~jnp.isfinite(leaf)) for leaf in jax.tree.leaves(grads))
            output = (loss, dict(gradient_norm=gradient_norm,
                                clipping_applied=gradient_norm > 1.0,
                                gradient_nonfinite_count=nonfinite,
                                loss_nonfinite=~jnp.isfinite(loss)))
        return (eqx.apply_updates(model_params, updates), state), output

    (params, optimizer_state), losses = jax.lax.scan(step, (params, optimizer_state), (batch, weights))
    result = (eqx.combine(params, static), optimizer_state)
    if return_diagnostics:
        losses, diagnostics = losses
        return result, losses, diagnostics
    return result, losses
