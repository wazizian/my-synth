"""Shared history-only GRU hedger and explicit option cash accounting."""

from dataclasses import dataclass
import math

import chex
import equinox as eqx
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Contract:
    """Financed short call; cap=None preserves an ordinary unbounded call.

    An explicit finite payoff cap and dollar holding cap bound losses above
    over a finite daily grid, allowing finite population entropic risk.
    Ordinary call experiments remain available, but have no such guarantee.
    """

    strike: float = 1.0
    premium: float = 0.0
    cost: float = 0.0003
    dollar_cap: float = 1.0
    payoff_cap: float | None = None

    def __post_init__(self):
        """Validate finite positive contract scales; return nothing or raise."""
        if not all(math.isfinite(x) for x in (self.strike, self.premium, self.cost, self.dollar_cap)):
            raise ValueError("Contract fields must be finite")
        if self.strike <= 0 or not 0 <= self.cost < 1 or self.dollar_cap <= 0:
            raise ValueError("Invalid strike, cost or dollar cap")
        if self.payoff_cap is not None and (not math.isfinite(self.payoff_cap) or self.payoff_cap <= 0):
            raise ValueError("Payoff cap must be finite positive or None")

    def loss_upper_bound(self, n_steps):
        """Return analytic upper loss bound for n_steps trades; return infinity for an ordinary call."""
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError("n_steps must be a positive integer")
        if self.payoff_cap is None:
            return math.inf
        # A buy increases stock dollar inventory by at most dollar_cap.
        # Sales and final liquidation have nonnegative proceeds when cost<1.
        return self.payoff_cap - self.premium + n_steps * (1 + self.cost) * self.dollar_cap


class Hedger(eqx.Module):
    """Identical recurrent architecture for every baseline/specialist/teacher arm."""

    recurrent: eqx.nn.GRUCell
    output: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)

    def __init__(self, key, hidden_size=32):
        """Initialize GRU and readout from key; input is prices/time/own holding."""
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        recurrent_key, output_key = jax.random.split(key)
        self.recurrent = eqx.nn.GRUCell(4, hidden_size, key=recurrent_key)
        self.output = eqx.nn.Linear(hidden_size, 1, key=output_key)
        self.hidden_size = hidden_size


def terminal_losses(model, paths, contract):
    """Map shared model and joint SDE paths to terminal losses [batch].

    Only log-price history, time and the previous dollar holding reach the
    GRU. Latent factors and teacher identity never reach its feature vector.
    Cash pays transaction costs at opening, all changes and final liquidation.
    """
    log_spot = paths.log_spot
    batch, n_time = log_spot.shape
    chex.assert_shape(log_spot, (batch, n_time))
    if n_time < 2:
        raise ValueError("Hedging needs an initial observation and at least one later date")
    spots = jnp.exp(log_spot)
    chex.assert_shape(spots, (batch, n_time))
    hidden = jnp.zeros((batch, model.hidden_size), dtype=log_spot.dtype)
    chex.assert_shape(hidden, (batch, model.hidden_size))
    zeros = jnp.zeros((batch,), dtype=log_spot.dtype)
    chex.assert_shape(zeros, (batch,))

    def step(carry, index):
        """Advance hidden/cash/shares at a daily index; return carry and cost."""
        previous_hidden, cash, shares = carry
        price = spots[:, index]
        chex.assert_shape(price, (batch,))
        previous_index = jnp.maximum(index - 1, 0)
        features = jnp.stack((log_spot[:, index],
                              log_spot[:, index] - log_spot[:, previous_index],
                              jnp.full((batch,), index / (n_time - 1)),
                              shares * price / contract.dollar_cap), axis=-1)
        chex.assert_shape(features, (batch, 4))
        next_hidden = jax.vmap(model.recurrent)(features, previous_hidden)
        chex.assert_shape(next_hidden, (batch, model.hidden_size))
        allocation = jax.nn.sigmoid(jax.vmap(model.output)(next_hidden)[:, 0])
        chex.assert_shape(allocation, (batch,))
        next_shares = contract.dollar_cap * allocation / price
        chex.assert_shape(next_shares, (batch,))
        traded_cash = (next_shares - shares) * price
        chex.assert_shape(traded_cash, (batch,))
        cost = contract.cost * jnp.abs(traded_cash)
        chex.assert_shape(cost, (batch,))
        cash = cash - traded_cash - cost
        chex.assert_shape(cash, (batch,))
        return (next_hidden, cash, next_shares), cost

    (_, cash, shares), _ = jax.lax.scan(step, (hidden, zeros + contract.premium, zeros), jnp.arange(n_time - 1))
    liquidation = shares * spots[:, -1]
    chex.assert_shape(liquidation, (batch,))
    cash = cash + liquidation - contract.cost * jnp.abs(liquidation)
    chex.assert_shape(cash, (batch,))
    payoff = jax.nn.relu(spots[:, -1] - contract.strike)
    chex.assert_shape(payoff, (batch,))
    if contract.payoff_cap is not None:
        payoff = jnp.minimum(payoff, contract.payoff_cap)
        chex.assert_shape(payoff, (batch,))
    losses = payoff - cash
    chex.assert_shape(losses, (batch,))
    return losses
