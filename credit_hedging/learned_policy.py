"""Common recurrent target/retention policy initialized exactly at replenishment."""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp


class TargetHedger(eqx.Module):
    cell: eqx.nn.GRUCell
    head: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)
    n_vintages: int = eqx.field(static=True)
    max_protection: float = eqx.field(static=True)
    action_parameterization: str = eqx.field(static=True, default="target_retention_v1")

    def __init__(self, key, n_vintages=126, hidden_size=32, max_protection=2.0):
        """Initialize the float64 historical GRU and zero head from key and static dimensions."""
        a, b = jax.random.split(key)
        self.hidden_size = hidden_size
        self.n_vintages = n_vintages
        self.max_protection = max_protection
        self.cell = eqx.nn.GRUCell(
            4 + 3 * n_vintages, hidden_size, key=a, dtype=jnp.float64
        )
        self.head = eqx.nn.Linear(hidden_size, n_vintages + 1, key=b, dtype=jnp.float64)
        self.head = eqx.tree_at(
            lambda h: (h.weight, h.bias),
            self.head,
            (jnp.zeros_like(self.head.weight), jnp.zeros_like(self.head.bias)),
        )

    def __call__(self, features, hidden):
        """Map causal spread/book features [4+3K] and GRU state [H] to new state and actions."""
        chex.assert_shape(features, (4 + 3 * self.n_vintages,))
        chex.assert_shape(hidden, (self.hidden_size,))
        hidden = self.cell(features, hidden)
        chex.assert_shape(hidden, (self.hidden_size,))
        logits = self.head(hidden)
        chex.assert_shape(logits, (self.n_vintages + 1,))
        target = self.max_protection * jax.nn.sigmoid(logits[0])
        retain = jnp.clip(1.0 + 1.02 * jnp.tanh(logits[1:]), 0.0, 1.0)
        retained_total = jnp.sum(
            features[4 : 4 + self.n_vintages] * self.max_protection * retain
        )
        buy = jax.nn.relu(target - retained_total)
        return hidden, dict(buy=buy, retain=retain)
