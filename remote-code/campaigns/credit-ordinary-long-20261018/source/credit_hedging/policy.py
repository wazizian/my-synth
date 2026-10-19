"""Shared recurrent architecture producing new protection and vintage-specific retention."""
import chex
import equinox as eqx
import jax
import jax.numpy as jnp


class Hedger(eqx.Module):
    cell: eqx.nn.GRUCell
    head: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)
    n_vintages: int = eqx.field(static=True)
    max_protection: float = eqx.field(static=True)
    action_parameterization: str = eqx.field(static=True)

    def __init__(self, key, n_vintages=126, hidden_size=32, max_protection=2.0,
                 action_parameterization='clipped_stretched_sigmoid_v2'):
        """Build a GRU/linear policy from a PRNG key and static dimensions; returns an initialized PyTree."""
        if action_parameterization not in ('sigmoid_v1', 'clipped_stretched_sigmoid_v2'):
            raise ValueError('Unknown action parameterization')
        self.hidden_size = hidden_size
        self.n_vintages = n_vintages
        self.max_protection = max_protection
        self.action_parameterization = action_parameterization
        cell_key, head_key = jax.random.split(key)
        self.cell = eqx.nn.GRUCell(4+3*n_vintages, hidden_size, key=cell_key)
        self.head = eqx.nn.Linear(hidden_size, n_vintages+1, key=head_key)
        # Start inside the active region; exact zero/one remains reachable at finite logits.
        initial_bias = 5.0 if action_parameterization == 'sigmoid_v1' else 4.0
        bias = jnp.concatenate((jnp.array([-initial_bias]), jnp.full(n_vintages, initial_bias)))
        chex.assert_shape(bias, (n_vintages+1,))
        self.head = eqx.tree_at(lambda head: head.bias, self.head, bias)

    def __call__(self, features, hidden):
        """Map one causal observation and recurrent state to next state and buy/retain actions."""
        n_features = 4+3*self.n_vintages
        chex.assert_shape(features, (n_features,))
        chex.assert_shape(hidden, (self.hidden_size,))
        hidden = self.cell(features, hidden)
        chex.assert_shape(hidden, (self.hidden_size,))
        logits = self.head(hidden)
        chex.assert_shape(logits, (self.n_vintages+1,))
        fractions = jax.nn.sigmoid(logits)
        chex.assert_shape(fractions, (self.n_vintages+1,))
        if self.action_parameterization == 'clipped_stretched_sigmoid_v2':
            fractions = jnp.clip(1.02*fractions - 0.01, 0, 1)
            chex.assert_shape(fractions, (self.n_vintages+1,))
        buy = self.max_protection * fractions[0]
        retain = fractions[1:]
        chex.assert_shape(buy, ())
        chex.assert_shape(retain, (self.n_vintages,))
        return hidden, dict(buy=buy, retain=retain)
