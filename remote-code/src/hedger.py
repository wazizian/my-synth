"""Causal recurrent policy; no simulator regime or latent state is observed."""
import equinox as eqx
import jax
import jax.numpy as jnp

FEATURE_SIZE = 14


class Hedger(eqx.Module):
    cell: eqx.nn.GRUCell
    head: eqx.nn.MLP
    hidden_size: int = eqx.field(static=True)
    position_bound: float = eqx.field(static=True)

    def __init__(self, key, hidden_size=64, position_bound=2.0):
        a, b = jax.random.split(key)
        self.hidden_size = hidden_size
        self.position_bound = position_bound
        self.cell = eqx.nn.GRUCell(FEATURE_SIZE, hidden_size, key=a)
        self.head = eqx.nn.MLP(hidden_size, 2, width_size=32, depth=1,
                               activation=jax.nn.tanh, key=b)

    def __call__(self, features, hidden):
        hidden = self.cell(features, hidden)
        return hidden, self.position_bound * jnp.tanh(self.head(hidden))


def features(spot, call, prev_spot, prev_call, reference, inception_spot,
             date, inception, intervals, previous_holdings, trading):
    """Fixed schema; contract/liability quantities absent in context are zero."""
    active = jnp.asarray(trading, dtype=spot.dtype)
    return jnp.concatenate((jnp.stack([
        jnp.log(spot / reference), jnp.log(spot / prev_spot),
        call / reference, (call - prev_call) / reference,
        date / 252.0, (inception + 60 - date) / 252.0,
        active, active * jnp.log(spot / inception_spot),
        active * (inception + intervals - date) / 252.0,
        active * (1.05 / reference), active * (inception_spot / reference),
        active * (60.0 / 252.0),
    ]), active * previous_holdings))


def actions_one(hedger, spot, call, inception=64, intervals=20):
    """Observe context dates 0..63, then trade at dates 64..83."""
    reference = spot[0]
    initial = (jnp.zeros(hedger.hidden_size, dtype=spot.dtype),
               jnp.zeros(2, dtype=spot.dtype))

    def step(carry, date):
        hidden, previous = carry
        prev_date = jnp.maximum(date - 1, 0)
        trading = date >= inception
        x = features(spot[date], call[date], spot[prev_date], call[prev_date],
                     reference, spot[jnp.minimum(date, inception)], date, inception, intervals,
                     previous, trading)
        hidden, proposed = hedger(x, hidden)
        holdings = jnp.where(trading, proposed, jnp.zeros_like(proposed))
        return (hidden, holdings), holdings

    _, positions = jax.lax.scan(step, initial, jnp.arange(inception + intervals))
    return positions[inception:]
