"""Original distinct P/Q volatility markets and share-bounded option hedger."""
import math
from dataclasses import dataclass

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr

P_HALF = {"easy": (3., 3.), "base": (3., 3.), "hard": (40., 3.)}
Q_HALF = {"easy": (3., 3.), "base": (40., 40.), "hard": (1., 40.)}


@dataclass(frozen=True)
class Contract:
    """Original uncapped ATM call, domain-specific Q premium, and [0,1]-share trading bound."""
    strike: float = 1.
    cost: float = .0003
    max_shares: float = 1.
    payoff: str = "ordinary_call"
    premium: str = "fixed_domain_q_quote"

    def __post_init__(self):
        """Validate the declared proportional transaction cost; other conventions are frozen."""
        if not math.isfinite(self.cost) or not 0 <= self.cost < 1:
            raise ValueError("Option transaction cost must lie in [0,1)")


def variance(factors):
    """Return capped conditional variance [...] from two OU factors [...,2]."""
    result = jnp.minimum(.04*jnp.exp(jnp.minimum(jnp.sum(factors, axis=-1), 10.)), 1.)
    chex.assert_shape(result, factors.shape[:-1])
    return result


def advance(factors, noise, uniform, regime, dt, measure="P"):
    """Advance [...,2] factors using Gaussian noise, shared jump uniform [...] and P/Q law."""
    half = jnp.asarray((P_HALF if measure == "P" else Q_HALF)[regime])
    chex.assert_shape(half, (2,))
    decay = jnp.exp(-math.log(2.)/half)
    chex.assert_shape(decay, (2,))
    sd = .35*jnp.sqrt(dt*(1-decay**2)/(-2*jnp.log(decay)))
    chex.assert_shape(sd, (2,))
    jump = (uniform < (0. if regime == "easy" else 1/63))[..., None]*math.log(2.)
    chex.assert_shape(jump, (*factors.shape[:-1], 1))
    result = factors*decay + sd*noise + jump
    chex.assert_shape(result, factors.shape)
    return result


def sample_states(key, batch, config, regime, measure="P"):
    """Return issuance-to-maturity [B,T+1,3] states under the historical daily law."""
    if regime not in P_HALF or measure not in ("P", "Q"):
        raise ValueError("Unknown option regime or measure")
    gaussian_key, jump_key = jax.random.split(key)
    noise = jax.random.normal(gaussian_key, (config.n_steps, batch, 3))
    chex.assert_shape(noise, (config.n_steps, batch, 3))
    uniforms = jax.random.uniform(jump_key, (config.n_steps, batch))
    chex.assert_shape(uniforms, (config.n_steps, batch))
    initial = jnp.zeros((batch, 3)).at[:, 0].set(math.log(config.spot0))
    chex.assert_shape(initial, (batch, 3))

    def step(state, inputs):
        """Generate stock return at current variance, then OU evolution and end-day jump."""
        z, u = inputs
        var = variance(state[:, 1:])
        log_spot = state[:, 0] - .5*var*config.dt + jnp.sqrt(var*config.dt)*z[:, 2]
        chex.assert_shape(log_spot, (batch,))
        factors = advance(state[:, 1:], z[:, :2], u, regime, config.dt, measure)
        following = jnp.concatenate((log_spot[:, None], factors), axis=1)
        chex.assert_shape(following, (batch, 3))
        return following, following

    _, future = jax.lax.scan(step, initial, (noise, uniforms))
    chex.assert_shape(future, (config.n_steps, batch, 3))
    result = jnp.concatenate((initial[:, None], jnp.swapaxes(future, 0, 1)), axis=1)
    chex.assert_shape(result, (batch, config.n_steps+1, 3))
    return result


def price(key, factors, spot, strike, regime, remaining, max_steps, n_quote_paths, dt=1/252):
    """Return Q call marks [B] and MC standard errors [B], including future random arrivals.

    Inputs factors [B,2], spot [B], strike scalar, static max_steps/Q count, and
    remaining scalar may vary inside a daily scan. All paths integrate max_steps
    with the variance sum masked at remaining; conditional stock is lognormal.
    """
    batch = factors.shape[0]
    chex.assert_shape(factors, (batch, 2))
    chex.assert_shape(spot, (batch,))
    kz, ku = jax.random.split(key)
    noise = jax.random.normal(kz, (max_steps, n_quote_paths, 2))
    chex.assert_shape(noise, (max_steps, n_quote_paths, 2))
    uniform = jax.random.uniform(ku, (max_steps, n_quote_paths))
    chex.assert_shape(uniform, (max_steps, n_quote_paths))
    start = jnp.broadcast_to(factors[:, None], (batch, n_quote_paths, 2))
    chex.assert_shape(start, (batch, n_quote_paths, 2))

    def step(carry, inputs):
        """Accumulate current Q variance before the interval's OU update and future jump."""
        state, integrated = carry
        day, z, u = inputs
        increment = jnp.where(day < remaining, variance(state)*dt, 0.)
        chex.assert_shape(increment, (batch, n_quote_paths))
        integrated = integrated + increment
        following = advance(state, z[None], u[None], regime, dt, "Q")
        return (following, integrated), None

    (_, integrated), _ = jax.lax.scan(step, (start, jnp.zeros((batch, n_quote_paths))),
                                     (jnp.arange(max_steps), noise, uniform))
    chex.assert_shape(integrated, (batch, n_quote_paths))
    vol = jnp.sqrt(jnp.maximum(integrated, 1e-30))
    chex.assert_shape(vol, (batch, n_quote_paths))
    d1 = (jnp.log(spot/strike)[:, None] + .5*integrated)/vol
    chex.assert_shape(d1, (batch, n_quote_paths))
    conditional = spot[:, None]*ndtr(d1) - strike*ndtr(d1-vol)
    chex.assert_shape(conditional, (batch, n_quote_paths))
    conditional = jnp.where(remaining == 0, jax.nn.relu(spot-strike)[:, None], conditional)
    marks = jnp.mean(conditional, axis=1)
    standard_error = jnp.std(conditional, axis=1, ddof=1)/math.sqrt(n_quote_paths)
    chex.assert_shape((marks, standard_error), (batch,))
    return marks, standard_error


def atm_iv(key, factors, regime, remaining, max_steps, n_quote_paths, dt=1/252):
    """Return rolling ATM IV [B] and Q call marks [B] from conditional factors [B,2]."""
    batch = factors.shape[0]
    call, _ = price(key, factors, jnp.ones(batch), 1., regime, remaining, max_steps, n_quote_paths, dt)

    def bisect(_, bounds):
        """Bisect the normalized ATM BS call value; return lower/upper IV arrays [B]."""
        lo, hi = bounds
        middle = (lo+hi)/2
        value = 2*ndtr(.5*middle*jnp.sqrt(remaining*dt))-1
        return jnp.where(value < call, middle, lo), jnp.where(value < call, hi, middle)

    lo, hi = jax.lax.fori_loop(0, 48, bisect, (jnp.zeros(batch), jnp.full((batch,), 3.)))
    return (lo+hi)/2, call


class Hedger(eqx.Module):
    """Historical common GRU32, zero output head, share holdings in [0,1]."""
    recurrent: eqx.nn.GRUCell
    output: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)

    def __init__(self, key, hidden_size=32):
        """Initialize historical GRU/readout from key, with zero head and shared dtype."""
        a, b = jax.random.split(key)
        self.hidden_size = hidden_size
        self.recurrent = eqx.nn.GRUCell(4, hidden_size, key=a)
        self.output = eqx.nn.Linear(hidden_size, 1, key=b)
        self.output = eqx.tree_at(lambda h: (h.weight, h.bias), self.output,
                                  (jnp.zeros_like(self.output.weight), jnp.zeros_like(self.output.bias)))


def rollout(policy, log_spot, cost, premium=0.):
    """Return historical option ledger on [B,T+1] prices, including both trading endpoints."""
    batch, n_dates = log_spot.shape
    n_steps = n_dates-1
    hidden = jnp.zeros((batch, policy.hidden_size))
    holding = jnp.zeros(batch)

    def step(carry, day):
        """Trade shares using current/previous log price, remaining time and previous shares."""
        h, old = carry
        lp = log_spot[:, day]
        previous = log_spot[:, jnp.maximum(day-1, 0)]
        features = jnp.stack((lp, previous, jnp.full(batch, (n_steps-day)/n_steps), old), axis=1)
        chex.assert_shape(features, (batch, 4))
        h = jax.vmap(policy.recurrent)(features, h)
        proposed = jax.nn.sigmoid(jax.vmap(policy.output)(h)[:, 0])
        chex.assert_shape(proposed, (batch,))
        spot = jnp.exp(lp)
        following = jnp.exp(log_spot[:, day+1])
        gain = proposed*(following-spot)
        trading_cost = cost*jnp.abs(proposed-old)*spot
        return (h, proposed), (gain, trading_cost, proposed)

    (_, holding), (gain, costs, holdings) = jax.lax.scan(step, (hidden, holding), jnp.arange(n_steps))
    chex.assert_shape((gain, costs, holdings), (n_steps, batch))
    liquidation = cost*holding*jnp.exp(log_spot[:, -1])
    payoff = jax.nn.relu(jnp.exp(log_spot[:, -1])-1.)
    loss = payoff-premium-jnp.sum(gain, axis=0)+jnp.sum(costs, axis=0)+liquidation
    chex.assert_shape(loss, (batch,))
    daily_cost = costs.at[-1].add(liquidation)
    return {"loss": loss, "gains": gain.T, "daily_cost": daily_cost.T,
            "holdings": holdings.T, "cost": jnp.sum(costs, axis=0)+liquidation}
