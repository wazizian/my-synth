"""Continuous neural hazard proposals with exact conditional default averaging.

Like the option generator, this proposal learns neural drift, diffusion and a
risk-price term using reparameterized Euler paths. A common frozen neural-Q
pricing reference supplies differentiable interpolated bond/CDS marks. Adaptive
proposal laws are synthetic curricula, not newly calibrated economic P/Q pairs.
"""

from dataclasses import dataclass
import math

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp, ndtr

from credit_hedging.environment import book_step
from credit_hedging.market import MarketConfig, price_tables
from .dynamics import SDEPaths


@dataclass(frozen=True)
class CreditSDEConfig:
    """Continuous proposal dimensions and a separate fixed-reference pricing grid."""

    n_steps: int = 1260
    cds_days: int = 126
    grid_size: int = 601
    trading_days: int = 252
    grid_min: float = -10.
    grid_max: float = 3.
    initial_latent: float = -3.5
    drift_max: float = 2.
    volatility_min: float = .05
    volatility_max: float = .6
    risk_price_max: float = 2.

    def __post_init__(self):
        """Validate dimensions, finite ordered pricing grid and neural bounds."""
        if any(isinstance(x, bool) or not isinstance(x, int) for x in
               (self.n_steps, self.cds_days, self.grid_size, self.trading_days)):
            raise ValueError("Credit dimensions must be integers")
        if min(self.n_steps, self.cds_days, self.trading_days) < 1 or self.grid_size < 3:
            raise ValueError("Invalid credit calendar or pricing grid")
        if not all(math.isfinite(x) for x in (self.grid_min, self.grid_max,
            self.initial_latent, self.drift_max, self.volatility_min,
            self.volatility_max, self.risk_price_max)):
            raise ValueError("Credit coefficients must be finite")
        if not self.grid_min < self.initial_latent < self.grid_max:
            raise ValueError("Initial hazard factor must be inside the pricing grid")
        if min(self.drift_max, self.volatility_min, self.risk_price_max) <= 0 or self.volatility_max < self.volatility_min:
            raise ValueError("Invalid neural credit coefficient bounds")

    @property
    def state_dim(self):
        """Return the dimension of the continuous hazard factor."""
        return 1

    @property
    def dt(self):
        """Return the daily Euler step in years."""
        return 1. / self.trading_days


class NeuralCredit(eqx.Module):
    """Trainable neural hazard drift, diffusion and risk-price networks."""

    q_drift: eqx.nn.MLP
    diffusion_network: eqx.nn.MLP
    risk_price: eqx.nn.MLP

    def __init__(self, key, config, width=32, depth=2):
        """Initialize three trainable networks from a key and static architecture."""
        keys = jax.random.split(key, 3)
        self.q_drift = eqx.nn.MLP(1, 1, width, depth, activation=jax.nn.tanh, key=keys[0])
        self.diffusion_network = eqx.nn.MLP(1, 1, width, depth, activation=jax.nn.tanh, key=keys[1])
        self.risk_price = eqx.nn.MLP(1, 1, width, depth, activation=jax.nn.tanh, key=keys[2])


def coefficients(teacher, states, config, measure="P"):
    """Return bounded drift and positive volatility [batch,1] at states [batch,1]."""
    if measure not in ("P", "Q"):
        raise ValueError("Credit measure must be P or Q")
    batch = states.shape[0]
    chex.assert_shape(states, (batch, 1))
    features = jnp.tanh(states / 2.)
    chex.assert_shape(features, (batch, 1))
    drift = config.drift_max * jnp.tanh(jax.vmap(teacher.q_drift)(features))
    chex.assert_shape(drift, (batch, 1))
    volatility = config.volatility_min + (config.volatility_max-config.volatility_min) * jax.nn.sigmoid(jax.vmap(teacher.diffusion_network)(features))
    chex.assert_shape(volatility, (batch, 1))
    if measure == "P":
        risk_price = config.risk_price_max * jnp.tanh(jax.vmap(teacher.risk_price)(features))
        chex.assert_shape(risk_price, (batch, 1))
        drift = drift + volatility * risk_price
        chex.assert_shape(drift, (batch, 1))
    return drift, volatility


def sample_noise(key, batch, config):
    """Return Gaussian innovations [batch,days,1] for pathwise credit simulation."""
    noise = jax.random.normal(key, (batch, config.n_steps, 1))
    chex.assert_shape(noise, (batch, config.n_steps, 1))
    return noise


def simulate(teacher, innovations, config):
    """Return continuous Euler SDEPaths; default is integrated later, not sampled."""
    batch = innovations.shape[0]
    chex.assert_shape(innovations, (batch, config.n_steps, 1))
    initial = jnp.full((batch, 1), config.initial_latent)
    chex.assert_shape(initial, (batch, 1))
    scan_noise = jnp.swapaxes(innovations, 0, 1)
    chex.assert_shape(scan_noise, (config.n_steps, batch, 1))

    def step(state, noise):
        """Advance the continuous hazard factor with noise [batch,1]."""
        drift, volatility = coefficients(teacher, state, config)
        next_state = state + config.dt*drift + math.sqrt(config.dt)*volatility*noise
        chex.assert_shape(next_state, (batch, 1))
        row_score = -.5*(noise**2 + math.log(2*math.pi)) - jnp.log(math.sqrt(config.dt)*volatility)
        chex.assert_shape(row_score, (batch, 1))
        return next_state, (next_state, row_score[:, 0])

    _, (future, scores) = jax.lax.scan(step, initial, scan_noise)
    chex.assert_shape(future, (config.n_steps, batch, 1))
    chex.assert_shape(scores, (config.n_steps, batch))
    states = jnp.concatenate((initial[:, None, :], jnp.swapaxes(future, 0, 1)), axis=1)
    chex.assert_shape(states, (batch, config.n_steps + 1, 1))
    log_density = jnp.sum(scores, axis=0)
    chex.assert_shape(log_density, (batch,))
    times = jnp.arange(config.n_steps + 1)*config.dt
    chex.assert_shape(times, (config.n_steps + 1,))
    return SDEPaths(states, innovations, log_density, times)


def path_log_prob(teacher, paths, config):
    """Return full Gaussian Euler log density [batch] of stopped factor paths."""
    states = jax.lax.stop_gradient(paths.states)
    batch = states.shape[0]
    chex.assert_shape(states, (batch, config.n_steps + 1, 1))
    before = states[:, :-1].reshape((-1, 1))
    chex.assert_shape(before, (batch*config.n_steps, 1))
    drift, volatility = coefficients(teacher, before, config)
    drift = drift.reshape((batch, config.n_steps, 1))
    volatility = volatility.reshape((batch, config.n_steps, 1))
    chex.assert_shape((drift, volatility), (batch, config.n_steps, 1))
    residual = (states[:, 1:] - states[:, :-1] - config.dt*drift) / (math.sqrt(config.dt)*volatility)
    chex.assert_shape(residual, (batch, config.n_steps, 1))
    rows = -.5*(residual**2 + math.log(2*math.pi)) - jnp.log(math.sqrt(config.dt)*volatility)
    chex.assert_shape(rows, (batch, config.n_steps, 1))
    result = jnp.sum(rows, axis=(1, 2))
    chex.assert_shape(result, (batch,))
    return result


def market_contract(config):
    """Return the existing long-ZCB/rolling-CDS contract on the same calendar."""
    return MarketConfig(bond_days=config.n_steps, cds_days=config.cds_days,
        trading_days=config.trading_days, grid_size=config.grid_size,
        grid_min=config.grid_min, grid_max=config.grid_max, initial_latent=config.initial_latent)


def pricing_tables(reference, config, contract):
    """Build approximate fixed-reference neural-Q bond and CDS pricing tables.

    Gaussian Euler transitions are binned on a pricing grid. Continuous paths
    are never quantized; marks are interpolated from these tables. This is an
    approximation requiring grid refinement and boundary/martingale checks.
    """
    if (contract.bond_days != config.n_steps or contract.cds_days != config.cds_days
            or contract.grid_size != config.grid_size or contract.trading_days != config.trading_days
            or contract.grid_min != config.grid_min or contract.grid_max != config.grid_max
            or contract.initial_latent != config.initial_latent):
        raise ValueError("Continuous credit contract and pricing grid/calendar must agree")
    n_states = config.grid_size
    grid = jnp.linspace(config.grid_min, config.grid_max, n_states)
    chex.assert_shape(grid, (n_states,))
    drift, volatility = coefficients(reference, grid[:, None], config, "Q")
    means = grid + config.dt*drift[:, 0]
    chex.assert_shape(means, (n_states,))
    midpoints = .5*(grid[1:] + grid[:-1])
    chex.assert_shape(midpoints, (n_states-1,))
    interior = (midpoints[None, :] - means[:, None]) / (math.sqrt(config.dt)*volatility)
    chex.assert_shape(interior, (n_states, n_states-1))
    standardized = jnp.concatenate((jnp.full((n_states, 1), -12.),
        jnp.clip(interior, -12., 12.), jnp.full((n_states, 1), 12.)), axis=1)
    chex.assert_shape(standardized, (n_states, n_states+1))
    kernel = jnp.maximum(jnp.diff(ndtr(standardized), axis=1), 1e-12)
    chex.assert_shape(kernel, (n_states, n_states))
    kernel = kernel / jnp.sum(kernel, axis=1, keepdims=True)
    chex.assert_shape(kernel, (n_states, n_states))
    hazard = jax.nn.softplus(grid)
    survival = jnp.exp(-config.dt*hazard)
    chex.assert_shape((hazard, survival), (n_states,))
    return price_tables(grid, hazard, survival, kernel, contract)


def quotes_at(states, remaining, tables, config):
    """Interpolate bond [batch] and CDS legs [batch,tenor+1] at factor states."""
    batch = states.shape[0]
    chex.assert_shape(states, (batch, 1))
    clipped = jnp.clip(states[:, 0], config.grid_min, config.grid_max)
    chex.assert_shape(clipped, (batch,))
    index = jnp.clip(jnp.searchsorted(tables['grid'], clipped)-1, 0, config.grid_size-2)
    chex.assert_shape(index, (batch,))
    fraction = (clipped-tables['grid'][index]) / (tables['grid'][index+1]-tables['grid'][index])
    chex.assert_shape(fraction, (batch,))
    bond = (1-fraction)*tables['bond'][remaining, index] + fraction*tables['bond'][remaining, index+1]
    chex.assert_shape(bond, (batch,))
    protection = (1-fraction[:, None])*tables['protection'][:config.cds_days+1, index].T + fraction[:, None]*tables['protection'][:config.cds_days+1, index+1].T
    annuity = (1-fraction[:, None])*tables['annuity'][:config.cds_days+1, index].T + fraction[:, None]*tables['annuity'][:config.cds_days+1, index+1].T
    chex.assert_shape((protection, annuity), (batch, config.cds_days+1))
    return dict(bond=bond, protection=protection, annuity=annuity)


def default_branches(policy, paths, tables, contract, config):
    """Return losses/probabilities for all default dates plus survival per factor path.

    Inputs are a policy PyTree (or None), continuous factor paths [B,T+1,1],
    fixed pricing tables and static contract/config. One causal alive-prefix
    strategy suffices: each day settles the same action under default and
    survival, retaining only the survival book for the next day. This integrates
    Cox default exactly conditional on the Euler factor path, in O(B*T*tenor).
    Outputs include [B,T+1] event loss/log-probability and daily moments.
    """
    states = paths.states
    batch, n_dates, dim = states.shape
    n_time, n_vintages = config.n_steps, config.cds_days
    chex.assert_shape(states, (batch, n_time+1, 1))
    book = dict(notional=jnp.zeros((batch, n_vintages)), coupon=jnp.zeros((batch, n_vintages)))
    chex.assert_shape((book['notional'], book['coupon']), (batch, n_vintages))
    hidden_size = policy.hidden_size if policy is not None else 1
    hidden = jnp.zeros((batch, hidden_size))
    chex.assert_shape(hidden, (batch, hidden_size))
    first = quotes_at(states[:, 0], n_time, tables, config)
    initial_price = first['bond']
    spread = first['protection'][:, n_vintages]/first['annuity'][:, n_vintages]
    chex.assert_shape((initial_price, spread), (batch,))
    remaining = n_vintages-jnp.arange(n_vintages)
    chex.assert_shape(remaining, (n_vintages,))

    def step(carry, day):
        """Advance alive-prefix book and record default loss and two daily errors."""
        book, hidden, previous_spread, cash = carry
        current = quotes_at(states[:, day], n_time-day, tables, config)
        following = quotes_at(states[:, day+1], n_time-day-1, tables, config)
        spread = current['protection'][:, n_vintages]/current['annuity'][:, n_vintages]
        chex.assert_shape(spread, (batch,))
        if policy is None:
            action = dict(buy=jnp.zeros((batch,)), retain=jnp.ones((batch, n_vintages)))
        else:
            marks = current['protection'][:, remaining]-book['coupon']*current['annuity'][:, remaining]
            chex.assert_shape(marks, (batch, n_vintages))
            marks = jnp.where(book['notional'] > 0, marks, 0.)
            chex.assert_shape(marks, (batch, n_vintages))
            globals_ = jnp.stack((100*spread, 100*(spread-previous_spread),
                jnp.full((batch,), (n_time-day)/n_time),
                jnp.sum(book['notional'], axis=1)/contract.max_protection), axis=1)
            chex.assert_shape(globals_, (batch, 4))
            features = jnp.concatenate((globals_, book['notional']/contract.max_protection,
                100*book['coupon'], 100*marks), axis=1)
            chex.assert_shape(features, (batch, 4+3*n_vintages))
            hidden, action = jax.vmap(policy)(features, hidden)
            chex.assert_shape(hidden, (batch, hidden_size))
        quotes = current | {'next_bond': following['bond'],
            'next_protection': following['protection'], 'next_annuity': following['annuity']}
        market = dict(day=day, state=jnp.zeros((batch,), dtype=jnp.int32),
            next_state=jnp.zeros((batch,), dtype=jnp.int32), alive=jnp.ones((batch,), dtype=bool),
            default=jnp.zeros((batch,), dtype=bool), quotes=quotes)
        next_book, survived = book_step(book, action, market, tables, contract)
        _, defaulted = book_step(book, action, market | {'default': jnp.ones((batch,), dtype=bool)}, tables, contract)
        default_loss = initial_price-cash-defaulted['discounted_cashflow']
        chex.assert_shape(default_loss, (batch,))
        cash = cash+survived['discounted_cashflow']
        chex.assert_shape(cash, (batch,))
        return (next_book, hidden, spread, cash), (default_loss,
            survived['daily_loss'], defaulted['daily_loss'])

    days = jnp.arange(n_time)
    chex.assert_shape(days, (n_time,))
    (_, _, _, final_cash), outputs = jax.lax.scan(jax.checkpoint(step),
        (book, hidden, spread, jnp.zeros((batch,))), days)
    default_loss, daily_survived, daily_defaulted = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), outputs)
    chex.assert_shape((default_loss, daily_survived, daily_defaulted), (batch, n_time))
    event_loss = jnp.concatenate((default_loss, (initial_price-final_cash)[:, None]), axis=1)
    chex.assert_shape(event_loss, (batch, n_time+1))
    log_survival = -config.dt*jax.nn.softplus(states[:, :-1, 0])
    chex.assert_shape(log_survival, (batch, n_time))
    cumulative = jnp.cumsum(log_survival, axis=1)
    chex.assert_shape(cumulative, (batch, n_time))
    previous = jnp.concatenate((jnp.zeros((batch, 1)), cumulative[:, :-1]), axis=1)
    chex.assert_shape(previous, (batch, n_time))
    log_default = previous+jnp.log(-jnp.expm1(log_survival))
    chex.assert_shape(log_default, (batch, n_time))
    log_probability = jnp.concatenate((log_default, cumulative[:, -1:]), axis=1)
    chex.assert_shape(log_probability, (batch, n_time+1))
    default_probability = jnp.exp(log_default)
    survival_probability = jnp.exp(cumulative)
    chex.assert_shape((default_probability, survival_probability), (batch, n_time))
    daily_mean = jnp.mean(default_probability*daily_defaulted + survival_probability*daily_survived)
    daily_second = jnp.mean(default_probability*daily_defaulted**2 + survival_probability*daily_survived**2)
    chex.assert_shape((daily_mean, daily_second), ())
    return dict(event_loss=event_loss, log_probability=log_probability,
        daily_error_std=jnp.sqrt(jnp.maximum(daily_second-daily_mean**2, 0.)),
        quote_boundary_fraction=jnp.mean((states <= config.grid_min) | (states >= config.grid_max)))


def terminal_losses(policy, paths, tables, contract, config, gamma=10.):
    """Return conditional entropic losses [B], exactly integrating default dates.

    An outer entropic aggregation over these values equals entropic risk over
    factor paths AND defaults. They are certainty equivalents, not raw losses;
    CVaR and standard deviation must instead use weighted event losses below.
    """
    result = default_branches(policy, paths, tables, contract, config)
    losses = logsumexp(result['log_probability']+gamma*result['event_loss'], axis=1)/gamma
    chex.assert_shape(losses, (paths.states.shape[0],))
    return losses


def feedback_metrics(policy, paths, tables, contract, config, gamma=10.):
    """Return raw-event weighted risks and conditional losses for ESS diagnostics."""
    result = default_branches(policy, paths, tables, contract, config)
    batch = paths.states.shape[0]
    conditional = logsumexp(result['log_probability']+gamma*result['event_loss'], axis=1)/gamma
    chex.assert_shape(conditional, (batch,))
    losses = result['event_loss'].reshape((-1,))
    weights = jnp.exp(result['log_probability']).reshape((-1,))/batch
    size = batch*(config.n_steps+1)
    chex.assert_shape((losses, weights), (size,))
    weights = weights/jnp.sum(weights)
    chex.assert_shape(weights, (size,))
    order = jnp.argsort(losses)[::-1]
    chex.assert_shape(order, (size,))
    ordered_weights = weights[order]
    chex.assert_shape(ordered_weights, (size,))
    preceding = jnp.cumsum(ordered_weights)-ordered_weights
    chex.assert_shape(preceding, (size,))
    tail_weights = jnp.minimum(ordered_weights, jnp.maximum(.05-preceding, 0.))
    chex.assert_shape(tail_weights, (size,))
    mean = jnp.sum(weights*losses)
    chex.assert_shape(mean, ())
    metrics = dict(entropic=(logsumexp(gamma*conditional)-math.log(batch))/gamma,
        cvar95=jnp.sum(tail_weights*losses[order])/.05,
        mean_loss=mean, terminal_std=jnp.sqrt(jnp.sum(weights*(losses-mean)**2)),
        daily_error_std=result['daily_error_std'],
        quote_boundary_fraction=result['quote_boundary_fraction'])
    for value in metrics.values():
        chex.assert_shape(value, ())
    return conditional, metrics
