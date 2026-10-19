"""Two-factor P/Q volatility experiment; run python -m volatility_dynamics.demo."""
from pathlib import Path
import json
import chex
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr
import numpy as np

DAYS = 140
SHOCK_DAY = 35
DT = 1 / 252
BASE_VOL = 0.20
ETA = 0.35
N_PATHS = 128
Q_PATHS = 256
TENOR = 21
WINDOW = 5


def simulate(key, half_lives, shock):
    """Simulate P log-vol factors and stock; return states, prices and returns for 128 paths."""
    batch, n_time, n_factors = N_PATHS, DAYS, 2
    half_lives = jnp.asarray(half_lives)
    chex.assert_shape(half_lives, (n_factors,))
    decay = jnp.exp(-jnp.log(2.) / half_lives)
    chex.assert_shape(decay, (n_factors,))
    innovation_sd = ETA * jnp.sqrt(DT * (1 - decay**2) / (-2 * jnp.log(decay)))
    chex.assert_shape(innovation_sd, (n_factors,))
    noise = jax.random.normal(key, (n_time, batch, 3))
    chex.assert_shape(noise, (n_time, batch, 3))

    def step(state, inputs):
        """Advance factors and stock using one daily innovation; output end states and return."""
        x, log_s = state
        day, z = inputs
        jump = jnp.where(day == SHOCK_DAY, shock, 0.)
        x = x + jump
        chex.assert_shape(x, (batch, n_factors))
        variance = BASE_VOL**2 * jnp.exp(jnp.sum(x, axis=-1))
        chex.assert_shape(variance, (batch,))
        ret = -0.5 * variance * DT + jnp.sqrt(variance * DT) * z[:, 2]
        chex.assert_shape(ret, (batch,))
        next_x = x * decay + innovation_sd * z[:, :2]
        chex.assert_shape(next_x, (batch, n_factors))
        next_s = log_s + ret
        chex.assert_shape(next_s, (batch,))
        return (next_x, next_s), (x, next_s, ret, variance)

    _, outputs = jax.lax.scan(step, (jnp.zeros((batch, 2)), jnp.zeros(batch)),
                              (jnp.arange(n_time), noise))
    x, log_s, returns, variance = outputs
    chex.assert_shape(x, (n_time, batch, 2))
    chex.assert_shape(log_s, (n_time, batch))
    chex.assert_shape(returns, (n_time, batch))
    chex.assert_shape(variance, (n_time, batch))
    # x is the start-of-day state: quotes and rolling RV below use start-of-day information.
    start_log_s = jnp.concatenate((jnp.zeros((1, batch)), log_s[:-1]), axis=0)
    chex.assert_shape(start_log_s, (n_time, batch))
    prices = jnp.exp(start_log_s)
    chex.assert_shape(prices, (n_time, batch))
    return x, prices, returns, variance


def option_iv(key, states, half_lives):
    """Price rolling ATM 21-day calls under Q and invert BS; input [time,batch,2], output IV."""
    n_time, batch, n_factors = states.shape
    chex.assert_shape(states, (n_time, batch, 2))
    decay = jnp.exp(-jnp.log(2.) / jnp.asarray(half_lives))
    chex.assert_shape(decay, (n_factors,))
    sd = ETA * jnp.sqrt(DT * (1 - decay**2) / (-2 * jnp.log(decay)))
    chex.assert_shape(sd, (n_factors,))
    # Common Q draws across dates/paths/regimes reduce Monte Carlo comparison noise.
    noise = jax.random.normal(key, (TENOR, Q_PATHS, n_factors))
    chex.assert_shape(noise, (TENOR, Q_PATHS, n_factors))
    initial = jnp.broadcast_to(states[:, :, None, :], (n_time, batch, Q_PATHS, n_factors))
    chex.assert_shape(initial, (n_time, batch, Q_PATHS, n_factors))

    def step(carry, z):
        """Accumulate future Q variance conditional on today's factors; return no trajectory."""
        x, integrated = carry
        v = BASE_VOL**2 * jnp.exp(jnp.sum(x, axis=-1))
        chex.assert_shape(v, (n_time, batch, Q_PATHS))
        integrated = integrated + v * DT
        chex.assert_shape(integrated, (n_time, batch, Q_PATHS))
        x = x * decay + sd * z[None, None, :, :]
        chex.assert_shape(x, (n_time, batch, Q_PATHS, n_factors))
        return (x, integrated), None

    (_, integrated), _ = jax.lax.scan(step, (initial, jnp.zeros((n_time, batch, Q_PATHS))), noise)
    chex.assert_shape(integrated, (n_time, batch, Q_PATHS))
    # Independent stock/vol Brownian motions: conditional terminal stock is lognormal.
    conditional_calls = 2 * ndtr(0.5 * jnp.sqrt(integrated)) - 1
    chex.assert_shape(conditional_calls, (n_time, batch, Q_PATHS))
    call = jnp.mean(conditional_calls, axis=-1)
    chex.assert_shape(call, (n_time, batch))

    def bisect(_, bounds):
        """Bisect normalized ATM BS call to obtain annualized implied volatility."""
        low, high = bounds
        mid = (low + high) / 2
        chex.assert_shape(mid, (n_time, batch))
        value = 2 * ndtr(0.5 * mid * jnp.sqrt(TENOR * DT)) - 1
        chex.assert_shape(value, (n_time, batch))
        return jnp.where(value < call, mid, low), jnp.where(value < call, high, mid)

    low, high = jax.lax.fori_loop(0, 45, bisect, (jnp.zeros_like(call), jnp.full_like(call, 3.)))
    iv = (low + high) / 2
    chex.assert_shape(iv, (n_time, batch))
    return iv, call


def main():
    """Generate matched regime simulations, raw arrays, validation and PNG/PDF figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    jax.config.update('jax_enable_x64', True)
    output = Path(__file__).parent / 'results'
    output.mkdir(exist_ok=True)
    simulate_jit = jax.jit(simulate)
    iv_jit = jax.jit(option_iv)
    p_key, q_key = jax.random.split(jax.random.PRNGKey(20261017))
    scenarios = {
        'Easy: no shock': ((3., 3.), (3., 3.), (0., 0.)),
        'Base: fast shock': ((3., 3.), (40., 40.), (np.log(2.), np.log(2.))),
        'Hard: persistent shock': ((40., 3.), (1., 40.), (np.log(2.), np.log(2.))),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    arrays, summary = {}, {}
    for ax, (name, (p_half, q_half, jump)) in zip(axes, scenarios.items()):
        states, prices, returns, variance = simulate_jit(p_key, jnp.array(p_half), jnp.array(jump))
        iv, call = iv_jit(q_key, states, jnp.array(q_half))
        returns, iv, variance = map(np.asarray, (returns, iv, variance))
        # RV(t) uses returns strictly before t, making it observable at the quote date.
        rv = np.full_like(returns, np.nan)
        for day in range(WINDOW, DAYS):
            rv[day] = np.sqrt(252 * np.mean(returns[day-WINDOW:day]**2, axis=0))
        realized = np.full(DAYS, np.nan)
        realized[WINDOW:] = np.sqrt(np.mean(rv[WINDOW:]**2, axis=1))
        quoted = np.mean(iv, axis=1)
        latent = np.sqrt(np.mean(variance, axis=1))
        time = np.arange(DAYS) - SHOCK_DAY
        ax.plot(time, 100 * realized, label='5-day RV: pooled RMS', color='#c65a12', lw=2)
        ax.plot(time, 100 * quoted, label='21-day ATM call IV: mean', color='#2468b4', lw=2)
        ax.plot(time, 100 * latent, label='Spot volatility: pooled RMS', color='#777777', ls=':', lw=1.4)
        lo, hi = np.quantile(rv, [0.1, 0.9], axis=1)
        ax.fill_between(time, 100*lo, 100*hi, color='#c65a12', alpha=.10, label='RV path 10–90%')
        ax.axvline(0, color='black', lw=.8, alpha=.5)
        ax.set(title=name, xlabel='Trading days from imposed shock', xlim=(-20, 85), ylim=(10, 65))
        ax.grid(alpha=.2)
        slug = name.split(':')[0].lower()
        arrays.update({f'{slug}_states': np.asarray(states), f'{slug}_prices': np.asarray(prices),
                       f'{slug}_returns': returns, f'{slug}_rv': rv, f'{slug}_iv': iv,
                       f'{slug}_variance': variance, f'{slug}_call': np.asarray(call)})
        summary[slug] = {str(d): {'rv': float(realized[SHOCK_DAY+d]), 'iv': float(quoted[SHOCK_DAY+d]),
                                  'spot_vol': float(latent[SHOCK_DAY+d])} for d in [0, 5, 15, 30, 60]}
        assert np.all(np.isfinite(iv)) and np.all(iv > 0)
        reconstructed = 2 * np.asarray(ndtr(.5*jnp.asarray(iv)*np.sqrt(TENOR*DT))) - 1
        assert np.max(np.abs(reconstructed - np.asarray(call))) < 1e-12
    # Economic pattern checks use predetermined observation dates, not selected favorable paths.
    assert summary['base']['15']['rv'] < summary['base']['5']['rv']
    assert summary['base']['15']['iv'] < summary['base']['5']['iv']
    assert summary['hard']['30']['rv'] > summary['base']['30']['rv'] + .04
    assert summary['hard']['30']['rv'] > summary['hard']['30']['iv'] + .03
    assert summary['hard']['15']['iv'] < summary['hard']['0']['iv'] - .04
    axes[0].set_ylabel('Annualized volatility (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    fig.suptitle('Two-factor volatility: physical persistence versus option pricing', fontsize=14)
    fig.tight_layout(rect=(0, .10, 1, .94))
    fig.savefig(output/'rv_iv.png', dpi=180)
    fig.savefig(output/'rv_iv.pdf')
    # One matched price realization shows the observations available to the hedger.
    fig2, ax2 = plt.subplots(figsize=(9, 3.6))
    for slug, label in [('easy','Easy'),('base','Base'),('hard','Hard')]:
        ax2.plot(np.arange(DAYS)-SHOCK_DAY, arrays[f'{slug}_prices'][:, 0], label=label)
    ax2.axvline(0, color='black', lw=.8)
    ax2.set(xlabel='Trading days from shock', ylabel='Underlying price / initial price', title='One matched price path; hedger observes prices only')
    ax2.legend(); ax2.grid(alpha=.2); fig2.tight_layout()
    fig2.savefig(output/'price_paths.png', dpi=180)
    np.savez_compressed(output/'paths.npz', **arrays)
    metadata = {'seed':20261017,'n_paths':N_PATHS,'q_paths':Q_PATHS,'tenor_days':TENOR,
                'rv_window_days':WINDOW,'eta':ETA,'base_vol':BASE_VOL,'shock_day':SHOCK_DAY,
                'scenarios':scenarios,'summary':summary,'checks':'passed'}
    (output/'summary.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(summary, indent=2))
    print('All pattern and BS inversion checks passed. Figures:', output)


if __name__ == '__main__':
    main()
