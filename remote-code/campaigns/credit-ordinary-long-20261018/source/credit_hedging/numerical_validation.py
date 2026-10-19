"""GPU-only full-calendar price refinement, without storing unused state prices."""
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .curve_market import CurveConfig, ou_transition


@eqx.filter_jit
def sampled_prices(config):
    """Independent maturity recursion; retain a fixed interior probe lattice.

    All 1260 maturities are checked on this lattice. These checks are not a
    uniform error bound over every latent state or continuous time.
    """
    dt = 1/config.trading_days
    x = jnp.linspace(config.grid_min, config.grid_max, config.grid_size)
    y = (jnp.array([config.fast_mean]) if config.fast_grid_size == 1 else
         jnp.linspace(config.fast_grid_min, config.fast_grid_max, config.fast_grid_size))
    px = ou_transition(x, config.mean_reversion, config.latent_mean, config.latent_vol, dt)
    py = ou_transition(y, config.fast_reversion, config.fast_mean, config.fast_vol, dt)
    sx = config.latent_mean+jnp.array([-1., -.5, 0., .5, 1.])
    sy = jnp.array([0.]) if len(y) == 1 else jnp.array([-.4, -.2, 0., .2, .4])
    ix = jnp.argmin(jnp.abs(x[:, None]-sx), axis=0)
    iy = jnp.argmin(jnp.abs(y[:, None]-sy), axis=0)
    survival = jnp.exp(-jax.nn.softplus(x[:, None]+y[None, :])*dt)
    discount = jnp.exp(-config.rate*dt)
    cash = jnp.stack(((1-survival)*config.recovery,
                      (1-survival)*(1-config.recovery), jnp.full_like(survival, dt)))
    initial = jnp.stack((jnp.ones_like(survival), jnp.zeros_like(survival), jnp.zeros_like(survival)))

    def step(previous, unused):
        expected = jax.vmap(lambda a: px @ a @ py.T)(previous)
        current = discount*(cash+survival[None]*expected)
        selected = current[:, ix[:, None], iy[None, :]]
        return current, selected

    _, selected = jax.lax.scan(step, initial, None, length=config.bond_days)
    x0 = jnp.argmin(jnp.abs(x-config.initial_latent))
    y0 = jnp.argmin(jnp.abs(y-config.fast_initial))

    def moments(grid, p, i, kappa, sigma):
        mean = p[i]@grid
        var = p[i]@((grid-mean)**2)
        exact = sigma**2*(-jnp.expm1(-2*kappa*dt))/(2*kappa)
        return jnp.array([mean, var, exact, (var-exact)/jnp.maximum(exact, 1e-30), p[i, 0]+p[i, -1]])

    return dict(prices=selected, x=x[ix], y=y[iy],
                x_moments=moments(x, px, x0, config.mean_reversion, config.latent_vol),
                y_moments=moments(y, py, y0, config.fast_reversion, config.fast_vol))


def run(design_path, output):
    assert jax.default_backend() == 'gpu', 'Numerical work requires GPU worker'
    jax.config.update('jax_enable_x64', True)
    design = json.loads(Path(design_path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    tolerance = design['thresholds']['pricing_max_error']
    for domain, parameters in design['components'].items():
        base = CurveConfig(**(design['common'] | parameters | design['calendar']))
        records = []
        previous = None
        previous_config = None
        accepted = None
        # Numerical refinement only: economic parameters and gates stay frozen.
        for multiplier in (1, 2, 4, 8):
            cfg = replace(base, grid_size=(base.grid_size-1)*multiplier+1,
                          fast_grid_size=(base.fast_grid_size-1)*multiplier+1)
            started = time.monotonic()
            current = jax.device_get(sampled_prices(cfg))
            np.savez_compressed(output/f'{domain}-grid-{multiplier}.npz', **current)
            record = dict(multiplier=multiplier, x_nodes=cfg.grid_size, y_nodes=cfg.fast_grid_size,
                          seconds=time.monotonic()-started,
                          x_moments=current['x_moments'].tolist(), y_moments=current['y_moments'].tolist())
            if not all(np.isfinite(v).all() for v in current.values()):
                raise RuntimeError('Nonfinite numerical reference')
            if previous is not None:
                # Probe grids must refer to identical states, not nearest-node shifts.
                if not (np.allclose(current['x'], previous['x'], atol=1e-12, rtol=0)
                        and np.allclose(current['y'], previous['y'], atol=1e-12, rtol=0)):
                    raise RuntimeError('Refinement probe states do not align')
                delta = np.abs(current['prices']-previous['prices'])
                record['bond_max_error'] = float(delta[:, 0].max())
                record['protection_max_error'] = float(delta[:cfg.cds_days, 1].max())
                # Annuity errors are converted to monetary errors at a fixed
                # 10% coupon, above the intended ordinary quoted coupon scale.
                record['annuity_cash_max_error_at_coupon_0_1'] = float(.1*delta[:cfg.cds_days, 2].max())
                record['passed'] = max(record['bond_max_error'], record['protection_max_error'],
                                       record['annuity_cash_max_error_at_coupon_0_1']) < tolerance
                if record['passed']:
                    # Production is the grid whose error against the finer
                    # reference passed the frozen tolerance. Retaining all
                    # reference maturities would exceed a 24GB GPU's memory.
                    accepted = previous_config
            records.append(record)
            (output/f'{domain}-refinement.json').write_text(json.dumps(records, indent=2, allow_nan=False))
            print(domain, record, flush=True)
            previous = current
            previous_config = cfg
            jax.clear_caches()
            gc.collect()
            if accepted is not None:
                break
        boundary_passed = False
        if accepted is not None:
            production = jax.device_get(sampled_prices(accepted))
            # Add one unit to each boundary while preserving grid spacing.
            x_extra = round((accepted.grid_size-1)/(accepted.grid_max-accepted.grid_min))
            y_extra = 0 if accepted.fast_grid_size == 1 else round((accepted.fast_grid_size-1)/(accepted.fast_grid_max-accepted.fast_grid_min))
            wide = replace(accepted, grid_min=accepted.grid_min-1, grid_max=accepted.grid_max+1,
                           grid_size=accepted.grid_size+2*x_extra,
                           fast_grid_min=accepted.fast_grid_min-1, fast_grid_max=accepted.fast_grid_max+1,
                           fast_grid_size=accepted.fast_grid_size+2*y_extra)
            reference = jax.device_get(sampled_prices(wide))
            np.savez_compressed(output/f'{domain}-boundary.npz', **reference)
            delta = np.abs(reference['prices']-production['prices'])
            boundary_error = float(max(delta[:, 0].max(), delta[:accepted.cds_days, 1].max(), .1*delta[:accepted.cds_days, 2].max()))
            boundary_passed = boundary_error < tolerance
        else:
            boundary_error = None
        from dataclasses import asdict
        summaries[domain] = dict(passed=accepted is not None and boundary_passed,
                                production_config=asdict(accepted) if accepted else None,
                                boundary_max_cash_error=boundary_error, refinement=records)
        (output/'numerical_summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False))
        jax.clear_caches()
        gc.collect()
    return summaries
