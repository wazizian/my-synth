"""GPU financial reductions; normalized shortfall is positive for losses."""
from pathlib import Path
import os
import tempfile
import numpy as np
import jax
import jax.numpy as jnp


def summarize_outcomes(shortfall, diagnostics, *, scale, domain, split_role,
                       data_hash, checkpoint_id):
    if domain not in {'base', 'easy', 'hard'}:
        raise ValueError('unknown evaluation domain')
    if split_role not in {'monitor', 'selection', 'final_test'}:
        raise ValueError('unknown evaluation split role')
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('scale must be fixed, positive and finite')
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Outcome reductions must run on a GPU worker')
    values = jnp.asarray(shortfall)
    if values.ndim != 1:
        raise ValueError('shortfall must have one scalar per episode')
    finite = jnp.isfinite(values)
    finite_count = int(jnp.sum(finite))
    result = dict(domain=domain, split_role=split_role, data_hash=data_hash,
                  checkpoint_id=checkpoint_id, sample_count=int(values.size),
                  finite_count=finite_count, nonfinite_count=int(values.size)-finite_count,
                  reference_scale=float(scale), pnl_convention='negative normalized shortfall',
                  cost_units='discounted simulator currency', turnover_units='instrument units')
    valid = values.size > 0 and finite_count == values.size
    names = ('raw_mse', 'scaled_mse', 'mean_shortfall', 'mean_pnl',
             'shortfall_std', 'mean_absolute_error', 'var95', 'es95')
    if valid:
        var = jnp.quantile(values, .95)
        tail = values >= var
        mse, mean = jnp.mean(values ** 2), jnp.mean(values)
        summary = jnp.stack((mse, mse / scale, mean, -mean,
                            jnp.std(values), jnp.mean(jnp.abs(values)), var,
                            jnp.sum(jnp.where(tail, values, 0)) / jnp.sum(tail)))
        result.update(zip(names, np.asarray(summary).tolist()))
    else:
        result.update({name: None for name in names})
    diagnostic_nonfinite = 0
    for name in ('turnover', 'costs', 'liquidation_costs', 'mean_abs_holdings', 'saturation_fraction'):
        value = diagnostics.get(name)
        if value is None:
            result[name] = None
            continue
        array = jnp.asarray(value)
        if array.shape[0] != values.size:
            raise ValueError('diagnostic episode count mismatch: ' + name)
        bad = int(jnp.sum(~jnp.isfinite(array)))
        diagnostic_nonfinite += bad
        result[name] = np.asarray(jnp.mean(array, axis=0)).tolist() if not bad and values.size else None
    costs = result['costs']
    result['total_costs'] = float(sum(costs)) if costs is not None else None
    result['diagnostic_nonfinite_count'] = diagnostic_nonfinite
    result['status'] = 'ok' if valid and not diagnostic_nonfinite else 'nonfinite_or_empty' 
    return result


def aggregate_domains(metrics, scales=None):
    """Keep each domain distinct; expose the predetermined equal scaled mean."""
    easy, hard = metrics['easy']['raw_mse'], metrics['hard']['raw_mse']
    if easy is None or hard is None:
        return dict(easy_hard_raw_mean=None, balanced_objective=None)
    scales = scales or {k: v['reference_scale'] for k, v in metrics.items()}
    if any(not np.isfinite(scales[k]) or scales[k] <= 0 for k in ('easy', 'hard')):
        raise ValueError('invalid reference scale')
    return dict(easy_hard_raw_mean=.5 * (easy + hard),
                balanced_objective=.5 * (easy / scales['easy'] + hard / scales['hard']))


def save_episode_metrics(path, shortfall, diagnostics):
    """Retain all episode outcomes, including failures, atomically as plain NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(value) for name, value in diagnostics.items()}
    arrays.update(shortfall=np.asarray(shortfall), pnl=-np.asarray(shortfall))
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.tmp-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(path)
