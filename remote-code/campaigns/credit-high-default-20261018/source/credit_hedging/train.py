"""Reproducible small training entrypoint; no cloud jobs or adaptive teachers."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .environment import rollout
from .market import MarketConfig, build_tables, simulate_paths
from .metrics import entropic_risk, evaluate_metrics
from .policy import Hedger


def validate_evaluation(result, metrics, accounting_tolerance=1e-9):
    """Check a completed rollout/metric PyTree on host; raise on nonfinite data or accounting error.

    Called outside JIT before any output is published. Validation inspects daily
    arrays as well as scalar summaries; a NaN must not evade a threshold comparison.
    """
    for value in jax.tree.leaves((result, metrics)):
        if eqx.is_array(value) and not np.isfinite(np.asarray(value)).all():
            raise RuntimeError('Nonfinite evaluation array or metric')
    if float(metrics['max_accounting_error']) > accounting_tolerance:
        raise RuntimeError('Evaluation accounting identity failed')


@eqx.filter_jit
def train_chunk(model, optimizer_state, key, tables, config, optimizer, n_updates, batch_size, gamma):
    """Train with fresh paths via scan; inputs model/optimizer/key PyTrees and settings.

    Returns updated model, optimizer state, PRNG key and per-update scalar logs.
    The single JIT boundary includes simulation, recurrent rollout and gradients.
    """
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def step(carry, unused):
        """Take one optimizer step from params/state/key; output next carry and loss/gradient logs."""
        params, optimizer_state, key = carry
        key, path_key = jax.random.split(key)
        paths = simulate_paths(path_key, tables, config, batch_size)

        def objective(parameters):
            """Map differentiable parameter PyTree to scalar empirical entropic hedging risk."""
            candidate = eqx.combine(parameters, static)
            result = rollout(candidate, paths, tables, config)
            return entropic_risk(result['loss'], gamma)

        loss, gradients = jax.value_and_grad(objective)(params)
        chex.assert_shape(loss, ())
        gradient_norm = optax.global_norm(gradients)
        chex.assert_shape(gradient_norm, ())
        defaulted = jnp.any(paths['default'], axis=1)
        chex.assert_shape(defaulted, (batch_size,))
        default_count = jnp.sum(defaulted)
        chex.assert_shape(default_count, ())
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        return (params, optimizer_state, key), dict(loss=loss, gradient_norm=gradient_norm, default_count=default_count)

    (params, optimizer_state, key), logs = jax.lax.scan(step, (params, optimizer_state, key), None, length=n_updates)
    chex.assert_shape((logs['loss'], logs['gradient_norm']), (n_updates,))
    return eqx.combine(params, static), optimizer_state, key, logs


def main():
    """Read CLI settings, train/evaluate locally and save config, arrays and resumable PyTrees."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updates', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--eval-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20261017)
    parser.add_argument('--hidden-size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=10.0)
    parser.add_argument('--output', type=Path, default=Path('credit_hedging/runs/smoke'))
    parser.add_argument('--short-smoke', action='store_true', help='Use 30 bond days/10 CDS days; not the 5-year experiment')
    args = parser.parse_args()
    if min(args.updates, args.batch_size, args.eval_size, args.hidden_size) < 1 or not math.isfinite(args.gamma) or args.gamma <= 0:
        parser.error('Counts must be positive and gamma finite and positive')
    if args.output.exists():
        parser.error('Output already exists; choose a new directory to preserve earlier results')
    jax.config.update('jax_enable_x64', True)
    config = MarketConfig()
    if args.short_smoke:
        config = MarketConfig(bond_days=30, cds_days=10, grid_size=101)
    tables = eqx.filter_jit(build_tables)(config)
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(tables)):
        raise RuntimeError('Nonfinite pricing tables')
    key_model, key_train, key_eval = jax.random.split(jax.random.key(args.seed), 3)
    model = Hedger(key_model, config.cds_days, args.hidden_size, config.max_protection)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    paths = eqx.filter_jit(simulate_paths)(key_eval, tables, config, args.eval_size)
    initial = eqx.filter_jit(rollout)(model, paths, tables, config)
    unhedged = eqx.filter_jit(rollout)(None, paths, tables, config)
    print(f'{jax.devices()}: bond={config.bond_days} days, CDS={config.cds_days} vintages', flush=True)
    model, optimizer_state, key_train, logs = train_chunk(model, optimizer_state, key_train, tables, config,
        optimizer, args.updates, args.batch_size, args.gamma)
    jax.block_until_ready(logs)
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((model, optimizer_state, logs)) if eqx.is_array(x)):
        raise RuntimeError('Nonfinite training state; refusing to publish a successful result')
    final = eqx.filter_jit(rollout)(model, paths, tables, config)
    summaries = {}
    for name, result in [('unhedged', unhedged), ('initial', initial), ('trained', final)]:
        metrics = evaluate_metrics(result, args.gamma)
        validate_evaluation(result, metrics)
        summaries[name] = {k:float(v) for k,v in metrics.items() if v.ndim == 0}
    args.output.mkdir(parents=True)
    metadata = dict(market=asdict(config), seed=args.seed, hidden_size=args.hidden_size,
        batch_size=args.batch_size, eval_size=args.eval_size, updates=args.updates, gamma=args.gamma,
        alpha=0.95, learning_rate=0.001, backend=jax.default_backend(),
        jax_version=jax.__version__, equinox_version=eqx.__version__, x64=True,
        policy_action_parameterization=model.action_parameterization,
        evaluation_role='fixed diagnostic set, not a held-out scientific final test',
        short_smoke=args.short_smoke)
    (args.output/'config.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    (args.output/'metrics.json').write_text(json.dumps(summaries, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(args.output/'training.npz', **{k:np.asarray(v) for k,v in logs.items()})
    arrays = {f'path_{k}':np.asarray(v) for k,v in paths.items()}
    for name, result in [('unhedged', unhedged), ('initial', initial), ('trained', final)]:
        arrays[name+'_loss'] = np.asarray(result['loss'])
        arrays.update({name+'_'+k:np.asarray(v) for k,v in result['records'].items()})
    np.savez_compressed(args.output/'evaluation.npz', **arrays)
    # Typed keys are serialized as raw uint32 data; wrap_key_data restores them.
    eqx.tree_serialise_leaves(args.output/'checkpoint.eqx', (model, optimizer_state, jax.random.key_data(key_train)))
    print(json.dumps(summaries, indent=2), flush=True)
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
