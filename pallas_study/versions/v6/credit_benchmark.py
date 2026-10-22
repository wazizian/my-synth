"""Check actual 1260-day credit hedger and continuous neural-credit score updates."""
import argparse
import json
from pathlib import Path
import time
import traceback

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from credit_hedging.curve_market import CurveConfig, build_curve_tables, simulate_curve_paths
from credit_hedging.environment import rollout
from credit_hedging.learned_policy import TargetHedger
from neural_sde_hedging import credit_continuous as credit
from pallas_study.benchmark import compare, risk
from pallas_study.kernels import convert


def main():
    """Run correctness-gated full Adam updates; save failures as well as successful timings."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['hedger', 'sde'], required=True)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--analytic-only', action='store_true')
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', args.task == 'hedger')
    jax.config.update('jax_default_matmul_precision', 'highest')
    assert jax.default_backend() == 'gpu'
    if args.task == 'hedger':
        config = CurveConfig(**json.loads(Path('distinct_regimes.json').read_text())['base'])
        tables = eqx.filter_jit(build_curve_tables)(config)
        paths = eqx.filter_jit(simulate_curve_paths)(jax.random.key(403), tables, config, args.batch)
        model = TargetHedger(jax.random.key(102))
        # Exercise recurrent gradients, rather than the zero-head initialization special case.
        head = eqx.nn.Linear(32, 127, key=jax.random.key(404), dtype=jnp.float64)
        model = eqx.tree_at(lambda m: m.head, model, head)
        data = (paths, tables)
    else:
        config = credit.CreditSDEConfig()
        model = credit.NeuralCredit(jax.random.key(101), config)
        noise = credit.sample_noise(jax.random.key(103), args.batch, config)
        data = eqx.filter_jit(credit.simulate)(model, noise, config)
    jax.block_until_ready(data)
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(.001))

    def objective(model, inputs):
        """Return original production credit risk or Euler negative log-likelihood scalar."""
        if args.task == 'hedger':
            paths, tables = inputs
            return risk(rollout(model, paths, tables, config)['loss'])
        return -jnp.mean(credit.path_log_prob(model, inputs, config))/config.n_steps

    def update(model, state, inputs):
        """Compute the full original objective, gradient and clipped AdamW update."""
        value, gradient = eqx.filter_value_and_grad(objective)(model, inputs)
        change, state = optimizer.update(gradient, state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, change), state, value

    compiled = eqx.filter_jit(update)
    derivative = eqx.filter_jit(eqx.filter_value_and_grad(objective))
    reference_value, reference_gradient = derivative(model, data)
    report = {'task': args.task, 'batch': args.batch, 'steps': 1260,
              'dtype': str(jax.tree.leaves(eqx.filter(model, eqx.is_array))[0].dtype), 'results': []}
    states = []

    def write():
        """Atomically save completed checks and timings after every variant/repetition."""
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(args.output)

    for name in (('jax', 'pallas_analytic') if args.analytic_only else ('jax', 'pallas_jvp', 'pallas_saved')):
        candidate = model if name == 'jax' else convert(model, gru=args.task == 'hedger', mlp=args.task == 'sde', saved_backward=name == 'pallas_saved', analytic_jvp=name == 'pallas_analytic')
        record = {'variant': name, 'update_ms': []}
        report['results'].append(record)
        print('COMPILE', args.task, name, flush=True)
        try:
            start = time.perf_counter()
            value, gradient = derivative(candidate, data)
            jax.block_until_ready((value, gradient))
            record['gradient_compile_s'] = time.perf_counter()-start
            record['loss_error'] = abs(float(value-reference_value))
            record['gradient_error'] = compare(reference_gradient, gradient)
            record['correct'] = record['loss_error'] < 2e-5 and (record['gradient_error']['relative_l2'] < .001 or record['gradient_error']['max_abs'] < 1e-7)
            if record['correct']:
                state = optimizer.init(eqx.filter(candidate, eqx.is_array))
                for _ in range(3):
                    jax.block_until_ready(compiled(candidate, state, data))
                states.append((candidate, state, record))
        except Exception as error:
            record['correct'] = False
            record['error'] = type(error).__name__+': '+str(error)
            traceback.print_exc()
        write()
    for repetition in range(7):
        order = states if repetition % 2 == 0 else states[::-1]
        for model, state, record in order:
            start = time.perf_counter()
            for _ in range(3):
                model, state, value = compiled(model, state, data)
            jax.block_until_ready((model, state, value))
            record['update_ms'].append(1000*(time.perf_counter()-start)/3)
        write()
    reference_model = None
    for model, state, record in states:
        for _ in range(10):
            model, state, value = compiled(model, state, data)
        jax.block_until_ready((model, state, value))
        if record['variant'] == 'jax':
            reference_model = eqx.filter(model, eqx.is_array)
        record['parameter_error_after_10_updates'] = compare(reference_model, eqx.filter(model, eqx.is_array))
        record['loss_after_10_updates'] = float(value)
        record['median_update_ms'] = float(np.median(record['update_ms']))
        print('RESULT', args.task, record['variant'], record['median_update_ms'], flush=True)
    report['complete'] = True
    write()


if __name__ == '__main__':
    main()
