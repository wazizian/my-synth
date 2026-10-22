"""Synchronized full-training benchmarks and gradient checks on identical inputs."""
import argparse
import json
import platform
from pathlib import Path
import time

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from neural_sde_hedging.config import SDEConfig
from neural_sde_hedging.dynamics import NeuralSDE, simulate, path_log_prob, sample_noise, transition_log_prob
from neural_sde_hedging.hedger import Hedger, Contract, terminal_losses
from neural_sde_hedging import option_benchmark
from pallas_study.kernels import convert


def risk(losses):
    """Return empirical entropic risk scalar from terminal losses [batch]."""
    batch = losses.shape[0]
    result = (jax.scipy.special.logsumexp(10*losses)-jnp.log(batch))/10
    chex.assert_shape(result, ())
    return result


def compare(reference, candidate):
    """Compare corresponding numeric PyTree leaves; return finite absolute/relative errors."""
    left = jax.tree.leaves(reference)
    right = jax.tree.leaves(candidate)
    if len(left) != len(right):
        raise AssertionError('Parameter leaf count differs')
    maximum, square_error, square_reference = 0., 0., 0.
    for a, b in zip(left, right):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        maximum = max(maximum, float(np.max(np.abs(a-b))))
        square_error += float(np.sum((a.astype(float)-b.astype(float))**2))
        square_reference += float(np.sum(a.astype(float)**2))
    return {'max_abs': maximum, 'relative_l2': (square_error/max(square_reference, 1e-30))**.5}


def main():
    """Run paired fixed-seed full-model checks and warm synchronized training timings."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--steps', type=int, default=126)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--updates', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', default=['hedger', 'historical', 'sde_score', 'sde_score_flat', 'sde_pathwise', 'joint'])
    args = parser.parse_args()
    assert jax.default_backend() == 'gpu'
    config = SDEConfig(n_steps=args.steps)
    teacher = NeuralSDE(jax.random.key(101), config, width=args.hidden)
    hedger = Hedger(jax.random.key(102), hidden_size=args.hidden)
    historical = option_benchmark.Hedger(jax.random.key(102), hidden_size=args.hidden)
    # A nonzero head exercises GRU gradients from the first check.
    historical = eqx.tree_at(lambda m: m.output, historical, hedger.output)
    noise = sample_noise(jax.random.key(103), args.batch, config)
    paths = eqx.filter_jit(simulate)(teacher, noise, config)
    jax.block_until_ready(paths)
    contract = Contract(premium=.04)
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(.001))
    report = {'config': vars(args) | {'output': str(args.output)},
              'jax': jax.__version__, 'python': platform.python_version(),
              'device': str(jax.devices()[0]), 'results': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def write():
        """Atomically save completed measurements so interrupted runs remain useful."""
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(args.output)

    for task in args.tasks:
        if task in ('hedger', 'historical'):
            baseline = hedger if task == 'hedger' else historical
            variants = [('jax', baseline), ('pallas_gru', convert(baseline, gru=True))]
        elif task in ('sde_score', 'sde_score_flat', 'sde_pathwise'):
            baseline = teacher
            variants = [('jax', teacher), ('pallas_mlp', convert(teacher, mlp=True))]
        else:
            baseline = (teacher, hedger)
            variants = [('jax', baseline), ('pallas_gru', (teacher, convert(hedger, gru=True))),
                        ('pallas_mlp', (convert(teacher, mlp=True), hedger)),
                        ('pallas_both', (convert(teacher, mlp=True), convert(hedger, gru=True)))]

        def objective(model, inputs):
            """Return selected production loss for model PyTree and dynamic noise/path inputs."""
            fixed_paths, innovations = inputs
            if task == 'hedger':
                return risk(terminal_losses(model, fixed_paths, contract))
            if task == 'historical':
                return risk(option_benchmark.rollout(model, fixed_paths.log_spot, .0003, .04)['loss'])
            if task == 'sde_score':
                return -jnp.mean(path_log_prob(model, fixed_paths, config))/args.steps
            if task == 'sde_score_flat':
                states = jax.lax.stop_gradient(fixed_paths.states)
                chex.assert_shape(states, (args.batch, args.steps+1, config.state_dim))
                previous = jnp.swapaxes(states[:, :-1], 0, 1)
                following = jnp.swapaxes(states[:, 1:], 0, 1)
                chex.assert_shape((previous, following), (args.steps, args.batch, config.state_dim))
                def score(before, after, time):
                    """Score one time slice across the batch; vmap exposes all times together."""
                    return transition_log_prob(after, before, time, model, config)
                values = jax.vmap(score)(previous, following, fixed_paths.times[:-1])
                chex.assert_shape(values, (args.steps, args.batch))
                return -jnp.mean(values)
            if task == 'sde_pathwise':
                generated = simulate(model, innovations, config)
                return risk(terminal_losses(hedger, generated, contract))
            generator, policy = model
            generated = simulate(generator, innovations, config)
            return risk(terminal_losses(policy, generated, contract))

        def update(model, state, inputs):
            """Compute full reverse gradient plus clipped AdamW update; return new state/loss."""
            value, gradients = eqx.filter_value_and_grad(objective)(model, inputs)
            changes, state = optimizer.update(gradients, state, eqx.filter(model, eqx.is_array))
            model = eqx.apply_updates(model, changes)
            return model, state, value

        compiled = eqx.filter_jit(update)
        derivative = eqx.filter_jit(eqx.filter_value_and_grad(objective))
        forward = eqx.filter_jit(objective)
        inputs = (paths, noise)
        reference_value, reference_grad = derivative(baseline, inputs)
        jax.block_until_ready((reference_value, reference_grad))
        states = []
        for name, model in variants:
            print('COMPILE', task, name, flush=True)
            state = optimizer.init(eqx.filter(model, eqx.is_array))
            started = time.perf_counter()
            result = compiled(model, state, inputs)
            jax.block_until_ready(result)
            compile_seconds = time.perf_counter()-started
            value, gradient = derivative(model, inputs)
            jax.block_until_ready((value, gradient))
            error = compare(reference_grad, gradient)
            value_error = abs(float(value-reference_value))
            passed = value_error < 2e-5 and (error['relative_l2'] < .002 or error['max_abs'] < 2e-6)
            record = {'task': task, 'variant': name, 'compile_and_first_update_s': compile_seconds,
                      'loss': float(value), 'loss_error': value_error, 'gradient_error': error,
                      'correct': passed, 'update_ms': [], 'forward_ms': []}
            report['results'].append(record)
            write()
            if not passed:
                print('REJECT', task, name, error, flush=True)
                continue
            jax.block_until_ready(forward(model, inputs))
            for _ in range(3):
                jax.block_until_ready(compiled(model, state, inputs))
            states.append((name, model, state, record))

        # Alternate ordering to reduce thermal/clock/order bias.
        for repetition in range(args.repeats):
            order = states if repetition % 2 == 0 else states[::-1]
            for name, initial_model, initial_state, record in order:
                model, state = initial_model, initial_state
                started = time.perf_counter()
                for _ in range(args.updates):
                    model, state, value = compiled(model, state, inputs)
                jax.block_until_ready((model, state, value))
                record['update_ms'].append(1000*(time.perf_counter()-started)/args.updates)
                started = time.perf_counter()
                for _ in range(args.updates):
                    value = forward(initial_model, inputs)
                jax.block_until_ready(value)
                record['forward_ms'].append(1000*(time.perf_counter()-started)/args.updates)
            write()

        # Check cumulative optimizer trajectory, not just the first gradient.
        trained_reference = None
        for name, model, state, record in states:
            for _ in range(20):
                model, state, value = compiled(model, state, inputs)
            jax.block_until_ready((model, state, value))
            if name == 'jax':
                trained_reference = eqx.filter(model, eqx.is_array)
            record['parameters_after_20_updates'] = compare(trained_reference, eqx.filter(model, eqx.is_array))
            record['loss_after_20_updates'] = float(value)
            record['median_update_ms'] = float(np.median(record['update_ms']))
            record['median_forward_ms'] = float(np.median(record['forward_ms']))
            print('RESULT', task, name, record['median_update_ms'], flush=True)
        write()
    report['complete'] = True
    write()


if __name__ == '__main__':
    main()
