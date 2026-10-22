"""Exercise actual pathwise meta-learning and self-play JVPs with analytic fused kernels."""
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from neural_sde_hedging.config import SDEConfig
from neural_sde_hedging.dynamics import NeuralSDE, sample_noise, simulate
from neural_sde_hedging.hedger import Hedger, Contract, terminal_losses
from neural_sde_hedging.methods import pathwise_meta_learning, selfplay_rewards
from neural_sde_hedging.training import learner_block
from pallas_study.benchmark import compare, risk
from pallas_study.kernels import convert


def main():
    """Check and time a full 126-day two-inner-update meta gradient and self-play reward."""
    assert jax.default_backend() == 'gpu'
    jax.config.update('jax_default_matmul_precision', 'highest')
    config = SDEConfig()
    batch = 128
    teacher = NeuralSDE(jax.random.key(101), config)
    hedger = Hedger(jax.random.key(102))
    noise = sample_noise(jax.random.key(103), batch, config)
    feedback = eqx.filter_jit(simulate)(teacher, sample_noise(jax.random.key(104), batch, config), config)
    contract = Contract(premium=.04)
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(.001, eps_root=1e-8))
    report = {'batch': batch, 'steps': config.n_steps, 'inner_updates': 2, 'results': []}
    reference = None
    reward_reference = None
    prepared = []
    for name, fuse_gru, fuse_mlp in (('jax', False, False), ('pallas_gru', True, False),
                                    ('pallas_mlp', False, True), ('pallas_both', True, True)):
        generator = convert(teacher, mlp=fuse_mlp, analytic_jvp=True)
        learner = convert(hedger, gru=fuse_gru, analytic_jvp=True)
        params, static = eqx.partition(generator, eqx.is_inexact_array)
        learner_params, learner_static = eqx.partition(learner, eqx.is_inexact_array)
        student = (learner_params, optimizer.init(learner_params))

        def objective(candidate, current_student, innovations, target_paths):
            """Return true downstream risk change through the production two-step Adam learner."""
            def simulate_fn(model, z):
                """Generate differentiable full-horizon neural SDE paths."""
                return simulate(model, z, config)

            def block_fn(state, states, key):
                """Apply production clipped AdamW learner with rematerialized inner updates."""
                return learner_block(state, states, jnp.ones(batch), learner_static,
                                     optimizer, contract, 10., 2, key, bootstrap=False)

            def target_fn(state):
                """Evaluate entropic risk on the fixed independent feedback paths."""
                return risk(terminal_losses(eqx.combine(state[0], learner_static), target_paths, contract))

            value, _ = pathwise_meta_learning(eqx.combine(candidate, static), current_student,
                innovations, jax.random.key(105), simulate_fn, block_fn, target_fn)
            return value

        def rewards(candidate, paths):
            """Evaluate production historical-alignment rewards including the full hedger JVP."""
            historical = jax.tree.map(lambda x: x-.01, candidate)
            moments = jax.tree.map(jnp.ones_like, candidate)

            def losses_fn(p):
                """Return full terminal loss vector for a learner parameter PyTree."""
                return terminal_losses(eqx.combine(p, learner_static), paths, contract)

            return selfplay_rewards(candidate, historical, moments, jnp.array(10), .001,
                                    losses_fn, entropic_gamma=10., epsilon_root=1e-8)

        compiled = eqx.filter_jit(eqx.filter_value_and_grad(objective))
        reward_compiled = eqx.filter_jit(rewards)
        print('COMPILE', name, flush=True)
        started = time.perf_counter()
        result = compiled(params, student, noise, feedback)
        jax.block_until_ready(result)
        compilation = time.perf_counter()-started
        reward = reward_compiled(learner_params, feedback)
        if name == 'jax':
            reference, reward_reference = result, reward
        error, reward_error = compare(reference, result), compare(reward_reference, reward)
        record = {'variant': name, 'compile_and_first_gradient_s': compilation,
                  'meta_gradient_error': error, 'selfplay_reward_error': reward_error,
                  'meta_value': float(result[0]), 'meta_ms': [], 'selfplay_ms': []}
        record['correct'] = (error['relative_l2'] < .002 or error['max_abs'] < 2e-7) and (reward_error['relative_l2'] < .0001 or reward_error['max_abs'] < 1e-8)
        assert record['correct'], record
        report['results'].append(record)
        for _ in range(2):
            jax.block_until_ready(compiled(params, student, noise, feedback))
        prepared.append((compiled, reward_compiled, params, student, learner_params, record))
        Path('outputs/meta.json').write_text(json.dumps(report, indent=2))
    for repetition in range(9):
        for compiled, reward_compiled, params, student, learner_params, record in (prepared if repetition % 2 == 0 else prepared[::-1]):
            start = time.perf_counter()
            for _ in range(3):
                result = compiled(params, student, noise, feedback)
            jax.block_until_ready(result)
            record['meta_ms'].append(1000*(time.perf_counter()-start)/3)
            start = time.perf_counter()
            for _ in range(3):
                result = reward_compiled(learner_params, feedback)
            jax.block_until_ready(result)
            record['selfplay_ms'].append(1000*(time.perf_counter()-start)/3)
    for record in report['results']:
        record['median_meta_ms'] = float(np.median(record['meta_ms']))
        record['median_selfplay_ms'] = float(np.median(record['selfplay_ms']))
        print('RESULT', record['variant'], record['median_meta_ms'], record['median_selfplay_ms'], flush=True)
    report['complete'] = True
    Path('outputs/meta.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
