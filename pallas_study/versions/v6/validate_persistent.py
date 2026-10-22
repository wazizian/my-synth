"""GPU first-order parameter and path derivative checks for whole-trajectory kernels."""
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from neural_sde_hedging import option_benchmark
from neural_sde_hedging.dynamics import SDEPaths
from neural_sde_hedging.hedger import Hedger, Contract, terminal_losses
from pallas_study.benchmark import compare, risk
from pallas_study.persistent import PersistentHedger


def main():
    """Compare complete loss vectors and all model/price gradients on varied path lengths."""
    jax.config.update('jax_default_matmul_precision', 'highest')
    records = []
    for n_steps in (1, 8, 126):
        model = Hedger(jax.random.key(102))
        batch = 32
        noise = jax.random.normal(jax.random.key(301), (batch, n_steps))*.02
        log_prices = jnp.concatenate((jnp.zeros((batch, 1)), jnp.cumsum(noise, axis=1)), axis=1)
        for historical in (False, True):
            def original(model, prices):
                """Evaluate original terminal losses from model and arbitrary observable paths."""
                if historical:
                    return option_benchmark.rollout(model, prices, .0003, .04)['loss']
                states = jnp.stack((prices, jnp.zeros_like(prices), jnp.zeros_like(prices)), axis=-1)
                paths = SDEPaths(states, jnp.zeros((batch, n_steps, 3)), jnp.zeros(batch), jnp.arange(n_steps+1)/252)
                return terminal_losses(model, paths, Contract(premium=.04))

            def fused(model, prices):
                """Evaluate corresponding persistent losses on exactly the same inputs."""
                return PersistentHedger(model).terminal_losses(prices, historical=historical)

            def reference_risk(inputs):
                """Return entropic risk for model/observable-path PyTree."""
                return risk(original(*inputs))

            def fused_risk(inputs):
                """Return persistent entropic risk for the same input PyTree."""
                return risk(fused(*inputs))

            inputs = (model, log_prices)
            reference = eqx.filter_jit(original)(*inputs)
            candidate = eqx.filter_jit(fused)(*inputs)
            gradient = eqx.filter_jit(eqx.filter_grad(reference_risk))(inputs)
            fused_gradient = eqx.filter_jit(eqx.filter_grad(fused_risk))(inputs)
            errors = {'output': compare(reference, candidate),
                      'parameters': compare(gradient[0], fused_gradient[0]),
                      'prices': compare(gradient[1], fused_gradient[1])}
            record = {'steps': n_steps, 'historical': historical, **errors}
            for error in errors.values():
                assert error['relative_l2'] < .0001 or error['max_abs'] < 2e-6, record
            records.append(record)
            print(json.dumps(record), flush=True)
    Path('outputs/persistent-validation.json').write_text(json.dumps({'passed': True, 'results': records}, indent=2))


if __name__ == '__main__':
    main()
