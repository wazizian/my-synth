"""Check fused network values, parameter/input derivatives, JVP and second derivatives."""
import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from pallas_study.benchmark import compare
from pallas_study.kernels import fused_gru, gru_reference, fused_mlp, mlp_reference, fast_gru, fast_mlp


def main():
    """Validate original and fused functions on GPU, saving scalar error evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--analytic-jvp', action='store_true')
    args = parser.parse_args()
    assert jax.default_backend() == 'gpu'
    results = []
    for width in (32, 64):
        gru = eqx.nn.GRUCell(4, width, key=jax.random.key(201))
        parameters = (gru.weight_ih, gru.weight_hh, gru.bias, gru.bias_n)
        mlp = eqx.nn.MLP(4, 3, width, 2, activation=jnp.tanh, key=jax.random.key(202))
        layers = tuple((layer.weight, layer.bias) for layer in mlp.layers)
        features = jax.random.normal(jax.random.key(203), (4,))
        hidden = jax.random.normal(jax.random.key(204), (width,))
        for name, original, fused, arguments in (
            ('gru', gru_reference, fast_gru if args.analytic_jvp else fused_gru, (parameters, features, hidden)),
            ('mlp', mlp_reference, fast_mlp if args.analytic_jvp else fused_mlp, (layers, features))):
            direction = jax.tree.map(lambda x: jnp.full_like(x, .1), arguments)

            def original_loss(args):
                """Return squared output norm scalar for original model/input PyTree."""
                return jnp.sum(original(*args)**2)

            def fused_loss(args):
                """Return squared output norm scalar for fused model/input PyTree."""
                return jnp.sum(fused(*args)**2)

            value = jax.jit(original)(*arguments)
            candidate = jax.jit(fused)(*arguments)
            gradient = jax.jit(jax.grad(original_loss))(arguments)
            candidate_gradient = jax.jit(jax.grad(fused_loss))(arguments)
            jvp_original = jax.jit(lambda a, d: jax.jvp(original_loss, (a,), (d,)))(arguments, direction)
            jvp_fused = jax.jit(lambda a, d: jax.jvp(fused_loss, (a,), (d,)))(arguments, direction)
            hessian_original = jax.jit(lambda a, d: jax.jvp(jax.grad(original_loss), (a,), (d,))[1])(arguments, direction)
            hessian_fused = jax.jit(lambda a, d: jax.jvp(jax.grad(fused_loss), (a,), (d,))[1])(arguments, direction)
            epsilon = .001
            plus = jax.tree.map(lambda a, d: a+epsilon*d, arguments, direction)
            minus = jax.tree.map(lambda a, d: a-epsilon*d, arguments, direction)
            finite_difference = (jax.jit(fused_loss)(plus)-jax.jit(fused_loss)(minus))/(2*epsilon)
            record = {'network': name, 'width': width,
                      'output': compare(value, candidate),
                      'gradient': compare(gradient, candidate_gradient),
                      'jvp': compare(jvp_original, jvp_fused),
                      'hessian_vector': compare(hessian_original, hessian_fused),
                      'finite_difference': float(finite_difference),
                      'directional_derivative': float(jvp_fused[1])}
            for key in ('output', 'gradient', 'jvp', 'hessian_vector'):
                assert record[key]['relative_l2'] < 1e-4 or record[key]['max_abs'] < 1e-5, record
            assert abs(float(finite_difference-jvp_fused[1])) < .003*max(1., abs(float(jvp_fused[1]))), record
            results.append(record)
            print(json.dumps(record), flush=True)
    Path('outputs/analytic-validation.json' if args.analytic_jvp else 'outputs/validation.json').write_text(json.dumps({'passed': True, 'results': results}, indent=2))


if __name__ == '__main__':
    main()
