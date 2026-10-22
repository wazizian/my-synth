"""Batched small-MLP fusion using tensor cores and saved analytic derivatives."""
import chex
import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap
from jax.experimental import pallas as pl

from pallas_study.kernels import mlp_forward


@custom_vmap
def tiled_activations(parameters, x):
    """Evaluate one vector MLP; the vmap rule fuses tiles of batch rows instead."""
    return mlp_forward(parameters, x, saved=True)


@tiled_activations.def_vmap
def tiled_batch(axis_size, in_batched, parameters, x):
    """Evaluate shared-weight float32 MLPs on tiles of 16 input rows; return saved activations."""
    parameter_batching, x_batched = in_batched
    if any(jax.tree.leaves(parameter_batching)) or not x_batched or x.dtype != jnp.float32:
        axes = jax.tree.map(lambda flag: 0 if flag else None, in_batched)
        result = jax.vmap(lambda p, a: mlp_forward(p, a, saved=True), in_axes=tuple(axes), axis_size=axis_size)(parameters, x)
        return result, True
    batch, n_features = x.shape
    width = max(32, pl.next_power_of_2(max(max(weight.shape) for weight, _ in parameters)))
    tile = 16
    padded_batch = pl.cdiv(batch, tile)*tile
    padded = tuple((jnp.pad(weight, ((0, width-weight.shape[0]), (0, width-weight.shape[1]))),
                    jnp.pad(bias, (0, width-bias.shape[0]))) for weight, bias in parameters)
    for weight, bias in padded:
        chex.assert_shape(weight, (width, width))
        chex.assert_shape(bias, (width,))
    features = jnp.pad(x, ((0, padded_batch-batch), (0, width-n_features)))
    chex.assert_shape(features, (padded_batch, width))
    precision = jax.lax.Precision.HIGHEST if jax.config.jax_default_matmul_precision == 'highest' else jax.lax.Precision.DEFAULT

    def kernel(*refs):
        """Fuse all shared-weight affine/tanh layers on one 16-row tile without masked loads."""
        input_ref, output_ref = refs[-2:]
        rows = pl.program_id(0)*tile+jnp.arange(tile)
        columns = jnp.arange(width)
        value = pl.load(input_ref, (rows[:, None], columns[None, :]))
        chex.assert_shape(value, (tile, width))
        pl.store(output_ref, (rows[:, None], 0, columns[None, :]), value)
        for layer in range(len(parameters)):
            weight = pl.load(refs[2*layer], (columns[:, None], columns[None, :]))
            chex.assert_shape(weight, (width, width))
            bias = pl.load(refs[2*layer+1], (columns,))
            chex.assert_shape(bias, (width,))
            value = pl.dot(value, weight, trans_b=True, precision=precision)+bias[None, :]
            chex.assert_shape(value, (tile, width))
            if layer < len(parameters)-1:
                value = jnp.tanh(value)
                chex.assert_shape(value, (tile, width))
            pl.store(output_ref, (rows[:, None], layer+1, columns[None, :]), value)

    flat = tuple(value for pair in padded for value in pair)
    result = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((padded_batch, len(parameters)+1, width), x.dtype),
        grid=(padded_batch//tile,), name='tiled_tensorcore_mlp')(*flat, features)
    chex.assert_shape(result, (padded_batch, len(parameters)+1, width))
    result = result[:batch]
    chex.assert_shape(result, (batch, len(parameters)+1, width))
    return result, True


@jax.custom_jvp
def saved_activations(parameters, x):
    """Return differentiable saved activations, dispatching to the tiled vmap rule."""
    return tiled_activations(parameters, x)


@saved_activations.defjvp
def activation_jvp(primals, tangents):
    """Return all activation tangents for higher-order differentiation of saved values."""
    parameters, x = primals
    dp, tangent = tangents
    values = saved_activations(parameters, x)
    width = values.shape[1]
    rows = [jnp.pad(tangent, (0, width-tangent.shape[0]))]
    for index, ((weight, _), (dw, db)) in enumerate(zip(parameters, dp)):
        n_out, n_in = weight.shape
        tangent = weight@tangent+dw@values[index, :n_in]+db
        chex.assert_shape(tangent, (n_out,))
        if index < len(parameters)-1:
            tangent = tangent*(1-values[index+1, :n_out]**2)
            chex.assert_shape(tangent, (n_out,))
        rows.append(jnp.pad(tangent, (0, width-n_out)))
    result = jnp.stack(rows)
    chex.assert_equal_shape((result, values))
    return values, result


@jax.custom_jvp
def tensorcore_mlp(parameters, x):
    """Return fused MLP output with batched tensor-core forward execution and analytic AD."""
    values = saved_activations(parameters, x)
    result = values[-1, :parameters[-1][0].shape[0]]
    chex.assert_shape(result, (parameters[-1][0].shape[0],))
    return result


@tensorcore_mlp.defjvp
def tensorcore_jvp(primals, tangents):
    """Return final-output tangents while reusing the fused, shared-weight activation tape."""
    parameters, x = primals
    dp, tangent = tangents
    values = saved_activations(parameters, x)
    for index, ((weight, _), (dw, db)) in enumerate(zip(parameters, dp)):
        n_out, n_in = weight.shape
        tangent = weight@tangent+dw@values[index, :n_in]+db
        chex.assert_shape(tangent, (n_out,))
        if index < len(parameters)-1:
            tangent = tangent*(1-values[index+1, :n_out]**2)
            chex.assert_shape(tangent, (n_out,))
    return values[-1, :parameters[-1][0].shape[0]], tangent
