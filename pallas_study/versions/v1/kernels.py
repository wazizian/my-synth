"""Experimental small-network fusion; JAX derivatives preserve higher-order AD.

One Pallas program evaluates each vector network. Vmap batches the programs.
The derivative remains ordinary JAX: this isolates forward fusion without
silently dropping parameter/input gradients or forward-mode support.
"""
import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def mlp_reference(parameters, x):
    """Map tuple of (weight,bias) layers and vector input to tanh-MLP output."""
    for index, (weight, bias) in enumerate(parameters):
        n_out, n_in = weight.shape
        chex.assert_shape(x, (n_in,))
        x = weight @ x + bias
        chex.assert_shape(x, (n_out,))
        if index < len(parameters)-1:
            x = jnp.tanh(x)
            chex.assert_shape(x, (n_out,))
    return x


@jax.custom_jvp
def fused_mlp(parameters, x):
    """Evaluate vector tanh MLP in one Pallas kernel; inputs/outputs match reference."""
    shapes = tuple(weight.shape for weight, _ in parameters)
    width = pl.next_power_of_2(max(max(shape) for shape in shapes))
    n_output = shapes[-1][0]

    def kernel(*refs):
        """Load network/feature refs, fuse all dense layers, and store output vector."""
        input_ref, output_ref = refs[-2:]
        indices = jnp.arange(width)
        chex.assert_shape(indices, (width,))
        value = pl.load(input_ref, (indices,), mask=indices < x.shape[0], other=0.)
        chex.assert_shape(value, (width,))
        for layer, (n_out, n_in) in enumerate(shapes):
            weight = pl.load(refs[2*layer], (indices[:, None], indices[None, :]),
                             mask=(indices[:, None] < n_out) & (indices[None, :] < n_in), other=0.)
            chex.assert_shape(weight, (width, width))
            bias = pl.load(refs[2*layer+1], (indices,), mask=indices < n_out, other=0.)
            chex.assert_shape(bias, (width,))
            value = jnp.sum(weight * value[None, :], axis=1) + bias
            chex.assert_shape(value, (width,))
            if layer < len(shapes)-1:
                value = jnp.tanh(value)
                chex.assert_shape(value, (width,))
        pl.store(output_ref, (indices,), value, mask=indices < n_output)

    flat = tuple(item for pair in parameters for item in pair)
    result = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((n_output,), x.dtype),
                            grid=(), name='fused_small_mlp')(*flat, x)
    chex.assert_shape(result, (n_output,))
    return result


@fused_mlp.defjvp
def mlp_jvp(primals, tangents):
    """Return fused primal and exact JAX directional derivative for all inputs."""
    result = fused_mlp(*primals)
    _, tangent = jax.jvp(mlp_reference, primals, tangents)
    chex.assert_equal_shape((result, tangent))
    return result, tangent


def gru_reference(parameters, x, hidden):
    """Map Equinox GRU parameter tuple, feature vector and hidden vector to next hidden."""
    wi, wh, bias, bias_n = parameters
    n_hidden = hidden.shape[0]
    igates = wi @ x + bias
    chex.assert_shape(igates, (3*n_hidden,))
    hgates = wh @ hidden
    chex.assert_shape(hgates, (3*n_hidden,))
    ir, iz, inn = jnp.split(igates, 3)
    hr, hz, hn = jnp.split(hgates, 3)
    chex.assert_shape((ir, iz, inn, hr, hz, hn), (n_hidden,))
    reset = jax.nn.sigmoid(ir+hr)
    chex.assert_shape(reset, (n_hidden,))
    update = jax.nn.sigmoid(iz+hz)
    chex.assert_shape(update, (n_hidden,))
    new = jnp.tanh(inn + reset*(hn+bias_n))
    chex.assert_shape(new, (n_hidden,))
    output = new + update*(hidden-new)
    chex.assert_shape(output, (n_hidden,))
    return output


@jax.custom_jvp
def fused_gru(parameters, x, hidden):
    """Fuse GRU matrix-vector products and gates; return next hidden [H]."""
    n_hidden = hidden.shape[0]
    n_input = x.shape[0]
    width = pl.next_power_of_2(max(n_hidden, n_input))

    def kernel(wi, wh, bias, bias_n, xx, hh, out):
        """Fuse three input/hidden affine maps and gates from refs into out [H]."""
        rows = jnp.arange(n_hidden)
        columns = jnp.arange(width)
        feature = pl.load(xx, (columns,), mask=columns<n_input, other=0.)
        chex.assert_shape(feature, (width,))
        old = pl.load(hh, (columns,), mask=columns<n_hidden, other=0.)
        chex.assert_shape(old, (width,))
        igates, hgates = [], []
        for gate in range(3):
            row = rows + gate*n_hidden
            input_weight = pl.load(wi, (row[:, None], columns[None, :]), mask=columns[None, :]<n_input, other=0.)
            chex.assert_shape(input_weight, (n_hidden, width))
            hidden_weight = pl.load(wh, (row[:, None], columns[None, :]), mask=columns[None, :]<n_hidden, other=0.)
            chex.assert_shape(hidden_weight, (n_hidden, width))
            ii = jnp.sum(input_weight*feature[None, :], axis=1) + pl.load(bias, (row,))
            chex.assert_shape(ii, (n_hidden,))
            hi = jnp.sum(hidden_weight*old[None, :], axis=1)
            chex.assert_shape(hi, (n_hidden,))
            igates.append(ii)
            hgates.append(hi)
        reset = jax.nn.sigmoid(igates[0]+hgates[0])
        chex.assert_shape(reset, (n_hidden,))
        update = jax.nn.sigmoid(igates[1]+hgates[1])
        chex.assert_shape(update, (n_hidden,))
        new = jnp.tanh(igates[2]+reset*(hgates[2]+pl.load(bias_n, (rows,))))
        chex.assert_shape(new, (n_hidden,))
        result = new+update*(pl.load(hh, (rows,))-new)
        chex.assert_shape(result, (n_hidden,))
        pl.store(out, (rows,), result)

    result = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((n_hidden,), hidden.dtype),
                            grid=(), name='fused_small_gru')(*parameters, x, hidden)
    chex.assert_shape(result, (n_hidden,))
    return result


@fused_gru.defjvp
def gru_jvp(primals, tangents):
    """Return fused GRU primal and exact JAX directional derivative of all arguments."""
    result = fused_gru(*primals)
    _, tangent = jax.jvp(gru_reference, primals, tangents)
    chex.assert_equal_shape((result, tangent))
    return result, tangent


class PallasMLP(eqx.Module):
    """Drop-in tanh MLP wrapper retaining the original ordered parameter leaves."""
    layers: tuple

    def __call__(self, x):
        """Map vector input to vector output via fused forward and JAX derivative."""
        return fused_mlp(tuple((layer.weight, layer.bias) for layer in self.layers), x)


class PallasGRU(eqx.Module):
    """Drop-in GRU wrapper retaining original parameters and cell metadata."""
    cell: eqx.nn.GRUCell

    def __call__(self, x, hidden):
        """Map feature/hidden vectors to next hidden with the fused GRU."""
        cell = self.cell
        return fused_gru((cell.weight_ih, cell.weight_hh, cell.bias, cell.bias_n), x, hidden)


def convert(model, gru=False, mlp=False):
    """Replace selected GRU/tanh MLP submodules in model PyTree; preserve parameter values."""
    def selected(node):
        """Return whether this PyTree node is one of the requested fused modules."""
        return (gru and isinstance(node, eqx.nn.GRUCell)) or (mlp and isinstance(node, eqx.nn.MLP))

    def replace(node):
        """Wrap selected original neural module; leave other leaves unchanged."""
        if gru and isinstance(node, eqx.nn.GRUCell):
            return PallasGRU(node)
        if mlp and isinstance(node, eqx.nn.MLP):
            return PallasMLP(node.layers)
        return node

    return jax.tree.map(replace, model, is_leaf=selected)
