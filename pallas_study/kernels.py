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


def mlp_forward(parameters, x, saved=False):
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
        if saved:
            pl.store(output_ref, (0, indices), value)
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
            if saved:
                pl.store(output_ref, (layer+1, indices), value)
        if not saved:
            pl.store(output_ref, (indices,), value, mask=indices < n_output)

    flat = tuple(item for pair in parameters for item in pair)
    shape = (len(parameters)+1, width) if saved else (n_output,)
    result = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(shape, x.dtype),
                            grid=(), name='fused_small_mlp')(*flat, x)
    chex.assert_shape(result, shape)
    return result


@jax.custom_jvp
def fused_mlp(parameters, x):
    """Return fused MLP output with forward/reverse higher-order JAX derivative support."""
    return mlp_forward(parameters, x)


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


def gru_forward(parameters, x, hidden, saved=False):
    """Fuse GRU matrix-vector products and gates; return next hidden [H]."""
    n_hidden = hidden.shape[0]
    n_input = x.shape[0]
    input_width = pl.next_power_of_2(n_input)
    hidden_width = pl.next_power_of_2(n_hidden)

    def kernel(wi, wh, bias, bias_n, xx, hh, out):
        """Fuse three input/hidden affine maps and gates from refs into out [H]."""
        rows = jnp.arange(n_hidden)
        input_columns = jnp.arange(input_width)
        hidden_columns = jnp.arange(hidden_width)
        input_mask = None if input_width == n_input else input_columns<n_input
        input_other = None if input_mask is None else 0.
        hidden_mask = None if hidden_width == n_hidden else hidden_columns<n_hidden
        hidden_other = None if hidden_mask is None else 0.
        feature = pl.load(xx, (input_columns,), mask=input_mask, other=input_other)
        chex.assert_shape(feature, (input_width,))
        old = pl.load(hh, (hidden_columns,), mask=hidden_mask, other=hidden_other)
        chex.assert_shape(old, (hidden_width,))
        igates, hgates = [], []
        for gate in range(3):
            row = rows + gate*n_hidden
            input_weight = pl.load(wi, (row[:, None], input_columns[None, :]),
                mask=None if input_mask is None else input_mask[None, :], other=input_other)
            chex.assert_shape(input_weight, (n_hidden, input_width))
            hidden_weight = pl.load(wh, (row[:, None], hidden_columns[None, :]),
                mask=None if hidden_mask is None else hidden_mask[None, :], other=hidden_other)
            chex.assert_shape(hidden_weight, (n_hidden, hidden_width))
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
        if saved:
            for index, value in enumerate((result, reset, update, new, hgates[2])):
                pl.store(out, (index, rows), value)
        else:
            pl.store(out, (rows,), result)

    shape = (5, n_hidden) if saved else (n_hidden,)
    result = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(shape, hidden.dtype),
                            grid=(), name='fused_small_gru')(*parameters, x, hidden)
    chex.assert_shape(result, shape)
    return result


@jax.custom_jvp
def fused_gru(parameters, x, hidden):
    """Return fused GRU output with forward/reverse higher-order JAX derivative support."""
    return gru_forward(parameters, x, hidden)


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
    saved_backward: bool = eqx.field(static=True, default=False)
    analytic_jvp: bool = eqx.field(static=True, default=False)
    tensorcore: bool = eqx.field(static=True, default=False)

    def __call__(self, x):
        """Map vector input to vector output via fused forward and JAX derivative."""
        function = fast_mlp if self.analytic_jvp else saved_mlp if self.saved_backward else fused_mlp
        if self.tensorcore:
            from pallas_study.tensorcore import tensorcore_mlp
            function = tensorcore_mlp
        return function(tuple((layer.weight, layer.bias) for layer in self.layers), x)


class PallasGRU(eqx.Module):
    """Drop-in GRU wrapper retaining original parameters and cell metadata."""
    cell: eqx.nn.GRUCell
    saved_backward: bool = eqx.field(static=True, default=False)
    analytic_jvp: bool = eqx.field(static=True, default=False)

    def __call__(self, x, hidden):
        """Map feature/hidden vectors to next hidden with the fused GRU."""
        cell = self.cell
        function = fast_gru if self.analytic_jvp else saved_gru if self.saved_backward else fused_gru
        return function((cell.weight_ih, cell.weight_hh, cell.bias, cell.bias_n), x, hidden)


def convert(model, gru=False, mlp=False, saved_backward=False, analytic_jvp=False, tensorcore=False):
    """Replace selected GRU/tanh MLP submodules in model PyTree; preserve parameter values."""
    def selected(node):
        """Return whether this PyTree node is one of the requested fused modules."""
        return (gru and isinstance(node, eqx.nn.GRUCell)) or (mlp and isinstance(node, eqx.nn.MLP))

    def replace(node):
        """Wrap selected original neural module; leave other leaves unchanged."""
        if gru and isinstance(node, eqx.nn.GRUCell):
            return PallasGRU(node, saved_backward, analytic_jvp)
        if mlp and isinstance(node, eqx.nn.MLP):
            return PallasMLP(node.layers, saved_backward, analytic_jvp, tensorcore)
        return node

    return jax.tree.map(replace, model, is_leaf=selected)


@jax.custom_vjp
def saved_mlp(parameters, x):
    """Fused MLP with saved activations and analytic reverse derivative; first-order prototype."""
    return mlp_forward(parameters, x)


def saved_mlp_fwd(parameters, x):
    """Return MLP output and padded activations needed by reverse differentiation."""
    activations = mlp_forward(parameters, x, saved=True)
    n_output = parameters[-1][0].shape[0]
    result = activations[-1, :n_output]
    chex.assert_shape(result, (n_output,))
    return result, (parameters, activations)


def saved_mlp_bwd(residual, cotangent):
    """Map output cotangent and saved layers to parameter/input cotangents."""
    parameters, activations = residual
    gradients = []
    for index in reversed(range(len(parameters))):
        weight, _ = parameters[index]
        n_output, n_input = weight.shape
        if index < len(parameters)-1:
            cotangent = cotangent*(1-activations[index+1, :n_output]**2)
            chex.assert_shape(cotangent, (n_output,))
        weight_gradient = cotangent[:, None]*activations[index, None, :n_input]
        chex.assert_shape(weight_gradient, (n_output, n_input))
        gradients.append((weight_gradient, cotangent))
        cotangent = weight.T @ cotangent
        chex.assert_shape(cotangent, (n_input,))
    return tuple(reversed(gradients)), cotangent


saved_mlp.defvjp(saved_mlp_fwd, saved_mlp_bwd)


@jax.custom_vjp
def saved_gru(parameters, x, hidden):
    """Fused GRU with saved gates and analytic reverse derivative; first-order prototype."""
    return gru_forward(parameters, x, hidden)


def saved_gru_fwd(parameters, x, hidden):
    """Return next hidden and saved gate activations for the reverse pass."""
    activations = gru_forward(parameters, x, hidden, saved=True)
    return activations[0], (parameters, x, hidden, activations)


def saved_gru_bwd(residual, cotangent):
    """Compute exact GRU parameter, feature and hidden cotangents from saved gates."""
    parameters, x, hidden, activations = residual
    wi, wh, _, bias_n = parameters
    _, reset, update, new, hn = activations
    n_hidden = hidden.shape[0]
    n_input = x.shape[0]
    new_gradient = cotangent*(1-update)*(1-new**2)
    chex.assert_shape(new_gradient, (n_hidden,))
    update_gradient = cotangent*(hidden-new)*update*(1-update)
    chex.assert_shape(update_gradient, (n_hidden,))
    reset_gradient = new_gradient*(hn+bias_n)*reset*(1-reset)
    chex.assert_shape(reset_gradient, (n_hidden,))
    hn_gradient = new_gradient*reset
    chex.assert_shape(hn_gradient, (n_hidden,))
    ig = jnp.concatenate((reset_gradient, update_gradient, new_gradient))
    hg = jnp.concatenate((reset_gradient, update_gradient, hn_gradient))
    chex.assert_shape((ig, hg), (3*n_hidden,))
    wig = ig[:, None]*x[None, :]
    chex.assert_shape(wig, (3*n_hidden, n_input))
    whg = hg[:, None]*hidden[None, :]
    chex.assert_shape(whg, (3*n_hidden, n_hidden))
    xg = wi.T @ ig
    chex.assert_shape(xg, (n_input,))
    hidden_gradient = wh.T @ hg + cotangent*update
    chex.assert_shape(hidden_gradient, (n_hidden,))
    return (wig, whg, ig, hn_gradient), xg, hidden_gradient


saved_gru.defvjp(saved_gru_fwd, saved_gru_bwd)


@jax.custom_jvp
def mlp_activations(parameters, x):
    """Return padded input/layer activations from fused kernel, with differentiable residuals."""
    return mlp_forward(parameters, x, saved=True)


@mlp_activations.defjvp
def mlp_activation_jvp(primals, tangents):
    """Propagate tangents using saved activations instead of recomputing primal affine maps."""
    parameters, x = primals
    parameter_tangents, tangent = tangents
    activations = mlp_activations(parameters, x)
    width = activations.shape[1]
    tangent_rows = [jnp.pad(tangent, (0, width-tangent.shape[0]))]
    chex.assert_shape(tangent_rows[0], (width,))
    for index, ((weight, _), (dweight, dbias)) in enumerate(zip(parameters, parameter_tangents)):
        n_output, n_input = weight.shape
        tangent = weight @ tangent + dweight @ activations[index, :n_input] + dbias
        chex.assert_shape(tangent, (n_output,))
        if index < len(parameters)-1:
            tangent = tangent*(1-activations[index+1, :n_output]**2)
            chex.assert_shape(tangent, (n_output,))
        padded = jnp.pad(tangent, (0, width-n_output))
        chex.assert_shape(padded, (width,))
        tangent_rows.append(padded)
    result = jnp.stack(tangent_rows)
    chex.assert_equal_shape((result, activations))
    return activations, result


@jax.custom_jvp
def fast_mlp(parameters, x):
    """Return last fused activation; analytic JVP supports reverse and higher-order training."""
    return mlp_forward(parameters, x)


@fast_mlp.defjvp
def fast_mlp_jvp(primals, tangents):
    """Expose only the final tangent, keeping intermediate activation tangents out of first-order AD."""
    parameters, x = primals
    parameter_tangents, tangent = tangents
    activations = mlp_activations(parameters, x)
    for index, ((weight, _), (dweight, dbias)) in enumerate(zip(parameters, parameter_tangents)):
        n_output, n_input = weight.shape
        tangent = weight @ tangent + dweight @ activations[index, :n_input] + dbias
        chex.assert_shape(tangent, (n_output,))
        if index < len(parameters)-1:
            tangent = tangent*(1-activations[index+1, :n_output]**2)
            chex.assert_shape(tangent, (n_output,))
    result = activations[-1, :parameters[-1][0].shape[0]]
    chex.assert_equal_shape((result, tangent))
    return result, tangent


@jax.custom_jvp
def gru_activations(parameters, x, hidden):
    """Return fused GRU state/gates with differentiable saved intermediates."""
    return gru_forward(parameters, x, hidden, saved=True)


@gru_activations.defjvp
def gru_activation_jvp(primals, tangents):
    """Propagate GRU state/gate tangents from saved gates and affine-map directional derivatives."""
    parameters, x, hidden = primals
    dparameters, dx, dh = tangents
    wi, wh, _, bn = parameters
    dwi, dwh, db, dbn = dparameters
    n_hidden = hidden.shape[0]
    activations = gru_activations(parameters, x, hidden)
    _, reset, update, new, hn = activations
    di = dwi @ x + wi @ dx + db
    chex.assert_shape(di, (3*n_hidden,))
    dhr = dwh @ hidden + wh @ dh
    chex.assert_shape(dhr, (3*n_hidden,))
    dir_, diz, din = jnp.split(di, 3)
    dhr_, dhz, dhn = jnp.split(dhr, 3)
    dr = reset*(1-reset)*(dir_+dhr_)
    chex.assert_shape(dr, (n_hidden,))
    dz = update*(1-update)*(diz+dhz)
    chex.assert_shape(dz, (n_hidden,))
    dn = (1-new**2)*(din+dr*(hn+bn)+reset*(dhn+dbn))
    chex.assert_shape(dn, (n_hidden,))
    dout = dn+dz*(hidden-new)+update*(dh-dn)
    chex.assert_shape(dout, (n_hidden,))
    result = jnp.stack((dout, dr, dz, dn, dhn))
    chex.assert_equal_shape((result, activations))
    return activations, result


@jax.custom_jvp
def fast_gru(parameters, x, hidden):
    """Return fused GRU state with analytic, higher-order-compatible JVP."""
    return gru_forward(parameters, x, hidden)


@fast_gru.defjvp
def fast_gru_jvp(primals, tangents):
    """Expose only next-state tangents while retaining differentiable gate residuals."""
    parameters, x, hidden = primals
    dparameters, dx, dh = tangents
    wi, wh, _, bn = parameters
    dwi, dwh, db, dbn = dparameters
    n_hidden = hidden.shape[0]
    activations = gru_activations(parameters, x, hidden)
    output, reset, update, new, hn = activations
    di = dwi @ x + wi @ dx + db
    chex.assert_shape(di, (3*n_hidden,))
    dhr = dwh @ hidden + wh @ dh
    chex.assert_shape(dhr, (3*n_hidden,))
    dir_, diz, din = jnp.split(di, 3)
    dhr_, dhz, dhn = jnp.split(dhr, 3)
    dr = reset*(1-reset)*(dir_+dhr_)
    chex.assert_shape(dr, (n_hidden,))
    dz = update*(1-update)*(diz+dhz)
    chex.assert_shape(dz, (n_hidden,))
    dn = (1-new**2)*(din+dr*(hn+bn)+reset*(dhn+dbn))
    chex.assert_shape(dn, (n_hidden,))
    dout = dn+dz*(hidden-new)+update*(dh-dn)
    chex.assert_shape(dout, (n_hidden,))
    return output, dout
