"""Canonical pickle codec for SDE programs, independent of CLI entry points."""
from functools import lru_cache
import pickle

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .config import SDEConfig
from .dynamics import NeuralSDE


@lru_cache(maxsize=16)
def teacher_layout(specification):
    """Rebuild a cached architecture template from (latent,width,depth) metadata.

    Initialization supplies only a PyTree layout and static activation functions;
    every numerical value is replaced with the saved snapshot before use.
    """
    n_latent, width, depth = specification
    template = NeuralSDE(jax.random.key(0), SDEConfig(n_latent=n_latent), width, depth)
    return eqx.partition(template, eqx.is_inexact_array)


def restore_saved_teacher(specification, values):
    """Restore one frozen SDE from architecture metadata and numerical leaves."""
    template, static = teacher_layout(specification)
    expected, structure = jax.tree.flatten(template)
    if len(expected) != len(values) or any(
        old.shape != new.shape or old.dtype != new.dtype for old, new in zip(values, expected)
    ):
        raise RuntimeError("Saved SDE program has invalid numerical layout")
    arrays = [jnp.asarray(value) for value in values]
    return eqx.combine(jax.tree.unflatten(structure, arrays), static)


class CheckpointPickler(pickle.Pickler):
    """Serialize frozen SDE programs as numerical leaves, never activation code."""

    def reducer_override(self, value):
        """Reduce a NeuralSDE to a versioned architecture and host arrays; defer others."""
        if isinstance(value, NeuralSDE):
            networks = (value.latent_q_drift, value.diffusion_network, value.risk_price)
            # Parameters and Adam moment trees share the module type but have
            # their callable leaves replaced by None. Preserve those partitions
            # verbatim; only complete executable programs need reconstruction.
            if all(network.activation is None and network.final_activation is None
                   for network in networks):
                return NotImplemented
            width, depth = networks[0].width_size, networks[0].depth
            if any(network.width_size != width or network.depth != depth for network in networks):
                raise ValueError("Inconsistent neural SDE architecture")
            specification = (value.risk_price.out_size - 1, width, depth)
            params = eqx.filter(value, eqx.is_inexact_array)
            arrays = [np.asarray(jax.device_get(leaf)) for leaf in jax.tree.leaves(params)]
            return restore_saved_teacher, (specification, arrays)
        return NotImplemented

