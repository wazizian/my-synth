import jax
import jax.numpy as jnp
from src.protocol import STREAMS,stream_key

def test_prng_namespaces_disjoint():
    keys=[jax.random.key_data(stream_key(role,i)) for role in STREAMS for i in range(12)]
    packed=jnp.stack(keys)
    equal=jnp.all(packed[:,None,:]==packed[None,:,:],axis=-1)
    assert int(jnp.sum(equal))==len(keys)
