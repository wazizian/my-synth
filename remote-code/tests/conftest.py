# Collection itself must fail before numerical tests can execute on a CPU.
import os
os.environ.setdefault('JAX_PLATFORMS','cuda')
import jax
jax.config.update('jax_enable_x64',True)
from src.runtime import require_gpu
require_gpu()
