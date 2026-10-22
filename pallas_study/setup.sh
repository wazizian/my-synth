#!/bin/bash
set -euo pipefail
cd /workspace/pallas
export UV_CACHE_DIR=/workspace/uv-cache
curl -LsSf https://astral.sh/uv/install.sh -o /workspace/install-uv.sh
UV_UNMANAGED_INSTALL=/workspace/uv sh /workspace/install-uv.sh
/workspace/uv/uv venv --python 3.12 /workspace/pallas-env
/workspace/uv/uv pip install --python /workspace/pallas-env/bin/python 'jax[cuda12]==0.7.2' 'equinox==0.13.2' 'chex==0.1.91' 'optax==0.2.6' 'triton==3.4.0' pytest
/workspace/uv/uv pip freeze --python /workspace/pallas-env/bin/python > outputs/environment.txt
nvidia-smi > outputs/nvidia-smi.txt
touch outputs/setup.done
