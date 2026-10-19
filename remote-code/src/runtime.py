"""Numerical entrypoints are permitted only on a rented CUDA worker."""
import os
from pathlib import Path

def require_gpu():
    import jax
    devices=jax.devices()
    if jax.default_backend() != 'gpu' or any(d.platform != 'gpu' for d in devices):
        raise RuntimeError('GPU required: numerical work on the supervisor/CPU is prohibited')
    return devices

def configure(root):
    import json
    provenance=Path(root)/'runtime_metadata.json'
    if provenance.exists():
        for key,value in json.loads(provenance.read_text()).items():
            if key in ('INSTANCE_ID','INSTANCE_RATE_PER_HOUR','RENTAL_STARTED_AT','CAMPAIGN_COST_AT_LAUNCH'):os.environ.setdefault(key,str(value))
    import jax
    cache=Path(root)/'cache/jax'; cache.mkdir(parents=True,exist_ok=True); cache.chmod(0o700)
    jax.config.update('jax_compilation_cache_dir',str(cache))
    return require_gpu()
