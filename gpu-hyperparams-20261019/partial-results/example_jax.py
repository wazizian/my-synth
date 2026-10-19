"""GPU verification plus a small Equinox training job; writes outputs/."""
import json
import os
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import jaxlib

devices = jax.devices()
assert len(devices) == 1 and devices[0].platform == 'gpu', devices
assert '4090' in devices[0].device_kind, devices
model = eqx.nn.MLP(4, 1, 32, 2, key=jax.random.key(0))
x = jax.random.normal(jax.random.key(1), (128, 4))
y = jnp.sum(x, axis=1, keepdims=True)


@eqx.filter_jit
def loss(model):
    return jnp.mean((jax.vmap(model)(x) - y) ** 2)


@eqx.filter_jit
def step(model):
    value, grads = eqx.filter_value_and_grad(loss)(model)
    model = eqx.apply_updates(model, jax.tree.map(lambda g: -.01 * g, grads))
    return model, value


initial = float(loss(model))
out = Path(os.environ.get('JOB_OUTPUT_DIR', 'outputs'))
metrics = Path(os.environ.get('JOB_METRICS_DIR', out / 'metrics'))
checkpoints = Path(os.environ.get('JOB_CHECKPOINT_DIR', out / 'checkpoints'))
for directory in [out, metrics, checkpoints]:
    directory.mkdir(exist_ok=True, parents=True)
with (metrics / 'train.jsonl').open('w', buffering=1) as metric_file:
    for index in range(100):
        model, value = step(model)
        if (index + 1) % 10 == 0:
            metric_file.write(json.dumps({'step': index + 1, 'loss': float(value)}) + '\n')
        if (index + 1) % 25 == 0:
            target = checkpoints / f'step-{index + 1:06d}.eqx'
            temporary = checkpoints / f'.step-{index + 1:06d}.tmp.eqx'
            eqx.tree_serialise_leaves(temporary, model)
            os.replace(temporary, target)
final = float(loss(model))
assert jnp.isfinite(value).item() and final < initial, (initial, final)
result = dict(jax=jax.__version__, jaxlib=jaxlib.__version__,
              equinox=eqx.__version__, device=str(devices[0]),
              device_kind=devices[0].device_kind, initial_loss=initial, final_loss=final)
(out / 'result.json').write_text(json.dumps(result, indent=2))
print('PASS:', json.dumps(result), flush=True)

