"""Tiny GPU-only resume and integrity checks; no supervisor execution."""
import json
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.checkpoint import save_checkpoint, load_checkpoint
from src.hedger import Hedger
from src.teachers import find_adam_state
from src.objective import make_optimizer, student_block
from src.metrics import summarize_outcomes, aggregate_domains, save_episode_metrics


@pytest.fixture(autouse=True)
def require_gpu():
    assert jax.default_backend() == 'gpu', 'Numerical tests require a GPU worker'


def assert_tree_equal(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        if isinstance(a, jax.Array) and jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_array_equal(a, b)


def test_full_resume_continues_adam_and_keys(tmp_path):
    model = Hedger(jax.random.key(1), hidden_size=3)
    optimizer = make_optimizer(.001)
    opt = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    spot = jnp.array([[1., 1.01, 1.02, 1.03, 1.04], [1., .99, .98, .97, .96]])
    batch = dict(spot=spot[None], call=(.04 + .2 * (spot - 1))[None])
    task = dict(inception=2, intervals=2)
    (model, opt), _ = student_block(model, opt, batch, None, optimizer, task)
    state = dict(model=model, optimizer=opt, generator=jnp.zeros(12),
                 keys={'train': jax.random.key(12), 'teacher': jax.random.key(13)},
                 history=(eqx.filter(model, eqx.is_inexact_array),), step=jnp.array(1),
                 reward_ema=jnp.array(.2))
    first = save_checkpoint(tmp_path, state, {'hashes': {'pricing': 'abc'}, 'architecture': {'hidden_size': 3}})
    restored, meta = load_checkpoint(tmp_path, state, expected_hashes={'pricing': 'abc'})
    assert meta['checkpoint_id'] == first['checkpoint_id']
    assert_tree_equal(state, restored)
    direct, _ = student_block(state['model'], state['optimizer'], batch, None, optimizer, task)
    resumed, _ = student_block(restored['model'], restored['optimizer'], batch, None, optimizer, task)
    assert_tree_equal(direct, resumed)
    assert int(find_adam_state(resumed[1]).count) == 2
    for key in state['keys']:
        assert_tree_equal(jax.random.split(state['keys'][key]), jax.random.split(restored['keys'][key]))
    second = save_checkpoint(tmp_path, restored, {'hashes': {'pricing': 'abc'}})
    assert second['checkpoint_id'] != first['checkpoint_id']
    assert json.loads((tmp_path / 'previous.json').read_text())['checkpoint_id'] == first['checkpoint_id']
    with pytest.raises(ValueError, match='input hash'):
        load_checkpoint(tmp_path, state, expected_hashes={'pricing': 'wrong'})
    with open(tmp_path / second['checkpoint_id'] / 'state.eqx', 'ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='checksum'):
        load_checkpoint(tmp_path, state)
    previous, _ = load_checkpoint(tmp_path / 'previous.json', state)
    assert_tree_equal(previous, state)


def test_metrics_and_raw_archive(tmp_path):
    values = jnp.array([-1., 0., 1., 2.])
    diagnostics = {'costs': jnp.ones((4, 2)) * .01, 'turnover': jnp.ones((4, 2)),
                   'liquidation_costs': jnp.zeros((4, 2)),
                   'mean_abs_holdings': jnp.ones((4, 2)), 'saturation_fraction': jnp.zeros(4)}
    kwargs = dict(scale=2., domain='easy', split_role='monitor', data_hash='data', checkpoint_id='cp')
    result = summarize_outcomes(values, diagnostics, **kwargs)
    assert result['raw_mse'] == 1.5
    assert result['scaled_mse'] == .75
    assert result['mean_pnl'] == -.5
    assert result['es95'] == 2.
    assert result['status'] == 'ok'
    assert aggregate_domains({'easy': result, 'hard': result})['balanced_objective'] == .75
    bad = summarize_outcomes(values.at[0].set(jnp.nan), diagnostics, **kwargs)
    assert bad['raw_mse'] is None and bad['nonfinite_count'] == 1
    save_episode_metrics(tmp_path / 'outcomes.npz', values, diagnostics)
    with np.load(tmp_path / 'outcomes.npz') as archive:
        np.testing.assert_array_equal(archive['pnl'], -np.asarray(values))
        assert set(archive.files) == set(diagnostics) | {'shortfall', 'pnl'}
