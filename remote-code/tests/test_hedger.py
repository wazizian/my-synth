import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.hedger import Hedger, actions_one
from src.teachers import find_adam_state
from src.objective import portfolio_accounting, rollout, make_optimizer, student_block


@pytest.fixture(autouse=True)
def require_gpu():
    assert jax.default_backend() == 'gpu', 'Numerical tests require a GPU worker'


def test_analytic_accounting():
    prices = jnp.array([[1., .2], [1.1, .25], [1.3, .22]])
    holdings = jnp.array([[.5, 1.], [1., -.5]])
    gain, costs, liquidation, turnover = portfolio_accounting(prices, holdings, jnp.array([.01, .02]))
    np.testing.assert_allclose(gain, .05 + .05 + .2 + .015, rtol=1e-6)
    np.testing.assert_allclose(costs, [.005 + .0055 + .013, .004 + .0075 + .0022], rtol=1e-6)
    np.testing.assert_allclose(liquidation, [.013, .0022], rtol=1e-6)
    np.testing.assert_allclose(turnover, [2., 3.], rtol=1e-6)
    g, c, _, _ = portfolio_accounting(prices, jnp.zeros_like(holdings), jnp.array([.01, .02]))
    assert g == 0 and jnp.all(c == 0)


def test_constant_holding_two_dates():
    g, _, _, _ = portfolio_accounting(jnp.array([[1., .2], [1.3, .1]]), jnp.array([[2., -1.]]), jnp.zeros(2))
    np.testing.assert_allclose(g, .7, rtol=1e-6)


def test_causality_and_reset():
    model = Hedger(jax.random.key(0), hidden_size=4)
    spot = jnp.exp(jnp.linspace(0., .05, 85))
    call = jnp.linspace(.03, .04, 85)
    initial = actions_one(model, spot, call)
    altered = actions_one(model, spot.at[75:].multiply(1.2), call.at[75:].add(.1))
    np.testing.assert_array_equal(initial[:11], altered[:11])
    batch = dict(spot=jnp.stack((spot, spot)), call=jnp.stack((call, call)))
    _, diag = rollout(model, batch)
    np.testing.assert_array_equal(diag['holdings'][0], diag['holdings'][1])
    assert diag['holdings'].shape == (2, 20, 2)
    # Constant context doesn't generate any trading gain/cost: accounting has only 20 positions.
    assert jnp.all(jnp.abs(diag['holdings']) <= 2)


def test_carried_block_is_differentiable_in_weights():
    model = Hedger(jax.random.key(1), hidden_size=3)
    optimizer = make_optimizer(1e-3)
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    spot = jnp.stack((jnp.linspace(1., 1.1, 5), jnp.linspace(1., .9, 5)))
    call = .02 + .2 * (spot - 1)
    batch = dict(spot=jnp.stack((spot, spot)), call=jnp.stack((call, call)))
    config = dict(inception=2, intervals=2)
    weights = jnp.ones((2, 2))
    def target(w):
        (post, _), _ = student_block(model, state, batch, w, optimizer, config)
        return jnp.mean(rollout(post, dict(spot=spot, call=call), config)[0] ** 2)
    gradient = jax.grad(target)(weights)
    eps = .01
    direction = jnp.array([[1., -1.], [.3, -.3]])
    fd = (target(weights + eps * direction) - target(weights - eps * direction)) / (2 * eps)
    np.testing.assert_allclose(jnp.sum(gradient * direction), fd, rtol=.04, atol=1e-7)
    assert jnp.all(jnp.isfinite(gradient))
    # Shared scaling of a minibatch is exactly immaterial to normalized weights.
    np.testing.assert_allclose(gradient.sum(axis=1), 0, atol=1e-8)
    (_, state_post), losses = student_block(model, state, batch, weights, optimizer, config)
    assert int(find_adam_state(state_post).count) == 2
    assert losses.shape == (2,)


def test_optional_block_diagnostics_preserve_updates():
    model = Hedger(jax.random.key(10), hidden_size=3)
    optimizer = make_optimizer(.001)
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    spot = jnp.array([[[1., 1.01, 1.03, 1.04], [1., .99, .97, .96]]])
    batch = dict(spot=spot, call=.04 + .2 * (spot - 1))
    task = dict(inception=2, intervals=1)
    normal, normal_losses = student_block(model, state, batch, None, optimizer, task)
    observed, losses, diagnostics = student_block(model, state, batch, None, optimizer, task,
                                                  return_diagnostics=True)
    for left, right in zip(jax.tree.leaves(normal), jax.tree.leaves(observed)):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(losses, normal_losses)
    assert diagnostics['gradient_norm'].shape == (1,)
    assert jnp.all(jnp.isfinite(diagnostics['gradient_norm']))
    assert jnp.all(diagnostics['gradient_nonfinite_count'] == 0)
    assert not jnp.any(diagnostics['loss_nonfinite'])
    np.testing.assert_array_equal(diagnostics['clipping_applied'], diagnostics['gradient_norm'] > 1.)
