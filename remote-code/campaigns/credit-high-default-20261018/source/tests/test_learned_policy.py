import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from credit_hedging.learned_policy import TargetHedger
from credit_hedging.diagnostic_policies import FixedProtection
from credit_hedging.curve_market import CurveConfig,build_curve_tables,simulate_curve_paths
from credit_hedging.environment import rollout
from credit_hedging.learned_run import update


def test_target_policy_initially_matches_replenishment_and_has_learning_signal():
    c=CurveConfig(bond_days=12,cds_days=3,grid_size=31,fast_vol=0,fast_grid_size=1)
    t=build_curve_tables(c);p=simulate_curve_paths(jax.random.key(12),t,c,32)
    m=TargetHedger(jax.random.key(1),3)
    a=rollout(m,p,t,c);b=rollout(FixedProtection(1.,n_vintages=3),p,t,c)
    np.testing.assert_array_equal(a['loss'],b['loss'])
    np.testing.assert_array_equal(a['records']['notional'],b['records']['notional'])
    grad=eqx.filter_grad(lambda m:jnp.mean(rollout(m,p,t,c)['loss']**2))(m)
    assert float(optax.global_norm(grad))>1e-10


def test_target_policy_checkpoint_replays_same_update(tmp_path):
    c=CurveConfig(bond_days=12,cds_days=3,grid_size=31,fast_vol=0,fast_grid_size=1)
    t={'easy':build_curve_tables(c)};cfg={'easy':c};m=TargetHedger(jax.random.key(1),3)
    opt=optax.chain(optax.clip_by_global_norm(1.),optax.adam(.001));s=opt.init(eqx.filter(m,eqx.is_inexact_array));key=jax.random.key(3)
    path=tmp_path/'state.eqx';eqx.tree_serialise_leaves(path,(m,s,jax.random.key_data(key)))
    mm,ss,kk=eqx.tree_deserialise_leaves(path,(m,s,jax.random.key_data(key)))
    a=update(m,s,key,t,cfg,('easy',),(16,),(1.,),opt)
    b=update(mm,ss,jax.random.wrap_key_data(kk),t,cfg,('easy',),(16,),(1.,),opt)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
        if eqx.is_array(x) and not jax.dtypes.issubdtype(x.dtype,jax.dtypes.prng_key):np.testing.assert_array_equal(x,y)
