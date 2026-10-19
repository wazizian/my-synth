import numpy as np
import jax
import jax.numpy as jnp
import pytest
from src.pricing import require_gpu

@pytest.fixture(autouse=True)
def gpu_only():
    jax.config.update("jax_enable_x64", True)
    require_gpu()

from src.pricing import heston_call, black_scholes_call, build_price_tables, interpolate_prices
from src.market import simulate_latent, DEFAULT_CONFIG, REGIMES, mixture_probabilities


def test_frozen_benchmark_mixtures():
    for q, dominant, mass in zip(mixture_probabilities(), (3,0,11), (.7,.8,.8)):
        q=np.asarray(q)
        assert np.all(q>0)
        np.testing.assert_allclose(q.sum(),1.,atol=1e-7)
        np.testing.assert_allclose(q[dominant],mass)
        np.testing.assert_allclose(np.delete(q,dominant),(1-mass)/11)
    assert REGIMES[3]==(2.,.04,.4,-.8)
    assert REGIMES[0]==(2.,.04,.2,-.3)
    assert REGIMES[11]==(2.,.09,.6,-.8)


def test_fourier_bounds_monotonicity_and_quadrature():
    for xi in (.2,.6):
        for rho in (-.3,-.8):
            s=np.linspace(.65,1.4,25)
            c=heston_call(s,1.,.24,.09,xi=xi,rho=rho)
            fine=heston_call(s,1.,.24,.09,xi=xi,rho=rho,nodes=384,cutoff=350)
            assert np.max(np.abs(c-fine))<2e-6
            assert np.all(c>=np.maximum(s-1,0)-1e-7)
            assert np.all(c<=s+1e-7)
            assert np.all(np.diff(c)>=0)


def test_zero_vol_of_vol_limit():
    s=np.array([.8,1.,1.2]); t=.3; v=.08; bar=.04; kap=2.
    w=bar*t+(v-bar)*(1-np.exp(-kap*t))/kap
    expected=black_scholes_call(s,1.,w)
    assert np.max(np.abs(heston_call(s,1.,t,v,vbar=bar,xi=1e-5)-expected))<2e-6


def test_simulation_determinism_positivity_and_drift():
    ids=jnp.arange(8192)%12
    a,v=simulate_latent(ids,jax.random.key(1),DEFAULT_CONFIG)
    b,_=simulate_latent(ids,jax.random.key(1),DEFAULT_CONFIG)
    np.testing.assert_array_equal(a,b)
    assert a.shape==(8192,85)
    assert np.all(np.asarray(v)>=0)
    # 4 standard errors plus conservative weak-discretization allowance.
    end=np.asarray(a[:,-1]); assert abs(end.mean()-1)<4*end.std()/np.sqrt(len(end))+.001


def test_interpolation_and_outside_rejection():
    regimes=np.array([[2.,.04,.2,-.3]])
    tables=build_price_tables(regimes,np.linspace(-.2,.2,33),np.linspace(0,.3,25)**2,days=[60],nodes=192)
    assert bool(tables['projection_applied'])
    assert float(tables['projection_max_abs_correction']) <= 2e-6
    assert bool(jnp.all(tables['values'] >= 0))
    s=jnp.array([[1.03],[1.1],[4.]])
    v=jnp.array([[.04],[.045],[.04]])
    c,bad,dist=interpolate_prices(jnp.zeros(3,dtype=int),s,v,tables,strike=1.)
    direct=heston_call(np.asarray(s[:2]),1.,60/252,np.asarray(v[:2]))
    assert np.max(np.abs(np.asarray(c[:2])-direct))<1e-4
    assert bool(bad[2,0]) and np.isnan(c[2,0]) and dist[2,0]>0


def test_loaded_fallback_preserves_table_identity_and_repairs_tails(tmp_path):
    from src.pricing import save_tables,load_tables,table_hash
    table=build_price_tables([(2.,.04,.2,-.3)],jnp.linspace(-.2,.2,33),
                             jnp.linspace(0,.3,25)**2,days=[60])
    expected=table_hash(table)
    path=tmp_path/'table.npz'
    save_tables(path,table)
    with jax.enable_x64(False):
        loaded=load_tables(path)
        s=jnp.array([[.7],[1.],[1.8]],jnp.float32)
        v=jnp.array([[.04],[.04],[1.3]],jnp.float32)
        ids=jnp.zeros(3,jnp.int32)
        @jax.jit
        def paired(ids,s,v,loaded):
            plain_table={k:value for k,value in loaded.items() if not k.startswith('_fallback_')}
            return (interpolate_prices(ids,s,v,plain_table,strike=1.),
                    interpolate_prices(ids,s,v,loaded,strike=1.))
        (plain,mask,_),(prices,bad,_)=paired(ids,s,v,loaded)
        assert prices.dtype==jnp.float32
        assert bool(jnp.all(jnp.isfinite(prices)))
        assert bool(jnp.all(mask==bad))
        assert bool(jnp.all(jnp.where(bad,True,prices==plain)))
    assert table_hash(loaded)==expected
    save_tables(tmp_path/'roundtrip.npz',loaded)
    assert table_hash(load_tables(tmp_path/'roundtrip.npz'))==expected
