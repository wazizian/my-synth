"""GPU Heston Fourier prices and homogeneous lookup tables.

Negative-exponential (little-trap) characteristic function; direct prices and
quadrature use float64 on GPU. Loaded tables include a bounded float64
direct-pricing fallback with float32 outputs; failed fallback checks produce NaN.
https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/HestonTrap.pdf
"""
from pathlib import Path
import hashlib
import os
from functools import lru_cache
import numpy as np  # serialization only; numerical kernels are JAX
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr


def require_gpu():
    """Fail before numerical work if the selected default backend is CPU."""
    if jax.default_backend() != 'gpu' or any(d.platform != 'gpu' for d in jax.devices()):
        raise RuntimeError('Numerical work requires a GPU worker; CPU fallback is prohibited.')


@lru_cache(maxsize=8)
def quadrature(nodes, cutoff):
    require_gpu()
    if not jax.config.x64_enabled:
        raise RuntimeError('Reference pricing requires JAX_ENABLE_X64=1 on the GPU worker.')
    # Golub-Welsch: eigenpairs of the Legendre Jacobi matrix, entirely on GPU.
    k = jnp.arange(1, nodes, dtype=jnp.float64)
    b = k / jnp.sqrt(4*k*k-1)
    x, vectors = jnp.linalg.eigh(jnp.diag(b, 1)+jnp.diag(b, -1))
    return (x+1)*cutoff/2, cutoff*vectors[0]**2


def black_scholes_call(spot, strike, variance_time):
    require_gpu()
    s, k, w = jnp.broadcast_arrays(spot, strike, variance_time)
    root = jnp.sqrt(jnp.maximum(w, 1e-30))
    d = jnp.log(s/k)/root
    return jnp.where(w > 1e-20, s*ndtr(d+.5*root)-k*ndtr(d-.5*root), jnp.maximum(s-k, 0))


def characteristic(u, maturity, variance, kappa, vbar, xi, rho):
    """Characteristic function of log(S_T/S_now), zero carry."""
    require_gpu()
    u = jnp.asarray(u, dtype=jnp.complex128)
    t, v = jnp.asarray(maturity), jnp.asarray(variance)
    if xi < 1e-6:
        integrated = vbar*t+(v-vbar)*(-jnp.expm1(-kappa*t))/kappa
        return jnp.exp(-.5*(u*u+1j*u)*integrated)
    b = kappa-rho*xi*1j*u
    d = jnp.sqrt(b*b+xi*xi*(u*u+1j*u))
    g = (b-d)/(b+d)
    e = jnp.exp(-d*t)
    c = kappa*vbar/(xi*xi)*((b-d)*t-2*jnp.log((1-g*e)/(1-g)))
    D = (b-d)/(xi*xi)*(1-e)/(1-g*e)
    return jnp.exp(c+D*v)


def heston_call(spot, strike, maturity, variance, kappa=2., vbar=.04, xi=.2, rho=-.3, nodes=192, cutoff=220., quadrature_rule=None):
    """Unclipped vectorized GPU Gauss-Legendre P1/P2 inversion."""
    require_gpu()
    if not jax.config.x64_enabled:
        raise RuntimeError('Direct Heston pricing requires JAX_ENABLE_X64=1.')
    s,k,t,v = jnp.broadcast_arrays(*[jnp.asarray(a,jnp.float64) for a in (spot,strike,maturity,variance)])
    if xi < 1e-6:
        return black_scholes_call(s,k,vbar*t+(v-vbar)*(-jnp.expm1(-kappa*t))/kappa)
    u,w = quadrature(nodes,cutoff) if quadrature_rule is None else quadrature_rule
    phase = jnp.exp(1j*jnp.log(s/k)[...,None]*u)
    args = (t[...,None],v[...,None],kappa,vbar,xi,rho)
    p2 = .5+jnp.sum(w*jnp.real(phase*characteristic(u,*args)/(1j*u)),axis=-1)/jnp.pi
    p1 = .5+jnp.sum(w*jnp.real(phase*characteristic(u-1j,*args)/(1j*u)),axis=-1)/jnp.pi
    return jnp.where(t<=0,jnp.maximum(s-k,0),s*p1-k*p2)


def build_price_tables(regimes, log_moneyness=None, variance_grid=None, days=None, nodes=192, cutoff=220.):
    """GPU build in bounded maturity chunks, factorizing the rectangular grid."""
    require_gpu()
    lm=jnp.asarray(log_moneyness if log_moneyness is not None else jnp.linspace(-1.,.7,96),jnp.float64)
    vg=jnp.asarray(variance_grid if variance_grid is not None else jnp.linspace(0,jnp.sqrt(.8),48)**2,jnp.float64)
    day_list=list(range(124,39,-1)) if days is None else list(days)
    u,w=quadrature(nodes,cutoff)
    phase=jnp.exp(1j*lm[:,None]*u)
    values=[]
    for kap,bar,xi,rho in regimes:
        slices=[]
        for day in day_list:
            args=(day/252.,vg[:,None],kap,bar,xi,rho)
            p2=.5+jnp.real(phase @ (characteristic(u,*args)*w/(1j*u)).T)/jnp.pi
            p1=.5+jnp.real(phase @ (characteristic(u-1j,*args)*w/(1j*u)).T)/jnp.pi
            slices.append((jnp.exp(lm)[:,None]*p1-p2).block_until_ready())
        values.append(jnp.stack(slices))
    raw=jnp.stack(values)
    spot=jnp.exp(lm)[None,None,:,None]
    lower=jnp.maximum(spot-1,0)
    raw_lower=jnp.maximum(lower-raw,0).max()
    raw_upper=jnp.maximum(raw-spot,0).max()
    raw_monotonicity=jnp.maximum(-jnp.diff(raw,axis=2),0).max()
    # Only repair errors already within the declared quadrature tolerance.
    # Larger violations remain visible and fail the validation gate. This is
    # a numerical no-arbitrage projection, never a floor in transaction costs.
    tolerance=2e-6
    admissible=jnp.maximum(raw_lower,raw_upper)<=tolerance
    projected=jnp.minimum(jnp.maximum(raw,lower),spot)
    values=jnp.where(admissible,projected,raw)
    return {'log_moneyness':lm,'variance_grid':vg,'days':jnp.asarray(day_list),
            'values':values,'regimes':jnp.asarray(regimes),
            'raw_intrinsic_bound_violation':raw_lower,
            'raw_upper_bound_violation':raw_upper,
            'raw_spot_monotonicity_violation':raw_monotonicity,
            'projection_applied':admissible,
            'projection_max_abs_correction':jnp.max(jnp.abs(values-raw)),
            'projection_tolerance':jnp.asarray(tolerance)}


def table_hash(tables):
    h=hashlib.sha256()
    for k in sorted(k for k in tables if not k.startswith('_fallback_')):
        h.update(k.encode()); h.update(np.ascontiguousarray(tables[k]).tobytes())
    return h.hexdigest()


def save_tables(path,tables):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**{k:np.asarray(v) for k,v in tables.items() if not k.startswith('_fallback_')})
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)
    return table_hash(tables)


def load_tables(path):
    require_gpu()
    with np.load(path) as f: tables={k:f[k] for k in f.files}
    return with_fallback(tables)


def interpolate_prices(regime_ids, spot, variance, tables, strike=1.05):
    """Bilinear interpolation at exact daily maturity nodes, JAX pure."""
    require_gpu()
    if tables['values'].shape[1] != spot.shape[1]:
        raise ValueError('Pricing maturity nodes must match the complete fixed-contract path.')
    lm=jnp.asarray(tables['log_moneyness'],spot.dtype); vg=jnp.asarray(tables['variance_grid'],spot.dtype)
    x=jnp.log(spot/strike); v=variance
    ix=jnp.clip(jnp.searchsorted(lm,x)-1,0,lm.size-2)
    iv=jnp.clip(jnp.searchsorted(vg,v)-1,0,vg.size-2)
    wx=(x-lm[ix])/(lm[ix+1]-lm[ix]); wv=(v-vg[iv])/(vg[iv+1]-vg[iv])
    vals=jnp.asarray(tables['values'],spot.dtype); r=regime_ids[:,None]; ti=jnp.arange(spot.shape[1])[None,:]
    c00=vals[r,ti,ix,iv]; c10=vals[r,ti,ix+1,iv]; c01=vals[r,ti,ix,iv+1]; c11=vals[r,ti,ix+1,iv+1]
    c=strike*((1-wx)*((1-wv)*c00+wv*c01)+wx*((1-wv)*c10+wv*c11))
    dist=jnp.maximum(jnp.maximum(lm[0]-x,x-lm[-1]),0)+jnp.maximum(jnp.maximum(vg[0]-v,v-vg[-1]),0)
    bad=dist>0
    prices=jnp.where(bad,jnp.nan,c)
    if '_fallback_u' in tables:
        prices=jax.lax.cond(jnp.any(bad),
            lambda _: _replace_outside(regime_ids,spot,variance,prices,bad,tables,strike),
            lambda _: prices,operand=None)
    return prices,bad,dist


@lru_cache(maxsize=8)
def composite_quadrature(order,panel_width,cutoff):
    """Float64 GPU Gauss panels resolve oscillations across the full tail.

    A single global rule with a large cutoff undersamples oscillatory moderate
    frequencies. Fixed-width panels keep resolution independent of cutoff.
    """
    require_gpu()
    if not jax.config.x64_enabled:
        raise RuntimeError('Quadrature preparation requires explicit float64 context.')
    panels=int(round(cutoff/panel_width))
    if abs(panels*panel_width-cutoff)>1e-10:
        raise ValueError('Cutoff must be a multiple of panel width')
    u,w=quadrature(order,panel_width)
    offsets=jnp.arange(panels,dtype=jnp.float64)*panel_width
    return (offsets[:,None]+u[None,:]).reshape(-1),jnp.broadcast_to(w,(panels,order)).reshape(-1)


@lru_cache(maxsize=1)
def fallback_quadrature():
    """GPU-only setup, outside JIT; do not change default learner precision."""
    require_gpu()
    with jax.enable_x64(True):
        u,w=composite_quadrature(16,8.,2000.)
        uf,wf=composite_quadrature(24,8.,2400.)
        # Preserve every float64 bit through an x64-disabled outer JIT. NumPy
        # or floating64 dynamic arguments may otherwise be canonicalized down.
        return tuple(jax.lax.bitcast_convert_type(x,jnp.uint32) for x in (u,w,uf,wf))


def with_fallback(tables):
    """Attach private runtime arrays without changing serialized table identity."""
    u,w,uf,wf=fallback_quadrature()
    return dict(tables,_fallback_u=u,_fallback_w=w,_fallback_uf=uf,_fallback_wf=wf)


def _direct_call64(moneyness, maturity, variance, params, u, weights):
    """Fixed-menu complex128 inversion in an explicit x64 trace context."""
    s,t,v=(jnp.asarray(a,jnp.float64) for a in (moneyness,maturity,variance))
    kap,bar,xi,rho=jnp.asarray(params,jnp.float64).T
    kap,bar,xi,rho=(a[...,None] for a in (kap,bar,xi,rho))
    t,v=t[...,None],v[...,None]
    phase=jnp.exp(1j*jnp.log(s)[...,None]*u)
    def cf(z):
        b=kap-rho*xi*1j*z
        d=jnp.sqrt(b*b+xi*xi*(z*z+1j*z))
        g=(b-d)/(b+d)
        e=jnp.exp(-d*t)
        c=kap*bar/(xi*xi)*((b-d)*t-2*jnp.log((1-g*e)/(1-g)))
        D=(b-d)/(xi*xi)*(1-e)/(1-g*e)
        return jnp.exp(c+D*v)
    # Combine before reduction, avoiding subtraction of separately rounded
    # probabilities multiplied by potentially large spot values.
    integrand=jnp.real(phase*(s[...,None]*cf(u-1j)-cf(u))/(1j*u))
    return .5*(s-1)+jnp.sum(weights*integrand,axis=-1)/jnp.pi


def fallback_prices(regime_ids, spot, variance, maturity, tables, strike=1.05):
    """Small float64 GPU batch, float32 outputs; invalid checks return NaN.

    Error gates/projection are in C/K units. Exact float64 quadrature is packed
    into uint32 pairs so outer learner JIT precision remains unchanged.
    """
    require_gpu()
    with jax.enable_x64(True):
        s=jnp.asarray(spot,jnp.float64)/jnp.asarray(strike,jnp.float64)
        v=jnp.asarray(variance,jnp.float64)
        t=jnp.asarray(maturity,jnp.float64)
        params=jnp.asarray(tables['regimes'],jnp.float64)[regime_ids]
        u,w,uf,wf=(jax.lax.bitcast_convert_type(tables[name],jnp.float64) for name in
            ('_fallback_u','_fallback_w','_fallback_uf','_fallback_wf'))
        coarse=_direct_call64(s,t,v,params,u,w)
        fine=_direct_call64(s,t,v,params,uf,wf)
        lower=jnp.maximum(s-1,0)
        error=jnp.abs(fine-coarse)
        violation=jnp.maximum(jnp.maximum(lower-fine,fine-s),0)
        valid=(jnp.isfinite(fine)&jnp.isfinite(coarse)&(error<=2e-6)&(violation<=2e-6)
               &(s>0)&(v>=0)&(t>0))
        price=jnp.clip(fine,lower,s)*jnp.asarray(strike,jnp.float64)
        return (jnp.where(valid,price,jnp.nan).astype(jnp.float32),
                error.astype(jnp.float32),violation.astype(jnp.float32))



def _replace_outside(regime_ids,spot,variance,prices,bad,tables,strike):
    """Dynamic number of 32-state chunks, bounded Fourier temporaries."""
    size=spot.size
    indexes=jnp.nonzero(bad.reshape(-1),size=size,fill_value=0)[0]
    indexes=jnp.pad(indexes,(0,32))
    count=jnp.sum(bad,dtype=jnp.int32)
    regimes=jnp.broadcast_to(regime_ids[:,None],spot.shape).reshape(-1)
    days=jnp.broadcast_to(jnp.asarray(tables['days'],jnp.float32)[None,:],spot.shape).reshape(-1)
    s,v=spot.reshape(-1),variance.reshape(-1)
    def body(carry):
        offset,out=carry
        idx=jax.lax.dynamic_slice(indexes,(offset,),(32,))
        price,_,_=fallback_prices(regimes[idx],s[idx],v[idx],days[idx]/252.,tables,strike)
        active=offset+jnp.arange(32,dtype=jnp.int32)<count
        # Drop padded writes rather than overwriting a valid first state.
        out=out.at[jnp.where(active,idx,size)].set(price.astype(out.dtype),mode='drop')
        return offset+jnp.int32(32),out
    _,out=jax.lax.while_loop(lambda state:state[0]<count,body,
                            (jnp.int32(0),prices.reshape(-1)))
    return out.reshape(spot.shape)
