"""Continuous context/trading Heston simulation with explicit PRNG noise.

Full truncation Euler for the raw variance and log Euler spot, four substeps
per day. Raw variance can be negative; coefficients and reported variance use
its positive part. Reference: Lord, Koekkoek, van Dijk (2010),
https://repub.eur.nl/pub/18571. This is an approximation, not exact simulation.
"""
import jax
import jax.numpy as jnp
from .pricing import interpolate_prices, require_gpu

REGIMES = tuple((2.,bar,xi,rho) for bar in (.04,.09) for xi in (.2,.4,.6) for rho in (-.3,-.8))
DEFAULT_CONFIG = {'context_steps':64,'trading_steps':20,'substeps':4,'dt':1/252,'call_strike':1.05}


def mixture_probabilities():
    """Frozen base/easy/hard benchmark mixtures in REGIMES order."""
    require_gpu()
    q0=jnp.full((12,),.3/11).at[3].set(.7)
    easy=jnp.full((12,),.2/11).at[0].set(.8)
    hard=jnp.full((12,),.2/11).at[11].set(.8)
    return q0,easy,hard


def simulate_latent(regime_ids,key,market_config=None):
    require_gpu()
    config=DEFAULT_CONFIG if market_config is None else market_config
    n=config.get('context_steps',64)+config.get('trading_steps',20)
    sub=config.get('substeps',4); dt=config.get('dt',1/252)/sub
    params=jnp.asarray(REGIMES,dtype=jnp.float32)[regime_ids]; kap,bar,xi,rho=params.T
    noise=jax.random.normal(key,(n,sub,regime_ids.shape[0],2),dtype=jnp.float32)
    def day(carry,zs):
        def step(state,z):
            log_s,raw_v=state; v=jnp.maximum(raw_v,0)
            w1=z[:,0]; w2=rho*w1+jnp.sqrt(1-rho*rho)*z[:,1]
            return (log_s-.5*v*dt+jnp.sqrt(v*dt)*w1,raw_v+kap*(bar-v)*dt+xi*jnp.sqrt(v*dt)*w2),None
        state,_=jax.lax.scan(step,carry,zs)
        return state,(jnp.exp(state[0]),jnp.maximum(state[1],0))
    _,(s,v)=jax.lax.scan(day,(jnp.zeros_like(bar),bar),noise)
    return jnp.concatenate((jnp.ones_like(bar)[:,None],s.T),axis=1),jnp.concatenate((bar[:,None],v.T),axis=1)


def simulate(regime_ids,key,market_config,price_tables):
    s,v=simulate_latent(regime_ids,key,market_config)
    c,bad,distance=interpolate_prices(regime_ids,s,v,price_tables,market_config.get('call_strike',1.05))
    return {'spot':s,'call':c,'regime_ids':regime_ids,'variance':v,
            'out_of_bounds':bad,'max_grid_distance':jnp.max(distance,axis=1)}
