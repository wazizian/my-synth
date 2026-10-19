#!/usr/bin/env python
"""GPU table precompute and independent interpolation/coverage validation."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from src.market import REGIMES, DEFAULT_CONFIG, simulate_latent, mixture_probabilities
from src.pricing import build_price_tables, save_tables, load_tables, heston_call, interpolate_prices, require_gpu


def validate(tables, episodes=12288):
    require_gpu()
    if episodes < 120: raise ValueError('Validation needs at least 120 episodes.')
    ids=jnp.arange(episodes)%12
    s,v=simulate_latent(ids,jax.random.key(871),DEFAULT_CONFIG)
    c,bad,dist=interpolate_prices(ids,s,v,tables)
    sn,vn,cn=s,v,c
    ki,kt=jax.random.split(jax.random.key(881))
    ii=jax.random.randint(ki,(512,),0,episodes)
    tt=jax.random.randint(kt,(512,),0,85)
    direct=jnp.zeros(512,dtype=jnp.float64)
    reference_error=jnp.zeros(512,dtype=jnp.float64)
    for r in range(12):
        prices=heston_call(sn[ii,tt],1.05,(124-tt)/252,vn[ii,tt],*REGIMES[r],nodes=768,cutoff=1500)
        finer=heston_call(sn[ii,tt],1.05,(124-tt)/252,vn[ii,tt],*REGIMES[r],nodes=1024,cutoff=2000)
        reference_error=jnp.where(ii%12==r,jnp.abs(prices-finer),reference_error)
        direct=jnp.where(ii%12==r,finer,direct)
    error=cn[ii,tt]-direct
    spot_end=sn[:,-1]; call_delta=cn[:,-1]-cn[:,0]
    result={'episodes':episodes,'device':str(jax.devices()[0]),'reference_dtype':'float64',
            'reference_convergence_max_abs_error':float(reference_error.max()),
            'interpolation_absolute_tolerance':0.0005,'weak_discretization_absolute_allowance':0.0005,'out_of_bounds_fraction':float(jnp.asarray(bad).mean()),'maximum_out_of_grid_distance':float(jnp.asarray(dist).max()),
            'interpolation_max_abs_error':float(jnp.nanmax(jnp.abs(error))),'interpolation_rmse':float(jnp.sqrt(jnp.nanmean(error**2))),
            'spot_terminal_mean':float(spot_end.mean()),'spot_martingale_z':float((spot_end.mean()-1)/(spot_end.std()/jnp.sqrt(episodes))),
            'call_mean_change':float(jnp.nanmean(call_delta)),'call_martingale_z':float(jnp.nanmean(call_delta)/(jnp.nanstd(call_delta)/jnp.sqrt(episodes))),
            'nonfinite_fraction':float((~jnp.isfinite(cn)).mean())}
    values=jnp.asarray(tables['values']); spots=jnp.exp(tables['log_moneyness'])[None,None,:,None]
    result['table_intrinsic_bound_violation']=float(jnp.maximum(jnp.maximum(spots-1,0)-values,0).max())
    result['table_upper_bound_violation']=float(jnp.maximum(values-spots,0).max())
    result['table_spot_monotonicity_violation']=float(jnp.maximum(-jnp.diff(values,axis=2),0).max())
    for name in ('raw_intrinsic_bound_violation','raw_upper_bound_violation',
                 'raw_spot_monotonicity_violation','projection_max_abs_correction',
                 'projection_tolerance'):
        result[name]=float(tables.get(name,0.))
    result['projection_applied']=bool(tables.get('projection_applied',False))
    result['benchmarks']={name:jax.device_get(q).tolist() for name,q in zip(('base','easy','hard'),mixture_probabilities())}
    result['per_regime']=[]
    for r in range(12):
        mask=jnp.arange(episodes)%12==r
        delta=call_delta[mask]; terminal=spot_end[mask]; count=int(mask.sum())
        result['per_regime'].append({'regime':r,'episodes':count,
            'spot_martingale_z':float((terminal.mean()-1)/(terminal.std()/jnp.sqrt(count))),
            'call_martingale_z':float(jnp.mean(delta)/(jnp.std(delta)/jnp.sqrt(count))),
            'out_of_bounds_fraction':float(jnp.asarray(bad)[mask].mean())})
    # Weak-consistency comparison: independent keyed draws, standard-error-aware.
    fine=dict(DEFAULT_CONFIG,substeps=16)
    sf,vf=simulate_latent(ids,jax.random.key(872),fine)
    fine_payoff=jnp.maximum(jnp.asarray(sf[:,-1])-1.,0); coarse_payoff=jnp.maximum(spot_end-1.,0)
    se=jnp.sqrt((fine_payoff.var()+coarse_payoff.var())/episodes)
    result['weak_call_payoff_difference']=float(coarse_payoff.mean()-fine_payoff.mean())
    result['weak_call_payoff_difference_se']=float(se)
    result['passed']=bool(result['out_of_bounds_fraction']==0 and result['nonfinite_fraction']==0
        and result['reference_convergence_max_abs_error']<2e-6
        and result['interpolation_max_abs_error']<.0005
        and max(result['table_intrinsic_bound_violation'],result['table_upper_bound_violation'],result['table_spot_monotonicity_violation'],
                result['raw_intrinsic_bound_violation'],result['raw_upper_bound_violation'],result['raw_spot_monotonicity_violation'])<2e-6
        and float(jnp.min(values))>=0
        and abs(result['spot_martingale_z'])<5 and abs(result['call_martingale_z'])<5
        and all(abs(x['spot_martingale_z'])<6 and abs(x['call_martingale_z'])<6 for x in result['per_regime'])
        and abs(result['weak_call_payoff_difference'])<5*se+.0005)
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='cache/pricing/heston.npz'); p.add_argument('--validate-only',action='store_true'); p.add_argument('--episodes',type=int,default=12288); a=p.parse_args()
    require_gpu()
    start=time.monotonic()
    tables=load_tables(a.output) if a.validate_only else build_price_tables(REGIMES,nodes=384,cutoff=600.)
    report=validate(tables,a.episodes)
    report['table_quadrature']={'nodes':384,'cutoff':600.} if not a.validate_only else 'loaded_cache'
    report['reference_quadrature']={'nodes':1024,'cutoff':2000.}
    initial_report=dict(report)
    if not report['passed'] and not a.validate_only:
        # One permitted setup expansion/refinement; never hide invalid states.
        tables=build_price_tables(REGIMES,jnp.linspace(-1.4,.9,192),jnp.linspace(0,jnp.sqrt(1.2),96)**2,nodes=768,cutoff=1500.)
        report=validate(tables,a.episodes); report['expanded_once']=True
        report['table_quadrature']={'nodes':768,'cutoff':1500.}
        report['reference_quadrature']={'nodes':1024,'cutoff':2000.}
        report['initial_attempt']=initial_report
    jax.block_until_ready(tables['values'])
    report['sha256']=save_tables(a.output,tables); report['setup_seconds']=time.monotonic()-start
    path=Path(a.output).with_suffix('.validation.json'); path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']: raise SystemExit('Market validation failed; training is prohibited.')

if __name__=='__main__': main()
