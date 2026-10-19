#!/usr/bin/env python
"""GPU stress gate for bounded float64 out-of-grid pricing with float32 outputs; never rewrites market data."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
from src.pricing import (require_gpu,load_tables,table_hash,heston_call,
                         fallback_prices,interpolate_prices,composite_quadrature)
from src.protocol import file_hash,source_hash


def validate(tables):
    require_gpu()
    # Deliberately exceed both setup grid proposals, including high variance
    # and the low-variance Fourier stress corner for every regime and maturity.
    log_s,v,days=jnp.meshgrid(jnp.array([-3.,-1.6,-1.01,-.2,0.,.71,1.,2.],jnp.float32),
        jnp.array([0.,.04,.09,1.21,2.],jnp.float32),
        jnp.array([40.,60.,124.],jnp.float32),indexing='ij')
    s=jnp.exp(log_s.reshape(-1))*jnp.float32(1.05)
    v=v.reshape(-1); t=days.reshape(-1)/jnp.float32(252.)
    checked=jax.jit(fallback_prices,static_argnames=('strike',))
    errors=[]; convergence=[]; bounds=[]; finite=[]; reference_convergence=[]
    per_regime=[]
    failure_states=[]
    with jax.enable_x64(True):
        reference_rule=composite_quadrature(32,8.,2800.)
        finer_reference_rule=composite_quadrature(40,8.,3200.)
        original_rules=(*composite_quadrature(16,8.,2000.),*composite_quadrature(24,8.,2400.))
        packed_names=('_fallback_u','_fallback_w','_fallback_uf','_fallback_wf')
        lossless=all(bool(jnp.array_equal(original,jax.lax.bitcast_convert_type(tables[name],jnp.float64)))
                     for original,name in zip(original_rules,packed_names))
    for r,params in enumerate(tables['regimes']):
        ids=jnp.full(s.shape,r,jnp.int32)
        price,diff,bound=checked(ids,s,v,t,tables,strike=1.05)
        with jax.enable_x64(True):
            direct=heston_call(s,1.05,t,v,*params,quadrature_rule=reference_rule)
            ref=heston_call(s,1.05,t,v,*params,quadrature_rule=finer_reference_rule)
            error=jnp.abs(price-ref)
            refdiff=jnp.abs(direct-ref)
        errors.append(error.astype(jnp.float32));convergence.append(diff)
        bounds.append(bound);finite.append(jnp.isfinite(price));reference_convergence.append(refdiff.astype(jnp.float32))
        failed=(~jnp.isfinite(price))|(error>=5e-6)|(refdiff>=2e-6)
        failed_ids=jax.device_get(jnp.where(failed)[0]).tolist()
        for index in failed_ids:
            failure_states.append({'regime':r,'spot':float(s[index]),'variance':float(v[index]),
                'maturity':float(t[index]),'price_finite':bool(jnp.isfinite(price[index])),
                'absolute_error':float(error[index]) if bool(jnp.isfinite(error[index])) else None,
                'reference_price':float(ref[index]),'reference_convergence':float(refdiff[index]),
                'fallback_convergence':float(diff[index]),'bound_violation':float(bound[index])})
        per_regime.append({'regime':r,'finite':bool(jnp.all(jnp.isfinite(price))),
                           'max_absolute_error':float(jnp.nanmax(error)),
                           'reference_convergence':float(jnp.max(refdiff))})
    # Exercise compiled rare branch, >32 states, padded final chunk, and flat
    # index zero. In-grid output must remain bitwise identical.
    n=48; width=len(tables['days'])
    spots=jnp.ones((n,width),jnp.float32)
    variances=jnp.full_like(spots,.04)
    spots=spots.at[:,0].set(jnp.float32(1.05)*jnp.exp(jnp.asarray(tables['log_moneyness'][0],jnp.float32)-.1))
    variances=variances.at[:,0].set(jnp.asarray(tables['variance_grid'][-1],jnp.float32)+.1)
    ids=jnp.arange(n,dtype=jnp.int32)%12
    plain={k:value for k,value in tables.items() if not k.startswith('_fallback_')}
    @jax.jit
    def paired_interpolation(ids,spots,variances,tables):
        original={k:value for k,value in tables.items() if not k.startswith('_fallback_')}
        return (interpolate_prices(ids,spots,variances,original),
                interpolate_prices(ids,spots,variances,tables))
    (before,bad,dist),(after,after_bad,after_dist)=paired_interpolation(ids,spots,variances,tables)
    unchanged=bool(jnp.all(jnp.where(bad,True,before==after)))
    errors=jnp.concatenate(errors); finite=jnp.concatenate(finite)
    report={'pricing_hash':table_hash(tables),'dtype':'float32_outputs_float64_fallback','device':str(jax.devices()[0]),
        'stress_states':int(errors.size),'finite_count':int(jnp.sum(finite)),
        'max_absolute_error':float(jnp.nanmax(errors)),
        'rmse':float(jnp.sqrt(jnp.nanmean(errors**2))),
        'reference_convergence_max_error':float(jnp.max(jnp.concatenate(reference_convergence))),
        'fallback_quadrature_max_difference':float(jnp.max(jnp.concatenate(convergence))),
        'fallback_raw_bound_max_violation':float(jnp.max(jnp.concatenate(bounds))),
        'quadrature_storage_lossless':lossless,'learner_default_x64':bool(jax.config.x64_enabled),
        'output_is_float32':after.dtype==jnp.float32,'in_grid_bitwise_unchanged':unchanged,'compiled_fallback_all_finite':bool(jnp.all(jnp.isfinite(after))),
        'compiled_fallback_count':int(jnp.sum(after_bad)),
        'out_of_grid_counters_preserved':bool(jnp.all(bad==after_bad)&jnp.all(dist==after_dist)),
        'table_hash_unchanged':table_hash(tables)==table_hash(plain),'per_regime':per_regime,
        'failure_states':failure_states,
        'quadrature':{'storage':'lossless_uint32_pairs_decoded_float64','panel_width':8.,'coarse_order':16,'coarse_cutoff':2000.,
            'fine_order':24,'fine_cutoff':2400.,'reference_orders':[32,40],
            'reference_cutoffs':[2800.,3200.]}}
    report['passed']=bool(jnp.all(finite) and report['max_absolute_error']<5e-6
        and report['reference_convergence_max_error']<2e-6 and unchanged
        and report['compiled_fallback_all_finite'] and report['out_of_grid_counters_preserved']
        and report['table_hash_unchanged'] and lossless and report['output_is_float32']
        and not report['learner_default_x64'])
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--table',default='cache/pricing/heston.npz')
    p.add_argument('--output',default='cache/pricing/fallback.validation.json');a=p.parse_args()
    require_gpu()
    # Force the same default precision as learner processes. The independent
    # reference computation alone enters an explicit float64 context.
    jax.config.update('jax_enable_x64',False)
    tables=load_tables(a.table)
    report=validate(tables)
    root=Path(__file__).resolve().parents[1]
    report.update(pricing_source_hash=file_hash(root/'src/pricing.py'),source_hash=source_hash(root))
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if not report['passed']:raise SystemExit('Fallback stress gate failed; training prohibited.')

if __name__=='__main__':main()
