#!/usr/bin/env python
"""GPU-only paired mark sensitivity on fixed monitor subsets, no training.

Reprices the existing latent paths, including context, using converged direct
float64 Heston quadrature; reruns the frozen policy and unchanged accounting.
It does not alter datasets, checkpoints, selection, or the scientific arm menu.
"""
import argparse
import datetime
import json
import os
import sys
import time
import uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np  # serialization only
from src.runtime import configure,require_gpu
from src.runner import _config,_inputs,_skeleton
from src.datasets import load_dataset
from src.checkpoint import load_checkpoint,atomic_json
from src.objective import rollout
from src.pricing import load_tables,table_hash,composite_quadrature
from src.protocol import DOMAINS,file_hash,source_hash,canonical_hash


@jax.jit
def direct64(spot,variance,maturity,params,u,weights,strike):
    """Separate P1/P2 float64 inversion, independent of interpolation/fallback."""
    spot=jnp.asarray(spot,jnp.float64)
    variance=jnp.asarray(variance,jnp.float64)[:,None]
    maturity=jnp.asarray(maturity,jnp.float64)[:,None]
    kap,bar,xi,rho=jnp.asarray(params,jnp.float64).T
    kap,bar,xi,rho=(a[:,None] for a in (kap,bar,xi,rho))
    phase=jnp.exp(1j*jnp.log(spot/strike)[:,None]*u)
    def phi(z):
        b=kap-rho*xi*1j*z
        d=jnp.sqrt(b*b+xi*xi*(z*z+1j*z))
        g=(b-d)/(b+d)
        e=jnp.exp(-d*maturity)
        c=kap*bar/(xi*xi)*((b-d)*maturity-2*jnp.log((1-g*e)/(1-g)))
        D=(b-d)/(xi*xi)*(1-e)/(1-g*e)
        return jnp.exp(c+D*variance)
    p1=.5+jnp.sum(weights*jnp.real(phase*phi(u-1j)/(1j*u)),axis=-1)/jnp.pi
    p2=.5+jnp.sum(weights*jnp.real(phase*phi(u)/(1j*u)),axis=-1)/jnp.pi
    return spot*p1-strike*p2


def reprice(batch,tables,strike,chunk_size=32):
    require_gpu()
    shape=batch['spot'].shape
    s=batch['spot'].reshape(-1);v=batch['variance'].reshape(-1)
    ids=jnp.broadcast_to(batch['regime_ids'][:,None],shape).reshape(-1)
    if shape[1]!=len(tables['days']):raise ValueError('Fixed-contract maturity grid mismatch')
    outputs=[];differences=[];violations=[];corrections=[]
    with jax.enable_x64(True):
        rules=(composite_quadrature(32,8.,2800.),composite_quadrature(40,8.,3200.))
        times=jnp.broadcast_to(jnp.asarray(tables['days'],jnp.float64)[None,:],shape).reshape(-1)/252.
        params=jnp.asarray(tables['regimes'],jnp.float64)[ids]
        for start in range(0,s.size,chunk_size):
            stop=min(start+chunk_size,s.size);count=stop-start
            # Same compiled shape even for the last chunk.
            def padded(a):
                values=a[start:stop]
                return jnp.pad(values,[(0,chunk_size-count)]+[(0,0)]*(values.ndim-1),mode='edge')
            args=(padded(s),padded(v),padded(times),padded(params))
            coarse=direct64(*args,*rules[0],jnp.float64(strike))[:count]
            fine=direct64(*args,*rules[1],jnp.float64(strike))[:count]
            lower=jnp.maximum(s[start:stop].astype(jnp.float64)-strike,0)
            bound=jnp.maximum(jnp.maximum(lower-fine,fine-s[start:stop]),0)
            error=jnp.abs(fine-coarse)
            if not bool(jnp.all(jnp.isfinite(fine))&jnp.all(jnp.isfinite(coarse))):
                raise FloatingPointError('Nonfinite independent direct prices')
            if float(jnp.max(error))>=2e-6 or float(jnp.max(bound))>=2e-6:
                raise FloatingPointError('Independent direct-pricing convergence/bounds gate failed')
            projected=jnp.minimum(jnp.maximum(fine,lower),s[start:stop])
            outputs.append(projected.astype(jnp.float32))
            differences.append(error);violations.append(bound);corrections.append(jnp.abs(projected-fine))
        diagnostics={'reference_max_quadrature_difference':float(jnp.max(jnp.concatenate(differences))),
            'reference_max_raw_bound_violation':float(jnp.max(jnp.concatenate(violations))),
            'reference_max_projection':float(jnp.max(jnp.concatenate(corrections)))}
    calls=jnp.concatenate(outputs).reshape(shape)
    error=calls-batch['call']
    diagnostics.update(max_mark_abs_change=float(jnp.max(jnp.abs(error))),
        mark_change_rmse=float(jnp.sqrt(jnp.mean(error**2))),states=int(calls.size),chunk_size=chunk_size)
    return calls,diagnostics


def save_arrays(path,arrays):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**{k:np.asarray(jax.device_get(v)) for k,v in arrays.items()})
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def paired_se(values):
    return float(jnp.std(values,ddof=1)/jnp.sqrt(values.size)) if values.size>1 else None


def full_monitor_mse(run_dir,checkpoint_id,domain):
    path=run_dir/'evaluation_metrics.jsonl'
    if not path.exists():return None
    found=None
    for line in path.read_text().splitlines():
        row=json.loads(line)
        if row.get('checkpoint_id')==checkpoint_id and row.get('domain')==domain and row.get('split_role')=='monitor':
            found=row.get('raw_mse')
    return found


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',nargs='+',default=['baseline-lr001','hard_reference-lr001'])
    p.add_argument('--episodes',type=int,default=32)
    p.add_argument('--checkpoint',default='best',choices=['best','latest','final'])
    p.add_argument('--frozen-selection',action='store_true')
    p.add_argument('--output',default='results/pricing_sensitivity')
    a=p.parse_args()
    if not 2<=a.episodes<=128:raise ValueError('Choose a bounded monitor subset of2..128 episodes')
    root=Path(__file__).resolve().parents[1]
    configure(root);jax.config.update('jax_enable_x64',False)
    started=time.monotonic()
    invocation=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    destination=root/a.output/invocation
    destination.mkdir(parents=True,exist_ok=False)
    report={'invocation_id':invocation,'status':'running','role':'monitor','subset':'first N saved episodes, same across checkpoints',
        'episodes_per_domain':a.episodes,'runs':{},'domains':{},'gaps':{},
        'limitations':['Small fixed subset: does not establish a bound over full datasets or rare tails.',
          'Policies are frozen; this measures evaluation sensitivity, not retraining under exact prices.',
          'Both pricings use the same Heston law and discretized latent paths; model/discretization error is not tested.',
          'Episode standard errors are conditional diagnostics, not independent training-seed uncertainty.',
          'Diagnostic only: no configuration, checkpoint selection, or dataset is changed.'],
        'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        manifest,scales,hashes=_inputs(root)
        tables=load_tables(root/'cache/pricing/heston.npz')
        if table_hash(tables)!=hashes['pricing']:raise ValueError('Table hash mismatch')
        report.update(pricing_hash=hashes['pricing'],benchmark_hash=hashes['benchmark'],
            objective_scales_hash=hashes['scales'],pricing_source_hash=file_hash(root/'src/pricing.py'),
            diagnostic_source_hash=file_hash(Path(__file__)),source_hash=source_hash(root),
            device=str(jax.devices()[0]),reference_dtype='float64',policy_dtype='float32',
            reference_quadrature={'panel_width':8.,'orders':[32,40],'cutoffs':[2800.,3200.]})
        selected={}
        if a.frozen_selection:
            frozen=json.loads((root/'results/frozen_selection.json').read_text())
            if frozen['benchmark_hash']!=manifest['benchmark_hash'] or frozen['objective_scales_hash']!=hashes['scales']:raise ValueError('Selection benchmark/scales mismatch')
            selected={row['run_id']:row['checkpoint_id'] for row in frozen['selected'].values()}
        models={}
        for run_id in a.runs:
            if Path(run_id).name!=run_id:raise ValueError('run_id must be one path component')
            run_dir=root/'runs'/run_id
            config=_config(json.loads((run_dir/'config.json').read_text()))
            skeleton,_,_=_skeleton(config)
            pointer=run_dir/'checkpoints'/(selected[run_id] if a.frozen_selection else a.checkpoint+'.json')
            state,metadata=load_checkpoint(pointer,skeleton,expected_hashes=hashes)
            if metadata['config_hash']!=canonical_hash(config):raise ValueError('Checkpoint config mismatch')
            models[run_id]=(state['model'],config)
            report['runs'][run_id]={'method':config['method'],'checkpoint_id':metadata['checkpoint_id'],
                'checkpoint_state_hash':metadata['state_sha256'],'checkpoint_pointer':str(pointer.relative_to(root)),'checkpoint_source_hash':metadata.get('source_hash'),
                'config_hash':metadata['config_hash'],'domains':{}}
        for domain in DOMAINS:
            data,record=load_dataset(root,'monitor',domain)
            if a.episodes>record['count']:raise ValueError('Requested subset exceeds saved monitor')
            batch={k:jnp.asarray(data[k][:a.episodes]) for k in ('spot','call','variance','regime_ids','episode_id')}
            strike=float(manifest['market'].get('call_strike',1.05))
            direct,diagnostics=reprice(batch,tables,strike)
            report['domains'][domain]=dict(diagnostics,dataset_hash=record['sha256'],
                episode_ids=jax.device_get(batch['episode_id']).tolist())
            original={'spot':batch['spot'],'call':batch['call']}
            replaced={'spot':batch['spot'],'call':direct}
            save_arrays(destination/f'{domain}_marks.npz',dict(episode_id=batch['episode_id'],
                original_call=batch['call'],direct_call=direct))
            for run_id,(model,config) in models.items():
                evaluator=eqx.filter_jit(lambda m,b:rollout(m,b,config['task']))
                z0,d0=evaluator(model,original);z1,d1=evaluator(model,replaced)
                loss0=z0**2;loss1=z1**2;delta=loss1-loss0
                if not bool(jnp.all(jnp.isfinite(delta))):raise FloatingPointError('Nonfinite paired policy outcomes')
                metrics={'original_mse':float(jnp.mean(loss0)),'direct_mse':float(jnp.mean(loss1)),
                    'paired_mse_change':float(jnp.mean(delta)),'paired_mse_change_se':paired_se(delta),
                    'max_episode_loss_change':float(jnp.max(jnp.abs(delta))),
                    'shortfall_change_rmse':float(jnp.sqrt(jnp.mean((z1-z0)**2))),
                    'max_position_change':float(jnp.max(jnp.abs(d1['holdings']-d0['holdings']))),
                    'mean_cost_change':float(jnp.mean(jnp.sum(d1['costs']-d0['costs'],axis=-1))),
                    'mean_turnover_change':float(jnp.mean(jnp.sum(d1['turnover']-d0['turnover'],axis=-1))),
                    'full_monitor_original_mse':full_monitor_mse(root/'runs'/run_id,
                        report['runs'][run_id]['checkpoint_id'],domain)}
                metrics['absolute_mse_change_fraction']=abs(metrics['paired_mse_change'])/metrics['original_mse'] if metrics['original_mse']>1e-12 else None
                report['runs'][run_id]['domains'][domain]=metrics
                archive=destination/f'{run_id}_{domain}_paired.npz'
                save_arrays(archive,dict(episode_id=batch['episode_id'],shortfall_original=z0,shortfall_direct=z1,
                    loss_original=loss0,loss_direct=loss1,loss_change=delta,costs_original=d0['costs'],
                    costs_direct=d1['costs'],turnover_original=d0['turnover'],turnover_direct=d1['turnover'],
                    holdings_original=d0['holdings'],holdings_direct=d1['holdings']))
                metrics.update(episode_archive=str(archive.relative_to(root)),episode_archive_hash=file_hash(archive))
            atomic_json(destination/'report.json',report)
        baseline=next((run_id for run_id,row in report['runs'].items() if row['method']=='baseline'),None)
        if baseline:
            for run_id,row in report['runs'].items():
                if run_id==baseline:continue
                report['gaps'][run_id]={}
                for domain in DOMAINS:
                    b=report['runs'][baseline]['domains'][domain];m=row['domains'][domain]
                    g0=b['original_mse']-m['original_mse'];g1=b['direct_mse']-m['direct_mse']
                    report['gaps'][run_id][domain]={'baseline_run_id':baseline,
                        'original_baseline_minus_method':g0,'direct_baseline_minus_method':g1,
                        'gap_change':g1-g0,'absolute_gap_change_fraction':abs((g1-g0)/g0) if abs(g0)>1e-12 else None,
                        'sign_changed':bool(g0*g1<0)}
        report['status']='complete'
    except Exception as error:
        report.update(status='failed',error=repr(error))
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        atomic_json(destination/'report.json',report)
    print(json.dumps({'status':report['status'],'report':str(destination/'report.json'),'gaps':report['gaps']},indent=2))

if __name__=='__main__':main()
