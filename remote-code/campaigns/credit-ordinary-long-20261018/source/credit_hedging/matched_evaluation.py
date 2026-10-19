"""Distribution comparison at the baseline's already-frozen development-selected LR."""
from pathlib import Path
import json,os,time
import equinox as eqx
import jax
import numpy as np
import optax
from .learned_run import configs,evaluate,simulate,write
from .learned_policy import TargetHedger
from .curve_market import build_curve_tables
from .learned_evaluation import metrics,bootstrap_gains


def main(source,out,checkpoints):
    source=Path(source);out=Path(out);out.mkdir(exist_ok=True);checkpoints=Path(checkpoints)
    jax.config.update('jax_enable_x64',True);assert jax.default_backend()=='gpu'
    design=json.loads((source/'learned_design.json').read_text());cfgs=configs(source)
    freeze=json.loads((checkpoints/'matched_freeze.json').read_text());lr=freeze['common_learning_rate'];models={}
    model=TargetHedger(jax.random.key(design['seed']));opt=optax.chain(optax.clip_by_global_norm(1.),optax.adam(lr));state=opt.init(eqx.filter(model,eqx.is_inexact_array))
    for arm in ('baseline','easy','hard'):
        models[arm],_,_=eqx.tree_deserialise_leaves(checkpoints/(arm+'.eqx'),(model,state,jax.random.key_data(jax.random.key(0))))
    gold=np.load(checkpoints/'original_terminal.npz');results={};losses={};checks=[]
    for domain in ('easy','ordinary','hard'):
        cfg=cfgs[domain];tables=eqx.filter_jit(build_curve_tables)(cfg);jax.block_until_ready(tables)
        acc={arm:dict(loss=[],daily_sum=np.zeros(1260),daily_sq=np.zeros(1260),cost=0.,turnover=0.,notional=np.zeros(1260)) for arm in models};default=[]
        for start in range(0,8192,128):
            key=jax.random.fold_in(jax.random.key(design['confirmation_seed']),('easy','ordinary','hard').index(domain)*100000+start)
            paths=simulate(key,tables,cfg,128);flags=np.asarray(paths['default']).any(1);default.append(flags)
            for arm,model in models.items():
                result=evaluate(model,paths,tables,cfg);ls=np.asarray(result['loss']);records={k:np.asarray(v) for k,v in result['records'].items()}
                error=float(np.max(np.abs(records['daily_loss'].sum(1)-ls)))
                if error>1e-10 or not all(np.isfinite(v).all() for v in records.values()):raise RuntimeError('Matched accounting or finite check failed')
                if np.any(records['notional']<0) or np.any(records['notional']>2.+1e-12):raise RuntimeError('Bounds failed')
                if arm=='baseline':
                    replay=float(np.max(np.abs(ls-gold[domain+'_baseline'][start:start+128])))
                    if replay>1e-10:raise RuntimeError('Baseline/path replay differs')
                else:
                    np.savez_compressed(out/f'{domain}-{arm}-{start:05d}.npz',loss=ls,daily_loss=records['daily_loss'],defaulted=flags,cost=records['discounted_cost'].sum(1),turnover=(records['buy']+records['sold']).sum(1))
                a=acc[arm];a['loss'].append(ls);a['daily_sum']+=records['daily_loss'].sum(0);a['daily_sq']+=(records['daily_loss']**2).sum(0);a['cost']+=records['discounted_cost'].sum();a['turnover']+=(records['buy']+records['sold']).sum();a['notional']+=records['notional'].sum(0)
                checks.append(dict(domain=domain,arm=arm,start=start,max_accounting_error=error))
            if start%2048==0:print(json.dumps(dict(domain=domain,episodes=start+128)),flush=True)
        default=np.concatenate(default);results[domain]={};losses[domain]={}
        for arm,a in acc.items():
            ls=np.concatenate(a['loss']);losses[domain][arm]=ls;results[domain][arm]=metrics(ls,a['daily_sum'],a['daily_sq'],8192,a['cost'],a['turnover'],default)
            np.savez_compressed(out/f'{domain}-{arm}-daily-curve.npz',mean=a['daily_sum']/8192,std=np.sqrt(np.maximum(a['daily_sq']/8192-(a['daily_sum']/8192)**2,0)),mean_notional=a['notional']/8192)
        write(out/'metrics.partial.json',results)
        del tables
        import gc
        gc.collect();jax.clear_caches()
    gains={d:bootstrap_gains(losses[d]['baseline'],losses[d][d],1020000+design['seed']+(d=='hard')) for d in ('easy','hard')}
    ordering={k:results['hard']['hard'][k]-results['easy']['easy'][k] for k in ('terminal_std','centered_entropic','centered_cvar95','daily_error_std')}
    write(out/'metrics.json',results);write(out/'paired_gains.json',gains);write(out/'residual_ordering.json',ordering)
    write(out/'audit.json',dict(passed=True,checks=checks,baseline_replay_tolerance=1e-10,scope='Same original fresh paths; no new training, tuning or checkpoint selection from evaluation outcomes. Common LR determined by baseline frozen development choice.'))
    np.savez_compressed(out/'terminal.npz',**{d+'_'+a:v for d,rows in losses.items() for a,v in rows.items()})
    # Independent cross-check of terminal risk formulas against the previously GPU-tested implementation.
    from .metrics import entropic_risk
    for d,rows in losses.items():
        for a,v in rows.items():
            error=abs(float(entropic_risk(jax.numpy.asarray(v),10.))-results[d][a]['entropic'])
            if error>1e-12:raise RuntimeError('Independent entropic metric check failed')
    write(out/'complete.json',dict(status='matched_learning_rate_evaluation_complete',common_learning_rate=lr,paired_gains=gains,residual_ordering=ordering))
