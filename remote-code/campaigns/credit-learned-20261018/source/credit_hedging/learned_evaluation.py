"""Fresh paired cross-evaluation of frozen learned hedgers and tuned static rules."""
from pathlib import Path
import json,time,hashlib,os,gc
import equinox as eqx
import jax
import numpy as np
from .learned_policy import TargetHedger
from .learned_run import configs,evaluate,simulate,write
from .curve_market import build_curve_tables
from .diagnostic_policies import FixedProtection


def metrics(loss,daily_sum,daily_sq,n,cost,turnover,defaulted):
    z=10.*loss;zmax=z.max();mean=float(loss.mean());tail=.05*n
    vals=np.sort(loss)[::-1];w=np.clip(tail-np.arange(n),0,1)
    exp=np.exp(z-zmax);exp/=exp.sum()
    daily_mean=daily_sum/n;daily_var=np.maximum(daily_sq/n-daily_mean**2,0)
    entropic=float((zmax+np.log(np.exp(z-zmax).mean()))/10.)
    cvar=float(w@vals/tail)
    return dict(entropic=entropic,cvar95=cvar,mse=float(np.mean(loss**2)),mean=mean,terminal_std=float(loss.std()),centered_entropic=entropic-mean,centered_cvar95=cvar-mean,daily_error_std=float(np.sqrt(daily_var.mean())),mean_cost=float(cost/n),mean_turnover=float(turnover/n),default_count=int(defaulted.sum()),survival_count=int((~defaulted).sum()),default_path_mean_loss=float(loss[defaulted].mean()),survival_path_mean_loss=float(loss[~defaulted].mean()),entropic_ess=float(1/(exp@exp)),tail_mass=float(tail),n=n)


def bootstrap_gains(base,other,seed):
    rng=np.random.default_rng(seed);draws=[]
    def risk(x):
        z=10*x;m=z.max();return (m+np.log(np.exp(z-m).mean()))/10
    for _ in range(1000):
        ix=rng.integers(len(base),size=len(base));draws.append(risk(base[ix])-risk(other[ix]))
    return dict(gain=float(risk(base)-risk(other)),ci95=np.quantile(draws,[.025,.975]).tolist(),scope='Paired whole-episode bootstrap; conditional on single-seed trained and selected policies')


def main(source,out,checkpoint_root):
    jax.config.update('jax_enable_x64',True);assert jax.default_backend()=='gpu'
    source=Path(source);out=Path(out);out.mkdir(exist_ok=True);checkpoints=Path(checkpoint_root)
    design=json.loads((source/'learned_design.json').read_text());cfgs=configs(source)
    models={}
    for arm in ('baseline','easy','hard'):
        # Decode model leaves from whole training-state checkpoint using matching optimizer skeleton.
        import optax
        meta=json.loads((checkpoints/(arm+'-selection.json')).read_text());lr=meta['selected']['lr']
        model=TargetHedger(jax.random.key(design['seed']));optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(lr));state=optimizer.init(eqx.filter(model,eqx.is_inexact_array))
        models[arm],_,_=eqx.tree_deserialise_leaves(checkpoints/(arm+'.eqx'),(model,state,jax.random.key_data(jax.random.key(0))))
    for arm,model in models.items():
        if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(model) if eqx.is_array(v)):raise RuntimeError('Nonfinite selected model: '+arm)
    selection_audit=[]
    for arm in ('baseline','easy','hard'):
        meta=json.loads((checkpoints/(arm+'-selection.json')).read_text())
        domains=('easy','ordinary','hard') if arm=='baseline' else (arm,)
        probabilities=(.05,.9,.05) if arm=='baseline' else (1.,)
        def reconstruct(path):
            with np.load(path) as data:
                vals=np.concatenate([data[d] for d in domains]);weights=np.concatenate([np.full(len(data[d]),p/len(data[d])) for d,p in zip(domains,probabilities)])
            z=10*vals;maximum=z.max();return float((maximum+np.log(weights@np.exp(z-maximum)))/10.)
        initial=reconstruct(checkpoints/(arm+'-initial-selection.npz'))
        if abs(initial-meta['initial']['entropic'])>1e-12:raise RuntimeError('Initial development reconstruction failed')
        bests=[]
        for trial in meta['trials']:
            lr=trial['lr'];progress=json.loads((checkpoints/(arm+f'-lr-{lr:g}-progress.json')).read_text());scores=[initial]
            for boundary,row in zip((60,180,360,600),progress['history']):
                value=reconstruct(checkpoints/(arm+f'-lr-{lr:g}-selection-{boundary}.npz'));scores.append(value)
                if abs(value-row['entropic'])>1e-12:raise RuntimeError('Checkpoint selection metric reconstruction failed')
            if abs(min(scores)-trial['entropic'])>1e-12:raise RuntimeError('Incorrect best checkpoint')
            bests.append(min(scores))
        if abs(min(bests)-meta['selected']['entropic'])>1e-12:raise RuntimeError('Incorrect learning-rate selection')
        selection_audit.append(dict(arm=arm,passed=True,initial_entropic=initial,trial_best_entropics=bests))
    write(out/'selection_audit.json',dict(passed=True,checks=selection_audit))
    targets=np.concatenate((np.arange(.5,1.50001,.05),[0.,2.]))
    selection_losses={};final_losses={};all_metrics={};audit=[];default_flags={};replay_losses={arm:{} for arm in models}
    for d in ('easy','ordinary','hard'):
        cfg=cfgs[d];tables=eqx.filter_jit(build_curve_tables)(cfg);jax.block_until_ready(tables)
        print(json.dumps(dict(event='evaluation_tables',domain=d)),flush=True)
        # Static target selected on development data only, independently of final outcomes.
        pieces=[]
        for start in range(0,2048,128):
            key=jax.random.fold_in(jax.random.key(design['selection_seed']),('easy','ordinary','hard').index(d)*100000+start)
            paths=simulate(key,tables,cfg,128)
            a=evaluate(FixedProtection(0.),paths,tables,cfg);b=evaluate(FixedProtection(1.),paths,tables,cfg)
            l0=np.asarray(a['loss']);l1=np.asarray(b['loss']);pieces.append(l0[None]+targets[:,None]*(l1-l0)[None])
            for arm,model in models.items():
                if arm=='baseline' or arm==d:
                    replay_losses[arm].setdefault(d,[]).append(np.asarray(evaluate(model,paths,tables,cfg)['loss']))
            if start==0:
                for target in (.5,1.5,2.):
                    check=evaluate(FixedProtection(target),paths,tables,cfg)
                    error=float(np.max(np.abs(np.asarray(check['loss'])-(l0+target*(l1-l0)))))
                    if error>1e-10:raise RuntimeError('Static interpolation identity failed')
                    audit.append(dict(check='Static affine cashflow identity',domain=d,target=target,max_error=error))
        selection_losses[d]=np.concatenate(pieces,axis=1)
        del tables;gc.collect();jax.clear_caches()
    # Freeze static choices from the same development distributions used for learned selection.
    statics={}
    def risk(x,w=None):
        z=10*x;m=z.max();return float((m+np.log(np.exp(z-m).mean() if w is None else w@np.exp(z-m)))/10.)
    replay_audit=[]
    for arm,domain_rows in replay_losses.items():
        domains=('easy','ordinary','hard') if arm=='baseline' else (arm,)
        ps=(.05,.9,.05) if arm=='baseline' else (1.,)
        values=np.concatenate([np.concatenate(domain_rows[d]) for d in domains]);w=np.concatenate([np.full(2048,p/2048) for p in ps])
        actual=risk(values,w);expected=json.loads((checkpoints/(arm+'-selection.json')).read_text())['selected']['entropic'];error=abs(actual-expected)
        if error>1e-10:raise RuntimeError('Full-calendar selected-checkpoint replay failed')
        replay_audit.append(dict(arm=arm,passed=True,max_error=error,actual=actual,expected=expected))
    write(out/'checkpoint_replay_audit.json',dict(passed=True,checks=replay_audit))
    for arm in ('baseline','easy','hard'):
        if arm=='baseline':
            dev=np.concatenate([selection_losses[d] for d in ('easy','ordinary','hard')],axis=1);weights=np.concatenate([np.full(2048,p/2048) for p in (.05,.9,.05)])
            scores=[risk(x,weights) for x in dev]
        else:scores=[risk(x) for x in selection_losses[arm]]
        ix=int(np.argmin(scores));statics[arm]=dict(target=float(targets[ix]),development_entropic=float(scores[ix]))
    write(out/'static_selection.json',dict(statics))
    write(out/'evaluation_freeze.json',dict(frozen_at=time.time(),static_targets=statics,neural_checkpoints={arm:hashlib.sha256((checkpoints/(arm+'.eqx')).read_bytes()).hexdigest() for arm in models},fresh_evaluation_opened=False))
    for d in ('easy','ordinary','hard'):
        cfg=cfgs[d];tables=eqx.filter_jit(build_curve_tables)(cfg);jax.block_until_ready(tables)
        print(json.dumps(dict(event='fresh_evaluation_tables',domain=d)),flush=True)
        names=list(models)+['unhedged','replenish_1.0'];aggregate={n:dict(loss=[],daily_sum=np.zeros(1260),daily_sq=np.zeros(1260),cost=0.,turnover=0.) for n in names};defaults=[]
        # Save fixed reference daily arrays; any target is reproducible by affine interpolation.
        for start in range(0,8192,128):
            key=jax.random.fold_in(jax.random.key(design['confirmation_seed']),('easy','ordinary','hard').index(d)*100000+start)
            paths=simulate(key,tables,cfg,128);host={k:np.asarray(v) for k,v in paths.items()};defaulted=host['default'].any(1);defaults.append(defaulted)
            np.savez_compressed(out/f'{d}-paths-{start:05d}.npz',**host)
            for name in names:
                model=models[name] if name in models else FixedProtection(0. if name=='unhedged' else 1.)
                res=evaluate(model,paths,tables,cfg);loss=np.asarray(res['loss']);rec={k:np.asarray(v) for k,v in res['records'].items()}
                error=float(np.max(np.abs(rec['daily_loss'].sum(1)-loss)))
                cash_error=float(np.max(np.abs(np.asarray(res['initial_price'])-rec['discounted_cashflow'].sum(1)-loss)))
                finite=all(np.isfinite(x).all() for x in rec.values()) and np.isfinite(loss).all()
                if max(error,cash_error)>1e-10 or not finite:raise RuntimeError('Independent accounting/finite check failed')
                if np.any(rec['daily_loss'][~host['alive'][:,:-1]]!=0):raise RuntimeError('Post-default increments not absorbing zero')
                if np.any(rec['notional']< -1e-12) or np.any(rec['notional']>2.+1e-12):raise RuntimeError('Protection bound failed')
                a=aggregate[name];a['loss'].append(loss);a['daily_sum']+=rec['daily_loss'].sum(0);a['daily_sq']+=(rec['daily_loss']**2).sum(0);a['cost']+=rec['discounted_cost'].sum();a['turnover']+=(rec['buy']+rec['sold']).sum()
                # Preserve complete per-episode daily economic/error records, avoiding large vintage cubes.
                np.savez_compressed(out/f'{d}-{name}-{start:05d}.npz',loss=loss,initial_price=np.asarray(res['initial_price']),defaulted=defaulted,**{k:rec[k] for k in ('daily_loss','discounted_cashflow','discounted_cost','buy','sold','notional')})
                audit.append(dict(check='Raw accounting, finite values and protection bounds',domain=d,policy=name,start=start,max_accounting_error=error,max_cash_error=cash_error))
            if start%1024==0:print(json.dumps(dict(event='evaluation',domain=d,episodes=start+128)),flush=True)
        defaults=np.concatenate(defaults);default_flags[d]=defaults;all_metrics[d]={};final_losses[d]={}
        for name,a in aggregate.items():
            loss=np.concatenate(a['loss']);final_losses[d][name]=loss
            all_metrics[d][name]=metrics(loss,a['daily_sum'],a['daily_sq'],8192,a['cost'],a['turnover'],defaults)
            all_metrics[d][name]['initial_bond_price']=float(np.asarray(tables['bond'][1260,tables['initial_state']]))
            np.savez_compressed(out/f'{d}-{name}-daily-curve.npz',mean=a['daily_sum']/8192,std=np.sqrt(np.maximum(a['daily_sq']/8192-(a['daily_sum']/8192)**2,0)))
        write(out/'metrics.partial.json',all_metrics)
        del tables;gc.collect();jax.clear_caches()
    # Exact empirical P0 summaries, including between-component daily means.
    all_metrics['base']={}
    mix_weights=np.concatenate([np.full(8192,p/8192) for p in (.05,.9,.05)])
    for name in list(models)+['unhedged','replenish_1.0']:
        vals=np.concatenate([final_losses[d][name] for d in ('easy','ordinary','hard')]);mean=float(mix_weights@vals)
        order=np.argsort(vals)[::-1];before=np.cumsum(mix_weights[order])-mix_weights[order];tail=np.minimum(mix_weights[order],np.maximum(.05-before,0))
        curves=[np.load(out/f'{d}-{name}-daily-curve.npz') for d in ('easy','ordinary','hard')]
        daily_mean=sum(p*a['mean'] for p,a in zip((.05,.9,.05),curves));daily_second=sum(p*(a['std']**2+a['mean']**2) for p,a in zip((.05,.9,.05),curves))
        ent=risk(vals,mix_weights);cv=float(tail@vals[order]/.05)
        all_metrics['base'][name]=dict(entropic=ent,cvar95=cv,mse=float(mix_weights@(vals**2)),mean=mean,terminal_std=float(np.sqrt(mix_weights@((vals-mean)**2))),centered_entropic=ent-mean,centered_cvar95=cv-mean,daily_error_std=float(np.sqrt(np.maximum(daily_second-daily_mean**2,0).mean())),mean_cost=sum(p*all_metrics[d][name]['mean_cost'] for p,d in zip((.05,.9,.05),('easy','ordinary','hard'))),mean_turnover=sum(p*all_metrics[d][name]['mean_turnover'] for p,d in zip((.05,.9,.05),('easy','ordinary','hard'))),n=24576,estimator='Exact empirical P0 mixture')
        flags=np.concatenate([default_flags[d] for d in ('easy','ordinary','hard')]);frequency=float(mix_weights@flags)
        exps=mix_weights*np.exp(10*(vals-vals.max()));exps/=exps.sum()
        all_metrics['base'][name].update(initial_bond_price=sum(p*all_metrics[d][name]['initial_bond_price'] for p,d in zip((.05,.9,.05),('easy','ordinary','hard'))),default_frequency=frequency,default_path_mean_loss=float(mix_weights@(vals*flags)/frequency),survival_path_mean_loss=float(mix_weights@(vals*(~flags))/(1-frequency)),weighted_tail_mass=.05,entropic_ess=float(1/(exps@exps)),max_exponential_weight=float(exps.max()))
        np.savez_compressed(out/f'base-{name}-daily-curve.npz',mean=daily_mean,std=np.sqrt(np.maximum(daily_second-daily_mean**2,0)))
    static_metrics={}
    for domain in ('easy','ordinary','hard'):
        static_metrics[domain]={}
        for arm,choice in statics.items():
            vals=final_losses[domain]['unhedged']+choice['target']*(final_losses[domain]['replenish_1.0']-final_losses[domain]['unhedged'])
            n=len(vals);mass=.05*n;cv=float(np.clip(mass-np.arange(n),0,1)@np.sort(vals)[::-1]/mass)
            static_metrics[domain][arm]=dict(target=choice['target'],entropic=risk(vals),cvar95=cv,mse=float(np.mean(vals**2)),mean=float(vals.mean()),terminal_std=float(vals.std()))
    target=statics['baseline']['target']
    baseline_parts=[final_losses[d]['baseline'] for d in ('easy','ordinary','hard')]
    static_parts=[final_losses[d]['unhedged']+target*(final_losses[d]['replenish_1.0']-final_losses[d]['unhedged']) for d in ('easy','ordinary','hard')]
    baseline_values=np.concatenate(baseline_parts);static_values=np.concatenate(static_parts)
    draws=[];rng=np.random.default_rng(920000+design['seed'])
    for _ in range(1000):
        indices=[rng.integers(8192,size=8192) for _ in range(3)]
        learned=np.concatenate([v[ix] for v,ix in zip(baseline_parts,indices)])
        static=np.concatenate([v[ix] for v,ix in zip(static_parts,indices)])
        draws.append(risk(static,mix_weights)-risk(learned,mix_weights))
    baseline_result=dict(domain='P0 exact empirical mixture',learned_entropic=risk(baseline_values,mix_weights),static_entropic=risk(static_values,mix_weights),static_target=target,gain=risk(static_values,mix_weights)-risk(baseline_values,mix_weights),ci95=np.quantile(draws,[.025,.975]).tolist(),scope='Stratified paired episode bootstrap with frozen component probabilities; conditional on selected single-seed policies')
    write(out/'in_distribution_baseline.json',baseline_result)
    write(out/'static_metrics.json',static_metrics)
    gains={}
    for d in ('easy','hard'):
        gains[d]=dict(specialist_vs_baseline=bootstrap_gains(final_losses[d]['baseline'],final_losses[d][d],720000+design['seed']+(d=='hard')))
        for arm in ('baseline',d):
            target=statics[arm]['target'];static=final_losses[d]['unhedged']+target*(final_losses[d]['replenish_1.0']-final_losses[d]['unhedged'])
            gains[d][arm+'_vs_tuned_static']=bootstrap_gains(static,final_losses[d][arm],820000+design['seed']+(d=='hard'))
    np.savez_compressed(out/'terminal_and_static_selection.npz',**{d+'_'+n:v for d,rs in final_losses.items() for n,v in rs.items()},**{d+'_static_development':v for d,v in selection_losses.items()},static_targets=targets)
    write(out/'metrics.json',all_metrics);write(out/'static_selection.json',statics);write(out/'paired_gains.json',gains);write(out/'independent_audit.json',dict(passed=True,checks=audit,scope='Every raw selected-policy/default episode, full calendar, cash reconstruction, finite values and bounds; static affine identities checked at representative targets.'))
    write(out/'complete.json',dict(status='paired_fresh_cross_evaluation_complete',episodes_per_component=8192,selected_hedgers=list(models),gains=gains))
