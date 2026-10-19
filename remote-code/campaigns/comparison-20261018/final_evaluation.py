"""GPU-only sealed paired evaluation for the frozen eight-arm comparison.

Manifest requires all eight frozen checkpoint hashes before any final path is drawn.
Large daily records are reduced online; terminal whole-episode records are retained.
"""
from pathlib import Path
from dataclasses import replace
import gc
import hashlib
import json
import pickle
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

ARMS=('baseline','easy_reference','hard_reference','dpg_hard','soar_hard','selfplay','dpg_mixed','soar_mixed')

def write(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def risk(x,gamma=10.):
    z=gamma*np.asarray(x);m=z.max();return float((m+np.log(np.exp(z-m).mean()))/gamma)

def terminal_metrics(x,gamma=10.):
    x=np.asarray(x);n=len(x)
    if not np.isfinite(x).all():raise RuntimeError('Nonfinite terminal sample')
    tail=.05*n;weights=np.clip(tail-np.arange(n),0,1);z=gamma*x;w=np.exp(z-z.max());w/=w.sum();mean=float(x.mean());ent=risk(x,gamma);cvar=float(weights@np.sort(x)[::-1]/tail)
    return dict(entropic=ent,cvar95=cvar,centered_cvar95=cvar-mean,finite_count=n,mean=mean,terminal_std=float(x.std(ddof=1)),centered_entropic=ent-mean,tail_mass=tail,n=n,entropic_ess=float(1/(w@w)),max_exponential_weight=float(w.max()))

def bootstrap(base,other,seed=101,count=1000):
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(count):
        ix=rng.integers(len(base),size=len(base));draws.append(risk(base[ix])-risk(other[ix]))
    return dict(gain=risk(base)-risk(other),ci95=np.quantile(draws,[.025,.975]).tolist(),resampling='paired whole episodes; conditional on one trained initialization',replicates=count)

def validate_manifest(manifest):
    if not manifest.get('frozen') or set(manifest['models'])!=set(ARMS):raise RuntimeError('Require frozen selections for every requested arm')
    if manifest.get('seed',520261017)!=520261017 or manifest.get('episodes',8192)!=8192:raise RuntimeError('Final protocol requires sealed520261017,8192 episodes')
    if manifest.get('bootstrap',1000)!=1000:raise RuntimeError('Require1000 paired whole-episode bootstrap draws')
    for arm,spec in manifest['models'].items():
        path=Path(spec['checkpoint']);actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=spec['sha256']:raise RuntimeError('Frozen checkpoint changed: '+arm)
        meta=json.loads(Path(spec['selection']).read_text())
        if meta.get('final_test_opened',False):raise RuntimeError('Selection improperly used final test: '+arm)
    return hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()

def load_models(manifest,make_model):
    models={}
    for arm,spec in manifest['models'].items():
        path=Path(spec['checkpoint']);kind=spec.get('kind','pickle')
        if kind=='pickle':
            with path.open('rb') as stream:payload=pickle.load(stream)
            models[arm]=jax.tree.map(lambda x:jnp.asarray(x) if isinstance(x,np.ndarray) else x,payload['student'][0])
        elif kind=='option-eqx':models[arm]=eqx.tree_deserialise_leaves(path,make_model(20261017))
        elif kind=='credit-eqx':
            model=make_model(20261017);optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(spec['lr']));state=optimizer.init(eqx.filter(model,eqx.is_inexact_array))
            models[arm],_,_=eqx.tree_deserialise_leaves(path,(model,state,jax.random.key_data(jax.random.key(0))))
        else:raise ValueError(kind)
        if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(models[arm]) if eqx.is_array(x)):raise RuntimeError('Nonfinite model '+arm)
    return models

class Daily:
    def __init__(self,days):self.n=0;self.sum=np.zeros(days);self.square=np.zeros(days);self.cost=0.;self.turnover=0.;self.max_accounting=0.;self.max_absorbing=0.
    def add(self,error,cost,turnover,terminal=None,alive=None):
        error=np.asarray(error);self.n+=len(error);self.sum+=error.sum(0);self.square+=(error*error).sum(0);self.cost+=float(np.asarray(cost).sum());self.turnover+=float(np.asarray(turnover).sum())
        if terminal is not None:self.max_accounting=max(self.max_accounting,float(np.max(np.abs(error.sum(1)-terminal))))
        if alive is not None:self.max_absorbing=max(self.max_absorbing,float(np.max(np.abs(np.where(alive,error,0)))))
    def finish(self):
        mean=self.sum/self.n;var=np.maximum((self.square-self.n*mean*mean)/max(1,self.n-1),0);sd=np.sqrt(var)
        return dict(mean=mean.tolist(),sd=sd.tolist(),rms_cross_episode_sd=float(np.sqrt(np.mean(var))),episodes=self.n,mean_cost=self.cost/self.n,mean_discounted_turnover=self.turnover/self.n,max_accounting_error=self.max_accounting,max_absorbing_error=self.max_absorbing)

def archive_block(path,arrays,metadata,terminal=None):
    """Atomic lossless raw archive with a worker-side readback/ledger proof."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp')
    arrays={name:np.asarray(value) for name,value in arrays.items()}
    with temporary.open('wb') as stream:np.savez_compressed(stream,**arrays)
    temporary.replace(path);maximum=0.
    with np.load(path,allow_pickle=False) as restored:
        for name,original in arrays.items():
            actual=restored[name]
            if actual.shape!=original.shape or actual.dtype!=original.dtype or not np.array_equal(actual,original):raise RuntimeError('Raw archive roundtrip failed: '+name)
        if terminal:
            for arm,loss in terminal.items():maximum=max(maximum,float(np.max(np.abs(restored['daily_loss_'+arm].sum(1)-loss))))
    if maximum>1e-10:raise RuntimeError('Archived daily ledger differs from terminal loss')
    metadata=dict(metadata,path=str(Path('raw-daily')/metadata['domain']/path.name),worker_path=str(path),compressed_bytes=path.stat().st_size,sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),arrays={name:dict(shape=list(value.shape),dtype=str(value.dtype)) for name,value in arrays.items()},max_archived_accounting_error=maximum,lossless_roundtrip_verified=True)
    return metadata

def credit_evaluate(manifest,out,models,seed=520261017,count=8192,audit_callback=None):
    from credit_hedging.curve_market import build_curve_tables,simulate_curve_paths
    from credit_hedging.environment import rollout
    from credit_hedging.learned_run import configs
    cfgs=configs(Path(manifest['source']));result={};raw_records=[];bp=manifest['bp']
    evaluate=eqx.filter_jit(rollout);simulate=eqx.filter_jit(simulate_curve_paths)
    for di,domain in enumerate(('easy','ordinary','hard')):
        cfg=replace(cfgs[domain],transaction_cost=bp/10000);tables=eqx.filter_jit(build_curve_tables)(cfg);jax.block_until_ready(tables)
        records={a:[] for a in ARMS};costs={a:[] for a in ARMS};defaulted=[];initial=[];daily={a:Daily(cfg.bond_days) for a in ARMS};inventory={a:np.zeros(cfg.cds_days) for a in ARMS};inventory_counts=0;default_day_loss={a:0. for a in ARMS};holding_sum={a:0. for a in ARMS}
        for start in range(0,count,128):
            key=jax.random.fold_in(jax.random.key(seed),di*100000+start);paths=simulate(key,tables,cfg,128)
            if start==0 and audit_callback is not None:audit_callback(domain,paths,tables,cfg)
            raw_block={name:np.asarray(paths[name]) for name in ('state','alive','default')};raw_terminal={}
            for arm,model in models.items():
                r=evaluate(model,paths,tables,cfg,record_inventory=start<256);loss=np.asarray(r['loss']);rec=r['records'];err=np.asarray(rec['daily_loss']);cost=np.asarray(rec['discounted_cost']).sum(1);turnover=cost/cfg.transaction_cost if cfg.transaction_cost else np.asarray(rec['buy']+rec['sold']).sum(1)
                if not np.isfinite(loss).all() or any(not np.isfinite(np.asarray(v)).all() for v in rec.values()):raise RuntimeError('Nonfinite final records')
                if float(np.min(np.asarray(rec['notional']))) < -1e-12 or float(np.max(np.asarray(rec['notional']))) > cfg.max_protection+1e-12:raise RuntimeError('Protection inventory bound failed')
                if start<256:
                    vintage=np.asarray(rec['inventory'])
                    if vintage.min() < -1e-12 or np.max(vintage.sum(-1)) > cfg.max_protection+1e-12:raise RuntimeError('Vintage inventory bound failed')
                    inventory[arm]+=vintage.sum((0,1))
                raw_block['daily_loss_'+arm]=err;raw_block['terminal_loss_'+arm]=loss;raw_terminal[arm]=loss
                default_day_loss[arm]+=float((err*np.asarray(paths['default'])).sum());holding_sum[arm]+=float(np.asarray(rec['notional']).sum());daily[arm].add(err,cost,turnover,loss,~np.asarray(paths['alive'][:,:-1]));records[arm].append(loss);costs[arm].append(cost)
                if arm=='baseline':defaulted.append(np.asarray(r['defaulted']));initial.append(np.asarray(r['initial_price']))
            raw_records.append(archive_block(out/'raw-daily'/domain/f'block-{start:06d}.npz',raw_block,dict(benchmark='credit',domain=domain,seed=seed,fold_in=di*100000+start,path_start=start,path_count=128,days=cfg.bond_days,arms=list(ARMS)),raw_terminal));write(out/'raw-record-manifest.partial.json',raw_records)
            if start<256:inventory_counts+=128*cfg.bond_days
            if start%1024==0:print(json.dumps(dict(event='final_credit',domain=domain,episodes=start+128)),flush=True)
        defaults=np.concatenate(defaulted);values={a:np.concatenate(v) for a,v in records.items()};np.savez_compressed(out/f'terminal-{domain}.npz',**values,**{a+'_cost':np.concatenate(v) for a,v in costs.items()},defaulted=defaults,initial_price=np.concatenate(initial))
        metrics={a:terminal_metrics(v) for a,v in values.items()};daily_values={a:d.finish() for a,d in daily.items()}
        for arm in ARMS:
            metrics[arm].update(mean_cost=daily_values[arm]['mean_cost'],mean_discounted_turnover=daily_values[arm]['mean_discounted_turnover'],default_day_mean_marked_loss=default_day_loss[arm]/int(defaults.sum()) if defaults.any() else None,mean_protection_fixed_calendar=holding_sum[arm]/(count*cfg.bond_days),default_count=int(defaults.sum()),survival_count=int((~defaults).sum()),default_path_mean_loss=float(values[arm][defaults].mean()) if defaults.any() else None,survival_path_mean_loss=float(values[arm][~defaults].mean()) if (~defaults).any() else None)
            if daily_values[arm]['max_accounting_error']>1e-10 or daily_values[arm]['max_absorbing_error']>1e-12:raise RuntimeError('Credit final ledger/default absorption audit failed')
        write(out/f'daily-{domain}.json',daily_values)
        holdings={a:dict(mean_notional_by_vintage=(v/inventory_counts).tolist(),mean_age_days=float(np.arange(cfg.cds_days)@v/v.sum()) if v.sum() else 0.,episodes=256,observation='post-settlement aged inventory; fixed calendar includes zeros after default') for a,v in inventory.items()}
        write(out/f'holdings-{domain}.json',holdings)
        result[domain]=dict(terminal=metrics,daily_marked_errors=daily_values,baseline_minus={a:bootstrap(values['baseline'],values[a],di+101) for a in ARMS if a!='baseline'},specialist_minus=({a:bootstrap(values['easy_reference' if domain=='easy' else 'hard_reference'],values[a],di+501) for a in ARMS[3:]} if domain in ('easy','hard') else {}),initial_price_mean=float(np.concatenate(initial).mean()),default_count=int(defaults.sum()),holdings_by_vintage=holdings,daily_definition='RMS across dates of cross-episode SD; fixed 1260-day calendar includes absorbing zeros after default')
        del tables;gc.collect();jax.clear_caches();write(out/'results.partial.json',result)
    write(out/'raw-record-manifest.json',raw_records)
    return result

def option_evaluate(manifest,out,models,market,seed=520261017,count=8192,daily_count=256,quote_paths=4096):
    m=market.model;initial=manifest['initial_premiums'];result={};raw_records=[];bp=manifest['bp'];cost=bp/10000
    daily_seed=920261017 if seed==520261017 else 620261017
    quote_seed=948 if seed==520261017 else 848
    for di,domain in enumerate(market.names):
        values={a:[] for a in ARMS};costs={a:[] for a in ARMS}
        for start in range(0,count,256):
            paths=m.simulate(jax.random.fold_in(jax.random.key(seed),di*100000+start),domain,256)
            for arm,model in models.items():
                r=m.rollout(model,paths['logs'],cost);values[arm].append(np.asarray(r['loss'])-initial[domain]);costs[arm].append(np.asarray(r['costs']))
        values={a:np.concatenate(v) for a,v in values.items()};np.savez_compressed(out/f'terminal-{domain}.npz',**values,**{a+'_cost':np.concatenate(v) for a,v in costs.items()})
        # Independent, smaller daily-mark diagnostic; never enters selection or policy.
        paths=m.simulate(jax.random.fold_in(jax.random.key(daily_seed),di),domain,daily_count);logs=paths['logs'];states=paths['states'];marks=[];noise=[]
        for t in range(m.HORIZON):
            mark=m.price(jax.random.fold_in(jax.random.key(quote_seed),di*1000+t),states[t],jnp.exp(logs[t]),jnp.ones(daily_count),domain,m.HORIZON-t,quote_paths);marks.append(np.asarray(mark))
            if t in (0,21,63,105):
                independent=m.price(jax.random.fold_in(jax.random.key(quote_seed+1),di*1000+t),states[t],jnp.exp(logs[t]),jnp.ones(daily_count),domain,m.HORIZON-t,quote_paths);noise.append(dict(day=t,rms_independent_price_difference=float(jnp.sqrt(jnp.mean((mark-independent)**2)))))
        s=np.exp(np.asarray(logs));marks.append(np.maximum(s[-1]-1.,0.));dmarks=np.diff(np.stack(marks),axis=0);daily_values={};raw_daily={'logs':np.asarray(logs),'marks':np.stack(marks),'states':np.asarray(states)}
        for arm,model in models.items():
            r=m.rollout(model,logs,cost);h=np.asarray(r['holdings']);old=np.concatenate((np.zeros((1,daily_count)),h[:-1]));trading=cost*np.abs(h-old)*s[:-1];trading[-1]+=np.asarray(r['liquidation']);error=(dmarks-np.asarray(r['gains'])+trading).T
            raw_daily['daily_loss_'+arm]=error;raw_daily['holdings_'+arm]=h.T;raw_daily['stock_gains_'+arm]=np.asarray(r['gains']).T;raw_daily['trading_cost_'+arm]=trading.T
            expected=np.asarray(r['loss'])-marks[0];raw_daily['terminal_loss_'+arm]=expected;d=Daily(m.HORIZON);d.add(error,np.asarray(r['costs']),np.asarray(r['costs'])/cost,expected);daily_values[arm]=d.finish();daily_values[arm]['mean_stock_holding']=float(h.mean())
            if d.max_accounting>1e-10:raise RuntimeError('Option daily marked ledger audit failed')
        raw_records.append(archive_block(out/'raw-daily'/domain/'diagnostic.npz',raw_daily,dict(benchmark='option',domain=domain,seed=daily_seed,fold_in=di,path_count=daily_count,days=m.HORIZON,quote_paths=quote_paths,quote_seed=quote_seed,arms=list(ARMS)),{arm:raw_daily['terminal_loss_'+arm] for arm in ARMS}));write(out/'raw-record-manifest.partial.json',raw_records)
        write(out/f'daily-{domain}.json',daily_values);np.savez_compressed(out/f'daily-marks-{domain}.npz',marks=np.stack(marks),logs=np.asarray(logs))
        result[domain]=dict(terminal={a:dict(**terminal_metrics(v),mean_cost=float(np.concatenate(costs[a]).mean())) for a,v in values.items()},daily_marked_errors=daily_values,daily_quote_error=noise,daily_seed=daily_seed,daily_episodes=daily_count,quote_paths=quote_paths,initial_premium=initial[domain],baseline_minus={a:bootstrap(values['baseline'],values[a],di+101) for a in ARMS if a!='baseline'},specialist_minus=({a:bootstrap(values['easy_reference' if domain=='easy' else 'hard_reference'],values[a],di+501) for a in ARMS[3:]} if domain in ('easy','hard') else {}),daily_definition='RMS across126 dates of cross-episode SD; independent256 episodes, dealer Qmarks include future arrivals, never policy inputs')
        write(out/'results.partial.json',result)
    write(out/'raw-record-manifest.json',raw_records)
    return result

def figures(manifest,out,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    domains=list(results);fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,d in zip(axes,domains):
        y=[results[d]['terminal'][a]['entropic'] for a in ARMS];ax.bar(np.arange(8),y);ax.set_xticks(np.arange(8),ARMS,rotation=70);ax.set_title(d);ax.set_ylabel('Terminal entropic loss')
    fig.tight_layout();fig.savefig(out/'terminal-comparison.pdf');fig.savefig(out/'terminal-comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for ax,d in zip(axes,domains):
        for a in ARMS:ax.plot(results[d]['daily_marked_errors'][a]['sd'],label=a)
        ax.set_title(d);ax.set_xlabel('Trading day');ax.set_ylabel('Cross-episode daily error SD')
    axes[-1].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'daily-errors.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,d in zip(axes,('easy','hard')):
        gaps=results[d].get('specialist_minus',{})
        if gaps:
            names=list(gaps);point=np.asarray([gaps[a]['gain'] for a in names]);bounds=np.asarray([gaps[a]['ci95'] for a in names]);ax.vlines(np.arange(len(names)),bounds[:,0],bounds[:,1]);ax.scatter(np.arange(len(names)),point);ax.set_xticks(np.arange(len(names)),names,rotation=35);ax.axhline(0,color='grey',linewidth=1);ax.set_title(d);ax.set_ylabel('Specialist minus adaptive entropic loss')
    fig.tight_layout();fig.savefig(out/'specialist-gaps.pdf');fig.savefig(out/'specialist-gaps.png',dpi=150);plt.close(fig)
    trials=manifest.get('trial_curves',[])
    if trials:
        fig,axes=plt.subplots(1,3,figsize=(16,5))
        for trial in trials:
            points={d:[] for d in domains}
            for row in trial.get('selection',[]):
                score=json.loads(Path(row['path']).read_text())
                for d in domains:
                    if d in score:points[d].append((row['active_seconds'],score[d]))
            for ax,d in zip(axes,domains):
                ordered=sorted(points[d])
                if ordered:ax.plot(*zip(*ordered),label=f"{trial['arm']} lr={trial['lr']}")
                ax.set_title(d);ax.set_xlabel('Active training seconds');ax.set_ylabel('Domain development entropic risk')
        axes[-1].legend(fontsize=6);fig.tight_layout();fig.savefig(out/'all-trial-learning-curves.pdf');fig.savefig(out/'all-trial-learning-curves.png',dpi=150);plt.close(fig)
    curves=manifest.get('curves',{})
    if curves:
        fig,ax=plt.subplots(figsize=(9,5))
        for arm,paths in curves.items():
            for path in paths:
                rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()];points=[]
                for r in rows:
                    if 'selection_score' in r:points.append((r.get('active_seconds',0),r['selection_score']))
                    elif 'scores' in r:points.append((r.get('active_seconds',0),r['scores'].get('balanced')))
                points=[p for p in points if p[1] is not None]
                if points:ax.plot(*zip(*points),label=arm)
        ax.set_xlabel('Active training seconds');ax.set_ylabel('Own-domain or balanced development score (see legend)');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/'learning-curves.pdf');plt.close(fig)

def training_budgets(manifest):
    """Read actual trial counters; feedback counts follow the frozen block protocol."""
    design=manifest['design']
    if isinstance(design,str):design=json.loads(Path(design).read_text())
    block=int(design['block_size']);batch=int(design['batch_size']);rows=[]
    for arm in ARMS:
        selected=manifest['models'][arm]['selection']
        if isinstance(selected,str):selected=json.loads(Path(selected).read_text())
        trials=selected['trials']
        if len(trials)!=3:raise RuntimeError('Expected all three LR trial counters: '+arm)
        for trial in trials:
            updates=int(trial['learner_updates']);paths=int(trial['training_paths'])
            if updates%block or paths!=updates*batch:raise RuntimeError('Inconsistent training counters: '+arm)
            multiplier=2 if arm.endswith('_mixed') else (1 if arm in ('dpg_hard','soar_hard') else 0)
            feedback=(updates//block)*batch*multiplier
            if 'teacher_feedback_paths' in trial and int(trial['teacher_feedback_paths'])!=feedback:
                raise RuntimeError('Inconsistent teacher feedback counters: '+arm)
            rows.append(dict(arm=arm,lr=trial['lr'],learner_updates=updates,training_paths=paths,
                teacher_feedback_paths=feedback,teacher_feedback_count_source='frozen block protocol from actual updates',
                active_seconds=trial['active_seconds'],compile_seconds=trial['compile_seconds'],
                development_seconds=trial['selection_seconds'],monitor_seconds=trial['monitor_seconds']))
    return dict(trials=rows,scope='All 24 trials, including unselected learning rates. Active time includes sampling/learning/teacher/history work. Compile, development and monitor times are separate. Checkpoint/export and total paid rental runtime require the provider/control supplement.')


def report(manifest,out,result):
    """Worker-side rendering of authoritative metrics, without further selection."""
    freeze=manifest.get('discovery_freeze',{})
    if isinstance(freeze,str):freeze=json.loads(Path(freeze).read_text())
    scope=manifest.get('comparison_scope',freeze.get('comparison_scope','gate status not supplied'))
    eps=manifest.get('adam_eps_root','not supplied');b=chr(92);nl=chr(10)
    title=f"{manifest['benchmark'].title()} specialist and adaptive comparison, {manifest['bp']} bp"
    notes=[f"Scope: {scope}.",
           'All eight checkpoints and learning-rate choices were frozen before terminal test seed 520261017 was opened. Each domain contains 8192 paired whole episodes; bootstrap intervals use 1000 paired whole-episode resamples.',
           'There is one shared training initialization. Intervals describe episode uncertainty conditional on the fitted and selected policies; they do not estimate training-seed uncertainty or prove optimality.',
           'The baseline trains on the ordinary/base family only. Adaptive teachers may sample easy/base/hard, so the comparison changes model-family coverage as well as allocation. Specialists are learned comparators.',
           f'Common learner Adam eps_root: {eps}. When 1e-16 is supplied, all final arms use this shared differentiable-Adam migration; preliminary specialist screens remain separate evidence.',
           'Reported entropic risks and CVaR95 are finite-sample empirical quantities. CVaR uses fractional boundary mass. Entropic ESS is retained in results.json.',
           'No continuous-model convergence or high-default pricing accuracy is inferred from these training/evaluation checks. Numerical claims require the separate supplied pricing audit.',
           'Cost, paid-runtime and provider-shutdown evidence will be attached by campaign control after exports and API verification.']
    if manifest['benchmark']=='option':notes.append('Policy inputs contain causal price history, time and its own holdings only. Daily marked errors use 256 independent episodes and dealer Q prices including future arrivals; Q price replicate errors are reported separately. Final daily stream 920261017 is distinct from terminal 520261017 and preliminary diagnostics. Finite-sample entropic performance does not establish population exponential-moment integrability.')
    else:notes.append('Daily error is RMS across dates of cross-episode SD over the fixed 1260-day calendar, including absorbing zeros after issuer default. Credit ledgers must sum to terminal loss within 1e-10. Raw daily loss/default records are exported in bounded 128-path blocks with lossless readback and ledger checks. Vintage holdings are 256-episode aggregates.')
    md=['# '+title,'',*notes,'']
    tex=[b+'documentclass[10pt]{article}',b+'usepackage[margin=1in]{geometry}',b+'usepackage{graphicx,booktabs,longtable}',b+'begin{document}']
    def esc(value):
        text=str(value)
        for token in ['_','%','&','#']:text=text.replace(token,b+token)
        return text
    tex += [b+'section*{'+esc(title)+'}']+[esc(n)+b+'par' for n in notes]
    budgets=training_budgets(manifest);write(out/'training-budgets.json',budgets)
    md += ['## Actual training budgets','',budgets['scope'],'',
           '| Arm | LR | Updates | Training paths | Feedback paths | Active s | Compile s | Dev s | Monitor s |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    tex += [b+'section*{Actual training budgets}',esc(budgets['scope'])+b+'par',b+'begingroup'+b+'scriptsize',
            b+'begin{longtable}{lrrrrrrrr}',b+'toprule Arm & LR & Updates & Train paths & Feedback & Active s & Compile s & Dev s & Monitor s '+b*2,b+'midrule']
    for row in budgets['trials']:
        vals=[f"{row['lr']:g}",str(row['learner_updates']),str(row['training_paths']),str(row['teacher_feedback_paths'])]+[f"{row[key]:.3f}" for key in ('active_seconds','compile_seconds','development_seconds','monitor_seconds')]
        md.append('| '+row['arm']+' | '+' | '.join(vals)+' |')
        tex.append(esc(row['arm'])+' & '+' & '.join(vals)+' '+b*2)
    tex += [b+'bottomrule',b+'end{longtable}',b+'endgroup']
    md.append('')
    for domain,row in result['domains'].items():
        md += ['## '+domain,'','| Arm | Entropic | CVaR95 | Terminal SD | Daily RMS SD | Mean cost |','|---|---:|---:|---:|---:|---:|']
        tex += [b+'section*{'+esc(domain)+'}',b+'begin{longtable}{lrrrrr}',b+'toprule Arm & Entropic & CVaR95 & Terminal SD & Daily RMS SD & Mean cost '+b*2,b+'midrule']
        for arm in ARMS:
            terminal=row['terminal'][arm];daily=row['daily_marked_errors'][arm];vals=[terminal['entropic'],terminal['cvar95'],terminal['terminal_std'],daily['rms_cross_episode_sd'],daily['mean_cost']]
            md.append('| '+arm+' | '+' | '.join(f'{v:.6g}' for v in vals)+' |')
            tex.append(esc(arm)+' & '+' & '.join(f'{v:.6g}' for v in vals)+' '+b*2)
        tex += [b+'bottomrule',b+'end{longtable}']
        md += ['','Baseline minus method entropic gains (positive means improvement):','']
        for arm,g in row['baseline_minus'].items():
            line=f"{arm}: {g['gain']:.6g}, 95% paired interval [{g['ci95'][0]:.6g}, {g['ci95'][1]:.6g}]."
            md.append('- '+line);tex.append(esc(line)+b+'par')
        if row.get('specialist_minus'):
            md += ['', 'Own-domain specialist minus adaptive entropic gains (positive means the adaptive method improves on the specialist):', '']
            tex.append(b+'subsection*{Direct specialist comparisons}')
            for arm,g in row['specialist_minus'].items():
                line=f"{arm}: {g['gain']:.6g}, 95% paired interval [{g['ci95'][0]:.6g}, {g['ci95'][1]:.6g}]."
                md.append('- '+line);tex.append(esc(line)+b+'par')
        if 'daily_quote_error' in row:
            md += ['', 'Daily Q mark independent-replicate RMS differences:', '']
            for q in row['daily_quote_error']:
                line=f"Day {q['day']}: {q['rms_independent_price_difference']:.6g}.";md.append('- '+line);tex.append(esc(line)+b+'par')
    retention=result['base_retention_domain'];md += ['','Base retention is reported in the '+retention+' domain above; compare each adaptive policy against baseline without selecting on this final result.','']
    for image in ['terminal-comparison.pdf','specialist-gaps.pdf','daily-errors.pdf','all-trial-learning-curves.pdf']:
        if (out/image).exists():tex += [b+'begin{figure}[p]'+b+'centering',b+'includegraphics[width='+b+'textwidth]{'+image+'}',b+'end{figure}']
    tex.append(b+'end{document}')
    (out/'report.md').write_text(nl.join(md)+nl);(out/'report.tex').write_text(nl.join(tex)+nl)
    import shutil,subprocess
    latex=shutil.which('pdflatex')
    if latex:
        with (out/'latex-build.log').open('w') as log:
            subprocess.run([latex,'-interaction=nonstopmode','-halt-on-error','report.tex'],cwd=out,stdout=log,stderr=log,timeout=60,check=False)

def main(manifest_path,output):
    if jax.default_backend()!='gpu':raise RuntimeError('GPU worker required for all evaluation')
    jax.config.update('jax_enable_x64',True);manifest=json.loads(Path(manifest_path).read_text());digest=validate_manifest(manifest);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    write(out/'frozen-evaluation-manifest.json',manifest);write(out/'test-opening.json',dict(manifest_sha256=digest,seed=520261017,all_eight_frozen=True,unix_time=time.time()))
    if manifest['benchmark']=='option':
        from option_adapter import OptionAdapter
        market=OptionAdapter(manifest['source'],manifest['bp']);models=load_models(manifest,market.make_model);results=option_evaluate(manifest,out,models,market)
    else:
        from credit_hedging.learned_policy import TargetHedger
        models=load_models(manifest,lambda seed:TargetHedger(jax.random.key(seed)));results=credit_evaluate(manifest,out,models)
    result=dict(benchmark=manifest['benchmark'],bp=manifest['bp'],seed=520261017,episodes=8192,bootstrap=1000,training_initializations=1,raw_record_manifest='raw-record-manifest.json',manifest_sha256=digest,domains=results,base_retention_domain='base' if manifest['benchmark']=='option' else 'ordinary',status='complete')
    figures(manifest,out,results);report(manifest,out,result);write(out/'results.json',result);write(out/'complete.json',dict(status='complete',manifest_sha256=digest))

if __name__=='__main__':main(sys.argv[1],sys.argv[2])
