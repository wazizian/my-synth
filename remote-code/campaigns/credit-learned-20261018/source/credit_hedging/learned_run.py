"""GPU learned credit baseline/specialist tuning with independent selection streams."""
from pathlib import Path
import json,time,hashlib,os,gc
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from .curve_market import CurveConfig,build_curve_tables,simulate_curve_paths
from .environment import rollout
from .learned_policy import TargetHedger
from .diagnostic_policies import FixedProtection
from .metrics import entropic_risk


def write(path,value):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def configs(source):
    return {d:CurveConfig(**v['production_config']) for d,v in json.loads((source/'numerical_summary.json').read_text()).items()}

@eqx.filter_jit
def evaluate(model,paths,tables,cfg):return rollout(model,paths,tables,cfg)

@eqx.filter_jit
def simulate(key,tables,cfg,n):return simulate_curve_paths(key,tables,cfg,n)

@eqx.filter_jit
def update(model,state,key,tables,cfgs,domains,counts,weights,optimizer):
    keys=jax.random.split(key,len(domains)+1)
    paths=tuple(simulate_curve_paths(keys[i+1],tables[d],cfgs[d],counts[i]) for i,d in enumerate(domains))
    def objective(candidate):
        # Fixed stratification estimates the mixture exponential moment, not an
        # average of per-domain entropic risks. Gradients use exact frozen weights.
        losses=tuple(rollout(candidate,p,tables[d],cfgs[d])['loss'] for p,d in zip(paths,domains))
        loss=jnp.concatenate(losses)
        logweights=jnp.concatenate(tuple(jnp.full(n,jnp.log(w/n)) for n,w in zip(counts,weights)))
        return jax.scipy.special.logsumexp(10.*loss+logweights)/10.
    value,grad=eqx.filter_value_and_grad(objective)(model)
    params=eqx.filter(model,eqx.is_inexact_array)
    updates,state=optimizer.update(grad,state,params)
    return eqx.apply_updates(model,updates),state,keys[0],value,optax.global_norm(grad)


def selection(model,tables,cfgs,domains,weights,seed,count=2048,archive=None):
    pieces=[];max_error=0.
    for d in domains:
        rows=[]
        for start in range(0,count,128):
            key=jax.random.fold_in(jax.random.key(seed),('easy','ordinary','hard').index(d)*100000+start)
            paths=simulate(key,tables[d],cfgs[d],128)
            result=evaluate(model,paths,tables[d],cfgs[d])
            loss=np.asarray(result['loss']);daily=np.asarray(result['records']['daily_loss'])
            if not np.isfinite(loss).all() or not np.isfinite(daily).all():raise RuntimeError('Nonfinite selection')
            err=float(np.max(np.abs(daily.sum(1)-loss)));max_error=max(max_error,err)
            if err>1e-10:raise RuntimeError('Selection accounting failed')
            rows.append(loss)
        pieces.append(np.concatenate(rows))
    if archive is not None:
        np.savez_compressed(archive,**dict(zip(domains,pieces)))
    values=np.concatenate(pieces);w=np.concatenate([np.full(count,weight/count) for weight in weights])
    z=10.*values;maximum=z.max();risk=float((maximum+np.log(w@np.exp(z-maximum)))/10.)
    return dict(entropic=risk,max_accounting_error=max_error,episodes_per_component=count)


def main(source,output,arm):
    jax.config.update('jax_enable_x64',True)
    assert jax.default_backend()=='gpu'
    out=Path(output);src=Path(source);out.mkdir(exist_ok=True)
    design=json.loads((src/'learned_design.json').read_text());cfgs=configs(src)
    domains=('easy','ordinary','hard') if arm=='baseline' else (arm,)
    weights=(.05,.90,.05) if arm=='baseline' else (1.,)
    counts=(8,144,8) if arm=='baseline' else (160,)
    tables={}
    for d in domains:
        t=time.monotonic();tables[d]=eqx.filter_jit(build_curve_tables)(cfgs[d]);jax.block_until_ready(tables[d])
        print(json.dumps(dict(event='tables_ready',domain=d,seconds=time.monotonic()-t)),flush=True)
    model0=TargetHedger(jax.random.key(design['seed']))
    # Full-calendar policy equivalence check: initialization must exactly match static rule.
    d=domains[0];paths=simulate(jax.random.key(910000+design['seed']),tables[d],cfgs[d],16)
    a=evaluate(model0,paths,tables[d],cfgs[d]);b=evaluate(FixedProtection(1.),paths,tables[d],cfgs[d])
    error=max(float(jnp.max(jnp.abs(a['loss']-b['loss']))),float(jnp.max(jnp.abs(a['records']['notional']-b['records']['notional']))))
    if error>1e-12:raise RuntimeError('Initialization does not match static hedge')
    write(out/'initialization_validation.json',dict(max_error=error,model_sha256=hashlib.sha256(b''.join(np.asarray(x).tobytes() for x in jax.tree.leaves(model0) if eqx.is_array(x))).hexdigest(),backend=jax.default_backend()))
    initial=selection(model0,tables,cfgs,domains,weights,design['selection_seed'],archive=out/'initial-selection.npz')
    trials=[]
    for lr in design['learning_rates']:
        trial=out/f'lr-{lr:g}';trial.mkdir(exist_ok=True)
        optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(lr))
        model=model0;state=optimizer.init(eqx.filter(model,eqx.is_inexact_array));key=jax.random.key(design['training_seed'])
        active=0.;steps=0;history=[];best=initial['entropic'];best_step=0;best_active=0.
        def save(tag):
            path=trial/(tag+'.eqx');tmp=path.with_name(path.name+'.tmp')
            eqx.tree_serialise_leaves(tmp,(model,state,jax.random.key_data(key)));os.replace(tmp,path)
        save('initial');save('best');save('latest')
        # First update is separately timed as compilation and retained, not free extra updates.
        t=time.monotonic();model,state,key,value,gn=update(model,state,key,tables,cfgs,domains,counts,weights,optimizer);jax.block_until_ready(value)
        compile_time=time.monotonic()-t;steps=1
        print(json.dumps(dict(event='compiled',arm=arm,lr=lr,seconds=compile_time,loss=float(value),gradient=float(gn))),flush=True)
        for boundary in design['checkpoint_active_seconds'][1:]:
            with (trial/'train.jsonl').open('a') as log:
                while active<boundary:
                    t=time.monotonic();model,state,key,value,gn=update(model,state,key,tables,cfgs,domains,counts,weights,optimizer);jax.block_until_ready(value);active+=time.monotonic()-t;steps+=1
                    if not np.isfinite(float(value)) or not np.isfinite(float(gn)):raise RuntimeError('Nonfinite training')
                    record=dict(step=steps,active_seconds=active,loss=float(value),gradient_norm=float(gn));log.write(json.dumps(record)+'\n');log.flush()
                    if steps%20==0:print(json.dumps(dict(event='train',arm=arm,lr=lr,**record)),flush=True)
            save('latest');score=selection(model,tables,cfgs,domains,weights,design['selection_seed'],archive=trial/f'selection-{boundary}.npz')
            history.append(dict(active_seconds=active,steps=steps,**score))
            if score['entropic']<best:
                best=score['entropic'];best_step=steps;best_active=active;save('best')
            save('active-'+str(boundary))
            write(trial/'progress.json',dict(lr=lr,initial=initial,history=history,best=best,best_step=best_step,best_active_seconds=best_active,compile_seconds=compile_time,steps=steps,paths=steps*160))
            print(json.dumps(dict(event='selection',arm=arm,lr=lr,best=best,**history[-1])),flush=True)
        trials.append(dict(lr=lr,entropic=best,checkpoint=str(trial/'best.eqx'),steps=steps,active_seconds=active,best_step=best_step,best_active_seconds=best_active,compile_seconds=compile_time))
        write(out/'trials.json',trials)
        del model,state;gc.collect()
    chosen=min(trials,key=lambda row:row['entropic'])
    (out/'selected.eqx').write_bytes(Path(chosen['checkpoint']).read_bytes())
    write(out/'selection.json',dict(arm=arm,criterion='own training-distribution entropic risk on fixed development episodes',initial=initial,selected=chosen,trials=trials,final_test_opened=False))
    write(out/'complete.json',dict(arm=arm,status='training_and_development_selection_complete',final_test_opened=False))
    print(json.dumps(dict(event='complete',arm=arm,selected=chosen)),flush=True)
