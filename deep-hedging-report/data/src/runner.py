"""Persistent single-GPU trial execution. Rentals and selection belong to the supervisor."""
from pathlib import Path
import datetime
import fcntl
import json
import os
import signal
import shutil
import socket
import subprocess
import time
import traceback
import uuid

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .checkpoint import atomic_json, save_checkpoint, load_checkpoint
from .datasets import load_dataset
from .generator import (init_generator, probabilities, base_probabilities,
                        hard_probabilities, distribution_metrics)
from .hedger import Hedger
from .market import simulate, DEFAULT_CONFIG as MARKET_CONFIG
from .metrics import summarize_outcomes, save_episode_metrics, aggregate_domains
from .objective import example_losses, rollout, make_optimizer, student_block
from .pricing import load_tables, table_hash
from .protocol import ARMS, DOMAINS, DEFAULT_CONFIG, canonical_hash, file_hash, source_hash
from .runtime import configure, require_gpu
from .teachers import (dpg_step, soar_objective_step, selfplay_step,
                       make_target_objective, detached)


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(x) for x in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return _plain(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _append(path, record):
    with open(path, 'a') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        stream.write(json.dumps(_plain(record), sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _manifest(root, config, record, event_type):
    run_dir = Path('runs')/config.get('run_id',config.get('method','invalid_trial'))
    _append(Path(root)/'runs/manifest.jsonl',dict(
        event_type=event_type,timestamp=_utc(),run_id=run_dir.name,
        method=config.get('method'),trial_id=config.get('trial_id',run_dir.name),
        parent_id=config.get('parent_id'),parent_checkpoint_id=config.get('parent_checkpoint_id'),
        config_hash=record.get('config_hash',canonical_hash(config)),
        source_hash=record.get('source_hash'),status=record.get('status'),
        checkpoint_id=record.get('last_checkpoint_id'),error=record.get('error'),
        active_training_seconds=record.get('active_training_seconds'),
        artifacts={'run_directory':str(run_dir),'config':str(run_dir/'config.json'),
                   'status':str(run_dir/'status.json'),'checkpoints':str(run_dir/'checkpoints'),
                   'training':str(run_dir/'train_metrics.jsonl'),
                   'evaluation':str(run_dir/'evaluation_metrics.jsonl'),
                   'episodes':str(run_dir/'episode_metrics')}))


def _sync(tree):
    for leaf in jax.tree.leaves(tree):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _allocator_highwater(device):
    """Read a worker allocator's process-wide high-water marks without reset.

    JAX memory_stats may be unavailable on a backend. These counters include
    earlier sequential runs and evaluation in this process, not isolated trials.
    API reference: https://codelabs.developers.google.com/jax-gpu-03-profiling-debugging
    """
    result=dict(allocator_peak_bytes_in_use=None,allocator_peak_pool_bytes=None,
        allocator_memory_scope='current_process_device_allocator_lifetime_including_earlier_runs_and_evaluation_not_isolated_trial',
        allocator_memory_device=str(device),allocator_memory_pid=os.getpid(),
        allocator_memory_supported=False,allocator_memory_error=None)
    try:
        stats=device.memory_stats()
        if stats:
            for key in ('peak_bytes_in_use','peak_pool_bytes'):
                value=stats.get(key)
                if value is not None and int(value)>=0:
                    result['allocator_'+key]=int(value)
            result['allocator_memory_supported']=any(result['allocator_'+key] is not None
                for key in ('peak_bytes_in_use','peak_pool_bytes'))
    except Exception as error:
        result['allocator_memory_error']=repr(error)
    return result


def _config(config):
    result = dict(DEFAULT_CONFIG, **config)
    if result.get('method') not in ARMS:
        raise ValueError('Unknown seven-arm method')
    result.setdefault('run_id', result['method'])
    if Path(result['run_id']).name != result['run_id']:
        raise ValueError('run_id must be a single directory name')
    result.setdefault('trial_id', result['run_id'])
    result.setdefault('campaign_id', 'adaptive-hedging')
    return result


def _inputs(root):
    manifest = json.loads((root / 'data/manifest.json').read_text())
    scales = json.loads((root / 'objective_scales.json').read_text())['scales']
    validity = json.loads((root / 'cache/pricing/heston.validation.json').read_text())
    if not validity['passed']:
        raise ValueError('Pricing validity gate failed')
    if validity.get('sha256') != manifest['pricing_hash']:
        raise ValueError('Pricing validity evidence belongs to a different table')
    if manifest.get('objective_scales_hash') != file_hash(root/'objective_scales.json'):
        raise ValueError('Frozen objective scales content mismatch')
    fallback=json.loads((root/'cache/pricing/fallback.validation.json').read_text())
    if not fallback['passed'] or fallback['pricing_hash']!=manifest['pricing_hash'] or fallback['pricing_source_hash']!=file_hash(root/'src/pricing.py'):
        raise ValueError('Fallback pricing validation missing or stale')
    hashes = dict(benchmark=manifest['benchmark_hash'], datasets=canonical_hash(manifest['datasets']),
                  pricing=manifest['pricing_hash'], scales=file_hash(root / 'objective_scales.json'))
    return manifest, scales, hashes


def _skeleton(config):
    key = jax.random.key(config['seed'])
    # Exactly the same model initialization for every method and LR candidate.
    model = Hedger(jax.random.fold_in(key, 1), hidden_size=config['hidden_size'])
    optimizer = make_optimizer(config['learner_lr'])
    generator_optimizer = optax.adam(config['generator_lr'])
    state = dict(model=model, optimizer=optimizer.init(eqx.filter(model, eqx.is_inexact_array)),
                 generator=init_generator(generator_optimizer),
                 regime_key=jax.random.fold_in(key, 2), noise_key=jax.random.fold_in(key, 3),
                 teacher_key=jax.random.fold_in(key, 50), baseline=jnp.array(0.),
                 outer_round=jnp.array(0, dtype=jnp.int32))
    return state, optimizer, generator_optimizer


def _base_metadata(root, config, hashes, devices):
    def environment_number(name, fallback=None):
        raw = os.getenv(name)
        if raw is None:
            return fallback
        try:
            value = float(raw)
        except ValueError as error:
            raise ValueError('Invalid numeric worker metadata: '+name) from error
        if not np.isfinite(value):
            raise ValueError('Nonfinite worker metadata: '+name)
        return value
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    method = config['method']
    mode = 'mixed' if method.endswith('_mixed') else ('hard' if method.endswith('_hard') else
            ('intrinsic' if method == 'selfplay' else 'none'))
    return dict(campaign_id=config['campaign_id'], run_id=config['run_id'],
                trial_id=config['trial_id'], parent_id=config.get('parent_id'), method=method,
                target_mode=mode, config_hash=canonical_hash(config), hashes=hashes,
                benchmark_hash=hashes['benchmark'], dataset_manifest_hash=hashes['datasets'],
                pricing_table_hash=hashes['pricing'], code_commit=commit, source_hash=source_hash(root),
                master_seed=config['seed'], worker=socket.gethostname(),
                instance=os.getenv('INSTANCE_ID', config.get('instance_id')),
                gpu_model=devices[0].device_kind, device=str(devices[0]),
                instance_rate_per_hour=environment_number('INSTANCE_RATE_PER_HOUR',config.get('instance_rate_per_hour')),
                rental_started_at=environment_number('RENTAL_STARTED_AT'),
                campaign_cost_at_launch=environment_number('CAMPAIGN_COST_AT_LAUNCH'),
                campaign_cost_at_launch_scope='launch_snapshot_not_live_cross_worker_estimate',
                campaign_start=config.get('campaign_start'), deadline=config.get('deadline'),
                config=config)


def _event(base, counters, **extra):
    record = dict(base, **counters, timestamp=_utc(), **extra)
    now = time.time()
    start = base.get('campaign_start')
    record['campaign_elapsed_seconds'] = max(0.,now-start) if isinstance(start,(int,float)) else None
    rental_start,rate = base.get('rental_started_at'),base.get('instance_rate_per_hour')
    paid = max(0.,now-rental_start) if isinstance(rental_start,(int,float)) else None
    record['paid_elapsed_seconds'] = paid
    record['paid_elapsed_scope'] = 'current_instance_since_rental_start_including_setup_and_idle'
    record['instance_cost_estimate'] = paid*rate/3600 if paid is not None and isinstance(rate,(int,float)) else None
    record['instance_cost_scope'] = 'current_instance_compute_only_central_ledger_adds_other_charges'
    deadline = base.get('deadline')
    record['deadline_remaining_seconds'] = max(0.,deadline-now) if isinstance(deadline,(int,float)) else None
    record['campaign_cost_estimate'] = None  # Central ledger owns cross-worker money.
    record['budget_scope'] = 'central_supervisor_ledger'
    return record


def _compiled(function, args, counters):
    start = time.perf_counter()
    result = eqx.filter_jit(function).lower(*args).compile()
    counters['compile_seconds'] += time.perf_counter()-start
    return result


def _initial_counters():
    return dict(active_training_seconds=0., compile_seconds=0., teacher_seconds=0.,
                pricing_device_setup_seconds=0.,
                monitoring_seconds=0., selection_seconds=0., final_test_seconds=0.,
                checkpoint_seconds=0., simulation_seconds=0., integrated_update_seconds=0.,
                feedback_seconds=None, feedback_timing_scope='included_in_integrated_teacher_update',
                learner_steps=0, outer_rounds=0, synthetic_path_count=0,
                feedback_episode_evaluation_count=0, paid_elapsed_seconds=None,
                last_monitor_active_seconds=0., last_selection_active_seconds=0.,
                latest_evaluation_event_ids={})


def _evaluate(root, run_dir, model, config, manifest, scales, base, counters,
              checkpoint_id, role, evaluator=None):
    require_gpu()
    if role not in ('monitor', 'selection', 'final_test'):
        raise ValueError('Unknown evaluation role')
    started = time.perf_counter()
    compile_before = counters['compile_seconds']
    output = {}
    compiled_evaluators = {}
    batch_size = int(config['evaluation_batch_size'])
    if batch_size > 256:
        raise ValueError('Evaluation must stream at most 256 episodes per GPU batch')
    for domain in DOMAINS:
        dataset, record = load_dataset(root, role, domain, allow_final=role == 'final_test')
        pieces = []
        for offset in range(0, record['count'], batch_size):
            batch = {key: jnp.asarray(dataset[key][offset:offset+batch_size]) for key in ('spot','call')}
            if evaluator is None:
                evaluator = eqx.filter_jit(lambda m,b: rollout(m,b,config['task']))
            size = batch['spot'].shape[0]
            if size not in compiled_evaluators:
                compile_started = time.perf_counter()
                compiled_evaluators[size] = evaluator.lower(model,batch).compile()
                counters['compile_seconds'] += time.perf_counter()-compile_started
            shortfall, diagnostics = compiled_evaluators[size](model, batch)
            # Keep compact outcome vectors on GPU; discard full position paths.
            diagnostics = {k:v for k,v in diagnostics.items() if k != 'holdings'}
            pieces.append((shortfall, diagnostics))
        shortfall = jnp.concatenate([p[0] for p in pieces])
        diagnostics = jax.tree.map(lambda *xs:jnp.concatenate(xs), *[p[1] for p in pieces])
        _sync((shortfall,diagnostics))
        metrics = summarize_outcomes(shortfall, diagnostics, scale=scales[domain], domain=domain,
                    split_role=role, data_hash=record['sha256'], checkpoint_id=checkpoint_id)
        event_id = uuid.uuid4().hex
        archive = run_dir / 'episode_metrics' / f'{role}_{domain}_{checkpoint_id}_{event_id}.npz'
        save_episode_metrics(archive, shortfall, diagnostics)
        metrics.update(event_id=event_id, episode_metrics_path=str(archive.relative_to(root)),
                       pricing_out_of_bounds_count=int(jnp.sum(jnp.asarray(dataset.get('out_of_bounds',0)))))
        counters['latest_evaluation_event_ids'][f'{role}_{domain}'] = event_id
        output[domain] = metrics
        _append(run_dir/'evaluation_metrics.jsonl', _event(base,counters,event_type='evaluation',**metrics))
        if metrics['status'] != 'ok':
            raise FloatingPointError(f'Nonfinite {role}/{domain} outcomes')
    aggregate = aggregate_domains(output,scales)
    elapsed = time.perf_counter()-started-(counters['compile_seconds']-compile_before)
    counters[{'monitor':'monitoring_seconds','selection':'selection_seconds','final_test':'final_test_seconds'}[role]] += elapsed
    _append(run_dir/'evaluation_metrics.jsonl', _event(base,counters,event_type='evaluation_aggregate',
                split_role=role,checkpoint_id=checkpoint_id,**aggregate))
    return dict(domains=output, **aggregate)


def _snapshot(directory, index, model):
    directory.mkdir(exist_ok=True)
    path = directory / f'{index:08d}.eqx'
    temporary = directory / f'.{index:08d}.{uuid.uuid4().hex}.tmp'
    with open(temporary,'wb') as stream:
        eqx.tree_serialise_leaves(stream,model)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)
    return dict(index=index,path=path.name,sha256=file_hash(path),bytes=path.stat().st_size)


def _history_model(directory, record, skeleton):
    path = directory/record['path']
    if file_hash(path) != record['sha256']:
        raise ValueError('Historical snapshot checksum mismatch')
    return eqx.tree_deserialise_leaves(path,skeleton)


def _train_one(root, config, active_seconds, resume=True):
    """Train up to a cumulative active-time cap; preserve exact continuing state."""
    root = Path(root).resolve()
    devices = configure(root)
    config = _config(config)
    run_dir = root/'runs'/config['run_id']
    run_dir.mkdir(parents=True,exist_ok=True)
    config_path = run_dir/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Immutable run configuration differs; use a new run_id')
    atomic_json(config_path,config)
    manifest,scales,hashes = _inputs(root)
    state,optimizer,generator_optimizer = _skeleton(config)
    base = _base_metadata(root,config,hashes,devices)
    counters = _initial_counters()
    history = []
    best_score = None
    last_checkpoint = None
    checkpoint_dir = run_dir/'checkpoints'
    if (checkpoint_dir/'latest.json').exists():
        if not resume:
            raise ValueError('Run already exists; resume or choose a new run_id')
        state, previous = load_checkpoint(checkpoint_dir,state,expected_hashes=hashes)
        if previous['config_hash'] != base['config_hash']:
            raise ValueError('Resume config hash mismatch')
        counters.update(previous['counters'])
        history = previous.get('history',[])
        best_score = previous.get('selection_score')
        last_checkpoint = previous['checkpoint_id']
        continuation_path = run_dir/'continuation.json'
        if continuation_path.exists():
            continuation = json.loads(continuation_path.read_text())
            if continuation['checkpoint_id'] == last_checkpoint:
                counters.update(continuation['counters'])
                best_score = continuation['selection_score']
        # Selection is published after saving its numerical checkpoint. Recover
        # from the authoritative pointer and append-only measured score, even if
        # the process stopped before writing another checkpoint or continuation.
        best_pointer = checkpoint_dir/'best.json'
        if best_pointer.exists():
            best_id = json.loads(best_pointer.read_text())['checkpoint_id']
            scores = []
            with open(run_dir/'evaluation_metrics.jsonl') as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    score = event.get('balanced_objective')
                    if (event.get('event_type')=='evaluation_aggregate' and
                        event.get('split_role')=='selection' and
                        event.get('checkpoint_id')==best_id and
                        isinstance(score,(int,float)) and np.isfinite(score)):
                        scores.append(score)
            if not scores:
                raise ValueError('Best checkpoint has no verified selection score')
            best_score = min(scores)
    stop = {'requested':False}
    old_handlers = {}
    for signum in (signal.SIGTERM,signal.SIGINT):
        old_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum,lambda *_:stop.update(requested=True))

    def save(tag='latest'):
        nonlocal last_checkpoint
        started = time.perf_counter()
        record = save_checkpoint(checkpoint_dir,state,
            dict(base,counters=_plain(counters),history=history,selection_score=best_score,
                 objective_scales=scales,created_at=_utc()),tag=tag)
        counters['checkpoint_seconds'] += time.perf_counter()-started
        last_checkpoint = record['checkpoint_id']
        return record

    previous_status = None
    def status(value, **extra):
        nonlocal previous_status
        result = _event(base,counters,status=value,last_checkpoint_id=last_checkpoint,
                        selection_score=best_score,**extra)
        if value!='running':
            result.update(_allocator_highwater(devices[0]))
            _append(run_dir/'train_metrics.jsonl',dict(result,event_type='allocator_memory'))
        atomic_json(run_dir/'status.json',_plain(result))
        if value != previous_status:
            _manifest(root,config,result,'run_started' if value=='running' else 'run_terminal')
            previous_status = value
        return result

    status('running')
    try:
        method = config['method']
        history_bytes = 0
        history_limit = int(config.get('selfplay_history_limit_bytes',12*1024**3))
        history_reserve = int(config.get('selfplay_storage_reserve_bytes',1024**3))
        history_max_blocks = config.get('selfplay_max_blocks')
        if method=='selfplay':
            validation = config.get('purpose') in ('integration_smoke','production_shape_profile')
            if not validation and history_max_blocks is None:
                raise ValueError('Production selfplay requires an immutable profile-derived block/storage guard')
            if validation and active_seconds>60:
                raise ValueError('Unprofiled selfplay validation is restricted to 60 cumulative active seconds')
            if history_max_blocks is not None and (history_max_blocks<=0 or active_seconds>config['selfplay_planned_active_seconds']):
                raise ValueError('Selfplay requested runtime exceeds its frozen block/storage plan')
            if not 0<history_limit<=12*1024**3 or history_reserve<0:
                raise ValueError('Invalid selfplay history storage bound')
            history_directory=run_dir/'history'
            history_bytes=sum((history_directory/record['path']).stat().st_size for record in history)
            arrays=[x for x in jax.tree.leaves(state['model']) if eqx.is_array(x)]
            # Fixed architecture serialization size; a conservative estimate is
            # needed only before validation's first actual snapshot is written.
            history_snapshot_bytes=int(config.get('selfplay_snapshot_estimate_bytes',
                sum(x.size*x.dtype.itemsize+1024 for x in arrays)))
            if history:
                history_snapshot_bytes=max(history_snapshot_bytes,max((history_directory/record['path']).stat().st_size for record in history))
            remaining_snapshots=max(0,int(history_max_blocks)-len(history)) if history_max_blocks is not None else 1
            projected=remaining_snapshots*history_snapshot_bytes
            if history_bytes+projected>history_limit or shutil.disk_usage(run_dir).free<projected+history_reserve:
                raise ValueError('Insufficient storage for remaining frozen selfplay snapshot plan')
        tables = load_tables(root/'cache/pricing/heston.npz')
        if table_hash(tables) != hashes['pricing']:
            raise ValueError('Pricing cache content mismatch')
        # load_tables returns host NumPy arrays for the immutable cache. Move
        # them once, after identity verification, instead of retransferring the
        # entire table for every simulator invocation. asarray applies the same
        # configured dtype canonicalization as the previous JIT boundary (float32
        # with learner x64 disabled); uint32 quadrature bit patterns stay intact.
        transfer_started=time.perf_counter()
        tables={name:jax.device_put(jnp.asarray(value),devices[0]) for name,value in tables.items()}
        _sync(tables)
        counters['pricing_device_setup_seconds']+=time.perf_counter()-transfer_started
        method = config['method']
        adaptive = method not in ('baseline','hard_reference')
        target = components = None
        if method.startswith(('dpg_','soar_')):
            hard,_ = load_dataset(root,'feedback','hard')
            hard = {k:jnp.asarray(hard[k]) for k in ('spot','call')}
            easy = None
            if method.endswith('_mixed'):
                easy,_ = load_dataset(root,'feedback','easy')
                easy = {k:jnp.asarray(easy[k]) for k in ('spot','call')}
            target,components = make_target_objective(base['target_mode'],
                lambda m,b:example_losses(m,b,config['task']),hard,scales,easy)
        nsteps,batch_size = config['block_steps'],config['batch_size']
        market = dict(MARKET_CONFIG,**config['market'])
        kwargs = dict(epsilon=config['epsilon'],kl_coefficient=config['kl_coefficient'])
        fixed_q = hard_probabilities() if method == 'hard_reference' else base_probabilities()

        def simulate_block(s,price_tables):
            regime_key,rkey = jax.random.split(s['regime_key'])
            noise_key,nkey = jax.random.split(s['noise_key'])
            q = probabilities(s['generator'].logits,epsilon=config['epsilon']) if adaptive else fixed_q
            ids = jax.random.categorical(rkey,jnp.log(q),shape=(nsteps,batch_size))
            batch = simulate(ids.reshape(-1),nkey,market,price_tables)
            diagnostics = dict(pricing_out_of_bounds_count=jnp.sum(batch['out_of_bounds']),
                               max_grid_distance=jnp.max(batch['max_grid_distance']),
                               simulator_nonfinite_count=jnp.sum(~jnp.isfinite(batch['spot']))+jnp.sum(~jnp.isfinite(batch['call'])))
            batch = {k:batch[k].reshape((nsteps,batch_size)+batch[k].shape[1:]) for k in ('spot','call')}
            return batch,ids,regime_key,noise_key,diagnostics

        def block(start,batch,weights):
            post,losses,diagnostics = student_block(start[0],start[1],batch,weights,optimizer,
                config['task'],return_diagnostics=True)
            return post,dict(losses=losses,**diagnostics)

        def update(s,batch,ids,historical):
            student = (s['model'],s['optimizer'])
            generator = s['generator']
            baseline = s['baseline']
            if method.startswith('dpg_'):
                generator,post,learner,teacher = dpg_step(generator,generator_optimizer,
                    student,batch,ids,block,target,target_components_fn=components,**kwargs)
            else:
                if method == 'selfplay':
                    flat_batch = {k:v.reshape((-1,)+v.shape[2:]) for k,v in batch.items()}
                    generator,teacher = selfplay_step(generator,generator_optimizer,
                        student[0],historical,student[1],flat_batch,
                        lambda m,b:example_losses(m,b,config['task']),ids.reshape(-1),
                        config['learner_lr'],s['outer_round'],
                        expert_iteration_coefficient=config['expert_coefficient'],**kwargs)
                post,learner = block(student,batch,jnp.ones(ids.shape))
                if method.startswith('soar_'):
                    generator,baseline,teacher = soar_objective_step(generator,generator_optimizer,
                        student,post,target,ids,baseline,target_components_fn=components,**kwargs)
                elif not adaptive:
                    teacher = dict(probabilities=fixed_q,entropy=-jnp.sum(fixed_q*jnp.log(fixed_q)),
                                   kl=jnp.sum(fixed_q*jnp.log(fixed_q/base_probabilities())))
            result = dict(s,model=post[0],optimizer=post[1],generator=generator,
                          baseline=baseline,outer_round=s['outer_round']+1)
            leaves = [x for x in jax.tree.leaves(result) if eqx.is_inexact_array(x)]
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))
            metrics = dict(teacher,synthetic_loss=jnp.mean(learner['losses']),
                learner_gradient_norm=jnp.mean(learner['gradient_norm']),
                learner_gradient_norm_max=jnp.max(learner['gradient_norm']),
                clipping_fraction=jnp.mean(learner['clipping_applied'].astype(jnp.float32)),
                learner_gradient_nonfinite_count=jnp.sum(learner['gradient_nonfinite_count']),
                learner_loss_nonfinite_count=jnp.sum(learner['loss_nonfinite']),state_finite=finite)
            return detached(result),metrics

        evaluator = eqx.filter_jit(lambda m,b:rollout(m,b,config['task']))
        if last_checkpoint is None:
            save()
            _append(run_dir/'train_metrics.jsonl',_event(base,counters,event_type='initialization',
                checkpoint_id=last_checkpoint,generator_probabilities=_plain(fixed_q),synthetic_loss=None))
            _evaluate(root,run_dir,state['model'],config,manifest,scales,base,counters,last_checkpoint,'monitor',evaluator)
            initial_selection = _evaluate(root,run_dir,state['model'],config,manifest,scales,base,counters,last_checkpoint,'selection',evaluator)
            best_score = initial_selection['balanced_objective']
            atomic_json(checkpoint_dir/'best.json',json.loads((checkpoint_dir/'latest.json').read_text()))
        status('running')
        simulator = _compiled(simulate_block,(state,tables),counters)
        compiled_update = None
        metrics = None
        last_saved_active = counters['active_training_seconds']
        last_log_active = counters['active_training_seconds']
        cached_history_index = None
        cached_history_model = state['model']
        while counters['active_training_seconds'] < active_seconds and not stop['requested']:
            deadline = config.get('deadline')
            if isinstance(deadline,(int,float)) and time.time() >= deadline:
                stop['requested'] = True
                break
            outer = int(state['outer_round'])
            historical = state['model']
            if method == 'selfplay':
                if history_max_blocks is not None and outer>=history_max_blocks:
                    stop.update(requested=True,reason='selfplay_max_blocks_reached_before_active_time_cap')
                    break
                if len(history)==outer and (history_bytes+history_snapshot_bytes>history_limit or
                    shutil.disk_usage(run_dir).free<history_snapshot_bytes+history_reserve):
                    stop.update(requested=True,reason='selfplay_history_storage_guard')
                    break
                started = time.perf_counter()
                if len(history) == outer:
                    snapshot=_snapshot(run_dir/'history',outer,state['model'])
                    history.append(snapshot)
                    history_bytes+=snapshot['bytes']
                    history_snapshot_bytes=max(history_snapshot_bytes,snapshot['bytes'])
                if len(history) <= outer:
                    raise ValueError('Incomplete historical snapshot bank')
                index = outer//2
                if cached_history_index != index:
                    cached_history_model = _history_model(run_dir/'history',history[index],state['model'])
                    cached_history_index = index
                historical = cached_history_model
                counters['checkpoint_seconds'] += time.perf_counter()-started
            started = time.perf_counter()
            batch,ids,regime_key,noise_key,sim_metrics = simulator(state,tables)
            _sync((batch,ids,sim_metrics))
            simulation_elapsed = time.perf_counter()-started
            if int(sim_metrics['simulator_nonfinite_count']):
                raise FloatingPointError('Nonfinite simulated training batch')
            if compiled_update is None:
                compiled_update = _compiled(update,(state,batch,ids,historical),counters)
            started = time.perf_counter()
            candidate,metrics = compiled_update(state,batch,ids,historical)
            _sync((candidate,metrics))
            elapsed = time.perf_counter()-started
            if not bool(metrics['state_finite']) or int(metrics['learner_gradient_nonfinite_count']) or int(metrics['learner_loss_nonfinite_count']):
                _append(run_dir/'train_metrics.jsonl',_event(base,counters,event_type='numerical_failure',**_plain(metrics)))
                raise FloatingPointError('Nonfinite learner/generator update')
            state = dict(candidate,regime_key=regime_key,noise_key=noise_key)
            counters['active_training_seconds'] += simulation_elapsed+elapsed
            counters['simulation_seconds'] += simulation_elapsed
            counters['integrated_update_seconds'] += elapsed
            counters['teacher_seconds'] += elapsed if adaptive else 0.
            counters['learner_steps'] += nsteps
            counters['outer_rounds'] += 1
            counters['synthetic_path_count'] += nsteps*batch_size
            if method.startswith(('dpg_','soar_')):
                counters['feedback_episode_evaluation_count'] += 4*hard['spot'].shape[0] + (4*easy['spot'].shape[0] if easy is not None else 0)
            if counters['active_training_seconds']-last_log_active >= 30 or outer == 0:
                record = _plain(metrics)
                record['generator_probabilities'] = record.pop('probabilities')
                if not adaptive:
                    record.update(teacher_grad_norm=None,reward_mean=None,teacher_status='not_applicable')
                _append(run_dir/'train_metrics.jsonl',_event(base,counters,event_type='training',
                    checkpoint_id=last_checkpoint,teacher_timing_scope='integrated_student_and_teacher' if adaptive else 'not_applicable',
                    **record,**_plain(sim_metrics)))
                last_log_active = counters['active_training_seconds']
                status('running')
            monitor_due = counters['active_training_seconds']-counters['last_monitor_active_seconds'] >= config['monitor_interval_seconds']
            select_due = counters['active_training_seconds']-counters['last_selection_active_seconds'] >= config['selection_interval_seconds']
            save_due = counters['active_training_seconds']-last_saved_active >= config['checkpoint_interval_seconds']
            if monitor_due or select_due or save_due:
                save()
                last_saved_active = counters['active_training_seconds']
            if monitor_due:
                _evaluate(root,run_dir,state['model'],config,manifest,scales,base,counters,last_checkpoint,'monitor',evaluator)
                counters['last_monitor_active_seconds'] = counters['active_training_seconds']
            if select_due:
                selected = _evaluate(root,run_dir,state['model'],config,manifest,scales,base,counters,last_checkpoint,'selection',evaluator)
                counters['last_selection_active_seconds'] = counters['active_training_seconds']
                if best_score is None or selected['balanced_objective'] < best_score:
                    best_score = selected['balanced_objective']
                    atomic_json(checkpoint_dir/'best.json',json.loads((checkpoint_dir/'latest.json').read_text()))
        save()
        terminal_id = last_checkpoint
        terminal_metrics = _plain(metrics) if metrics is not None else {}
        terminal_metrics['generator_probabilities'] = terminal_metrics.pop('probabilities',
            _plain(probabilities(state['generator'].logits,epsilon=config['epsilon']) if adaptive else fixed_q))
        _append(run_dir/'train_metrics.jsonl',_event(base,counters,event_type='training_terminal',
            checkpoint_id=terminal_id,**terminal_metrics))
        _evaluate(root,run_dir,state['model'],config,manifest,scales,base,counters,terminal_id,'monitor',evaluator)
        selected = _evaluate(root,run_dir,state['model'],config,manifest,scales,base,counters,terminal_id,'selection',evaluator)
        if best_score is None or selected['balanced_objective'] < best_score:
            best_score = selected['balanced_objective']
            atomic_json(checkpoint_dir/'best.json',json.loads((checkpoint_dir/'latest.json').read_text()))
        atomic_json(checkpoint_dir/'final.json',json.loads((checkpoint_dir/'latest.json').read_text()))
        # Metadata-only continuation counters include terminal evals without creating
        # an unevaluated duplicate numerical checkpoint.
        atomic_json(run_dir/'continuation.json',dict(checkpoint_id=terminal_id,counters=_plain(counters),selection_score=best_score))
        return status('stopped' if stop['requested'] else 'completed',
                      stop_reason=stop.get('reason'),selfplay_history_bytes=history_bytes if method=='selfplay' else None)
    except BaseException as error:
        failure = traceback.format_exc()
        (run_dir/'failure.log').write_text(failure)
        try:
            save()
        except BaseException:
            failure += '\nCheckpoint saving also failed:\n'+traceback.format_exc()
            (run_dir/'failure.log').write_text(failure)
        _append(run_dir/'train_metrics.jsonl',_event(base,counters,event_type='failure',
                checkpoint_id=last_checkpoint,error=repr(error),traceback=failure))
        # A failed candidate does not invalidate the last accepted learner.
        # Attempt its mandatory terminal evaluation before recording missing data.
        for role in ('monitor','selection'):
            try:
                if last_checkpoint is None:
                    raise ValueError('No valid checkpoint available')
                _evaluate(root,run_dir,state['model'],config,manifest,scales,base,counters,
                          last_checkpoint,role)
            except BaseException as evaluation_error:
                for domain in DOMAINS:
                    _append(run_dir/'evaluation_metrics.jsonl',_event(base,counters,event_type='evaluation_unavailable',
                        checkpoint_id=last_checkpoint,domain=domain,split_role=role,raw_mse=None,
                        scaled_mse=None,status='unavailable',reason=repr(evaluation_error)))
        return status('failed',error=repr(error),traceback_path=str((run_dir/'failure.log').relative_to(root)))
    finally:
        for signum,handler in old_handlers.items():
            signal.signal(signum,handler)


def train_one(root, config, active_seconds, resume=True):
    """Public cumulative-cap entrypoint, archiving even pre-initialization failures."""
    try:
        return _train_one(root,config,active_seconds,resume)
    except BaseException as error:
        root = Path(root).resolve()
        run_id = config.get('run_id',config.get('method','invalid_trial'))
        if Path(run_id).name != run_id:
            raise
        run_dir = root/'runs'/run_id
        run_dir.mkdir(parents=True,exist_ok=True)
        failure = traceback.format_exc()
        (run_dir/'failure.log').write_text(failure)
        result = dict(run_id=run_id,method=config.get('method'),status='failed',timestamp=_utc(),
                      error=repr(error),last_checkpoint_id=None,active_training_seconds=0.,
                      traceback_path=str((run_dir/'failure.log').relative_to(root)))
        atomic_json(run_dir/'status.json',result)
        _manifest(root,config,result,'run_terminal')
        _append(run_dir/'train_metrics.jsonl',dict(result,event_type='failure',traceback=failure))
        for domain in DOMAINS:
            _append(run_dir/'evaluation_metrics.jsonl',dict(result,event_type='evaluation_unavailable',
                domain=domain,split_role='monitor',raw_mse=None,scaled_mse=None,reason=repr(error)))
        return result


def _evaluate_checkpoint(root, run_id, checkpoint_path, role='final_test'):
    """Post-selection reporting only; test values never update selection pointers."""
    root = Path(root).resolve()
    devices = configure(root)
    run_dir = root/'runs'/run_id
    config = _config(json.loads((run_dir/'config.json').read_text()))
    manifest,scales,hashes = _inputs(root)
    state,_,_ = _skeleton(config)
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = run_dir/'checkpoints'/path
    state,metadata = load_checkpoint(path,state,expected_hashes=hashes)
    counters = _initial_counters()
    counters.update(metadata['counters'])
    base = _base_metadata(root,config,hashes,devices)
    result = _evaluate(root,run_dir,state['model'],config,manifest,scales,base,counters,
                      metadata['checkpoint_id'],role)
    atomic_json(run_dir/f'{role}_{metadata["checkpoint_id"]}.json',_plain(result))
    return result


def evaluate_checkpoint(root, run_id, checkpoint_path, role='final_test'):
    """Archive explicit per-domain failures without concealing valid prior events."""
    try:
        return _evaluate_checkpoint(root,run_id,checkpoint_path,role)
    except Exception as error:
        root = Path(root).resolve()
        run_dir = root/'runs'/run_id
        run_dir.mkdir(parents=True,exist_ok=True)
        path = Path(checkpoint_path)
        if not path.is_absolute():
            path = run_dir/'checkpoints'/path
        checkpoint_id = path.name
        try:
            if path.suffix=='.json':
                checkpoint_id = json.loads(path.read_text())['checkpoint_id']
            elif (path/'metadata.json').exists():
                checkpoint_id = json.loads((path/'metadata.json').read_text())['checkpoint_id']
        except (OSError,KeyError,ValueError):
            pass
        config_path = run_dir/'config.json'
        config = json.loads(config_path.read_text()) if config_path.exists() else {'run_id':run_id}
        domains = {}
        for domain in DOMAINS:
            event = dict(event_type='evaluation_unavailable',run_id=run_id,method=config.get('method'),
                checkpoint_id=checkpoint_id,domain=domain,split_role=role,timestamp=_utc(),
                raw_mse=None,scaled_mse=None,status='unavailable',reason=repr(error),
                sample_count=None,finite_count=None)
            _append(run_dir/'evaluation_metrics.jsonl',event)
            domains[domain] = event
        result = dict(status='failed',error=repr(error),domains=domains,
                      balanced_objective=None,easy_hard_raw_mean=None)
        atomic_json(run_dir/f'{role}_{checkpoint_id}.json',result)
        (run_dir/f'{role}_{checkpoint_id}.failure.log').write_text(traceback.format_exc())
        return result
