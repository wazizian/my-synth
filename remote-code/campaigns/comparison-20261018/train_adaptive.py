"""GPU worker training with resumable persistent learner and teacher state."""
from pathlib import Path
import gc
import hashlib
import json
import os
import pickle
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from adaptive import (detached, dpg_attributions, entropic, generator_step,
                      init_generator, make_target, probabilities,
                      selfplay_rewards, soar_reward, weighted_block)


def write(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def save(path, payload):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        pickle.dump(jax.device_get(payload), stream, protocol=5)
    os.replace(tmp, path)


def load(path):
    with path.open('rb') as stream:
        payload = pickle.load(stream)
    return jax.tree.map(lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x,
                        payload)


def canonical_model(loaded, template):
    """Restore model values using the current architecture's static metadata.

    Pickle can reconstruct distinct Equinox Missing sentinels. Canonicalizing
    before JIT prevents false static-argument cache misses for every snapshot.
    """
    if type(loaded) is not type(template):
        raise RuntimeError('Historical model architecture changed')
    leaves = jax.tree.leaves(loaded)
    expected = jax.tree.leaves(template)
    if len(leaves) != len(expected) or any(
        x.shape != y.shape or x.dtype != y.dtype for x, y in zip(leaves, expected)
    ):
        raise RuntimeError('Historical model parameter layout changed')
    return jax.tree.unflatten(jax.tree.structure(template), leaves)


def run(adapter, make_model, output, design, arm, lr):
    """One predeclared LR trial; callers freeze choices before final evaluation."""
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Numerical work requires GPU')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    # Pre-block snapshots support exact self-play continuation on the worker.
    # Keep this growing recovery cache outside incrementally exported outputs;
    # latest/selected complete checkpoints and all metrics remain exported.
    history_dir = Path(os.environ.get('JOB_SELFPLAY_HISTORY_DIR',
                                     str(out.parent.parent / 'selfplay-history' / arm / f'lr-{lr:g}')))
    history_dir.mkdir(parents=True, exist_ok=True)
    block_size = design['block_size']
    batch_size = design['batch_size']
    gamma = design['gamma']
    prior = jnp.array(design['generator_prior'])
    optimizer = optax.chain(optax.clip_by_global_norm(1.),
                           optax.adam(lr, eps_root=design.get('adam_eps_root', 0.)))
    teacher_optimizer = optax.adam(design['teacher_lr'])
    model = make_model(design['seed'])
    student = (model, optimizer.init(eqx.filter(model, eqx.is_inexact_array)))
    generator = init_generator(prior, teacher_optimizer)
    key = jax.random.key(design['training_seed'])
    baseline = jnp.array(0.)
    active = 0.
    outer = 0
    best = None
    best_score = float('inf')
    completed = []
    compile_seconds = 0.
    selection_seconds = 0.
    monitor_seconds = 0.
    history_bytes = 0
    intrinsic = arm == 'selfplay'
    fixed_domain = {'baseline': design['domains'][1], 'easy_reference': design['domains'][0],
                    'hard_reference': design['domains'][2]}.get(arm)
    is_dpg = arm.startswith('dpg')
    mixed = arm.endswith('mixed')
    domains = design['domains']
    recovery_marker = Path(os.environ.get('JOB_OUTPUT_DIR', str(out.parent))) / 'recovery-requirements.json'
    if intrinsic:
        write(recovery_marker, dict(external_history_required_until_complete=True,
              external_history_path=str(history_dir), arm=arm, lr=lr,
              rule='Do not destroy incomplete selfplay worker without its history cache'))
    config_hash = hashlib.sha256(json.dumps(dict(design=design, arm=arm, lr=lr),
                                           sort_keys=True).encode()).hexdigest()
    latest = out / 'latest.pkl'
    if latest.exists():
        payload = load(latest)
        if payload['config_hash'] != config_hash:
            raise RuntimeError('Resume configuration changed')
        student, generator, baseline = payload['student'], payload['generator'], payload['baseline']
        key = jax.random.wrap_key_data(payload['key_data'])
        active, outer = payload['active_seconds'], payload['outer_round']
        best, best_score, completed = payload['best'], payload['best_score'], payload['completed']
        compile_seconds = payload['compile_seconds']
        selection_seconds = payload['selection_seconds']
        monitor_seconds = payload.get('monitor_seconds', 0.)
        history_bytes = sum(p.stat().st_size for p in history_dir.glob('*.pkl'))
        if intrinsic and outer and not (history_dir / f'{outer // 2:08d}.pkl').exists():
            raise RuntimeError('Selfplay recovery history missing; do not reset learner')

    @eqx.filter_jit
    def draw(k, q, market):
        if fixed_domain is not None:
            batches = jax.vmap(lambda sub: market.domain(sub, fixed_domain, batch_size))(
                jax.random.split(k, block_size))
            return batches, jnp.full((block_size, batch_size), domains.index(fixed_domain))
        return market.draw(k, q, block_size, batch_size)

    @eqx.filter_jit
    def feedback(k, market):
        hard = market.domain(jax.random.fold_in(k, 2), domains[2], batch_size)
        easy = market.domain(jax.random.fold_in(k, 0), domains[0], batch_size) if mixed else None
        return hard, easy

    @eqx.filter_jit
    def step(s, g, ema, historical, batches, ids, hard, easy, round_index, market):
        def block(student_state, data, weights):
            return weighted_block(student_state, data, weights, market.losses,
                                  optimizer, gamma)
        if fixed_domain is not None:
            post, learner_values = block(s, batches, jnp.ones(ids.shape))
            return detached(post), g, ema, dict(learner_values=learner_values)
        target = None if intrinsic else make_target(market.losses, hard, easy,
                                                     tuple(design['scales']), gamma)
        if is_dpg:
            post, learner_values, rewards, target_after = dpg_attributions(
                s, batches, block, target, ids.shape)
            teacher_metrics = dict(target_after=target_after)
        else:
            if intrinsic:
                rewards, teacher_metrics = selfplay_rewards(
                    s[0], historical, s[1], batches, market.losses, lr,
                    round_index, gamma, adam_eps_root=design.get('adam_eps_root', 0.))
            post, learner_values = block(s, batches, jnp.ones(ids.shape))
            if not intrinsic:
                advantage, ema, teacher_metrics = soar_reward(s[0], post[0], target, ema)
                rewards = jnp.full(ids.shape, advantage)
        usable = ((round_index > 0) & (teacher_metrics['parameter_movement_norm'] > 0)
                  if intrinsic else True)
        g, gm = generator_step(g, teacher_optimizer, ids, rewards, prior,
                               selfplay=intrinsic, apply_update=usable,
                               epsilon=design['epsilon'],
                               kl_coefficient=design['kl_coefficient'])
        return detached(post), g, ema, dict(learner_values=learner_values,
                                            **teacher_metrics, **gm)

    @eqx.filter_jit
    def dev_losses(m, k, domain, market):
        return market.losses(m, market.domain(k, domain, 128))

    def selection(m, seed=None, count=None):
        result = {}
        seed = design['selection_seed'] if seed is None else seed
        count = design['selection_count'] if count is None else count
        for index, d in enumerate(domains):
            rows = []
            for start in range(0, count, 128):
                k = jax.random.fold_in(jax.random.key(seed),
                                       index * 100000 + start)
                rows.append(np.asarray(dev_losses(m, k, d, adapter)))
            losses = np.concatenate(rows)
            if not np.isfinite(losses).all():
                raise RuntimeError('Nonfinite development loss')
            result[d] = float(entropic(jnp.asarray(losses), gamma))
        result['balanced'] = .5 * (result[domains[0]] / design['scales'][0]
                                    + result[domains[2]] / design['scales'][1])
        return result

    def checkpoint(boundary_path=None):
        payload = dict(student=student, generator=generator, baseline=baseline,
                          key_data=jax.random.key_data(key), active_seconds=active,
                          outer_round=outer, best=best, best_score=best_score,
                          completed=completed, compile_seconds=compile_seconds,
                          selection_seconds=selection_seconds, monitor_seconds=monitor_seconds,
                          config_hash=config_hash)
        save(latest, payload)
        if boundary_path is not None:
            save(boundary_path, payload)
        write(out / 'progress.json', dict(active_seconds=active, outer_round=outer,
              learner_updates=outer * block_size, training_paths=outer * block_size * batch_size,
              teacher_feedback_paths=0 if intrinsic or fixed_domain else outer * batch_size * (2 if mixed else 1),
              probabilities=(np.eye(3)[domains.index(fixed_domain)].tolist() if fixed_domain else
                             np.asarray(probabilities(generator.logits, prior)).tolist()),
              completed_boundaries=completed, best_score=best_score if np.isfinite(best_score) else None,
              config_hash=config_hash))

    # Compile on a disjoint namespace; retain neither update nor consumed RNG.
    tick = time.monotonic()
    warm_batches, warm_ids = draw(jax.random.key(110261017), prior, adapter)
    warm_hard, warm_easy = ((None, None) if intrinsic or fixed_domain else
                           feedback(jax.random.key(120261017), adapter))
    warm = step(student, generator, baseline, student[0], warm_batches, warm_ids,
                warm_hard, warm_easy, jnp.array(1), adapter)
    jax.block_until_ready(warm)
    compile_seconds += time.monotonic() - tick
    del warm, warm_batches
    gc.collect()
    if latest.exists():
        replay = selection(load(out / best)['student'][0]) if best is not None else None
        if replay is not None and abs(replay[fixed_domain or 'balanced'] - best_score) > 1e-10:
            raise RuntimeError('Resume development replay mismatch')
        write(out / 'resume.json', dict(active_seconds=active, outer_round=outer,
                                        replay_error=0. if replay is not None else None))
    last_save = time.monotonic()
    for boundary in design['checkpoint_active_seconds']:
        if boundary in completed:
            continue
        with (out / 'training.jsonl').open('a') as stream:
            while active < boundary:
                tick = time.monotonic()
                key, draw_key = jax.random.split(key)
                q = probabilities(generator.logits, prior, design['epsilon'])
                batches, ids = draw(draw_key, q, adapter)
                hard, easy = ((None, None) if intrinsic or fixed_domain else feedback(
                    jax.random.fold_in(jax.random.key(design['feedback_seed']), outer), adapter))
                if intrinsic:
                    snapshot = history_dir / f'{outer:08d}.pkl'
                    old_size = snapshot.stat().st_size if snapshot.exists() else 0
                    save(snapshot, student[0])
                    history_bytes += snapshot.stat().st_size - old_size
                    if history_bytes > design['history_limit_bytes']:
                        raise RuntimeError('Frozen selfplay history storage bound exceeded')
                    historical = canonical_model(
                        load(history_dir / f'{outer // 2:08d}.pkl'), student[0])
                else:
                    historical = student[0]
                student, generator, baseline, logs = step(
                    student, generator, baseline, historical, batches, ids, hard, easy,
                    jnp.array(outer), adapter)
                jax.block_until_ready(logs)
                active += time.monotonic() - tick
                outer += 1
                if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(logs)):
                    raise RuntimeError('Nonfinite adaptive update')
                record = {k: np.asarray(v).tolist() for k, v in logs.items()}
                record.update(active_seconds=active, outer_round=outer,
                              learner_updates=outer * block_size)
                stream.write(json.dumps(record, allow_nan=False) + '\n')
                stream.flush()
                if time.monotonic() - last_save >= 60:
                    checkpoint()
                    last_save = time.monotonic()
        tick = time.monotonic()
        scores = selection(student[0])
        selection_seconds += time.monotonic() - tick
        name = f'active-{boundary}.pkl'
        score = scores[fixed_domain or 'balanced']
        if score < best_score:
            best_score, best = score, name
        completed.append(boundary)
        write(out / f'selection-{boundary}.json', scores)
        if design.get('monitor_count', 0):
            tick = time.monotonic()
            monitor = selection(student[0], design['monitor_seed'], design['monitor_count'])
            monitor_seconds += time.monotonic() - tick
            write(out / f'monitor-{boundary}.json', monitor)
        checkpoint(out / name)
    write(out / 'selection.json', dict(arm=arm, lr=lr, checkpoint=best,
          selection_score=best_score,
          criterion=fixed_domain or 'fixed balanced scaled easy/hard development',
          active_seconds=active, learner_updates=outer * block_size,
          training_paths=outer * block_size * batch_size,
          compile_seconds=compile_seconds, selection_seconds=selection_seconds,
          monitor_seconds=monitor_seconds,
          config_hash=config_hash, final_test_opened=False))
    write(out / 'complete.json', dict(status='complete', final_test_opened=False))
    if intrinsic:
        write(recovery_marker, dict(external_history_required_until_complete=False, arm=arm, lr=lr,
              reason='All trial checkpoints and selection durable on worker; pre-block cache disposable'))
