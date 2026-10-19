"""GPU-only algorithm audit and full credit-shape feasibility profile."""
import json
import os
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import adaptive as a
from credit_adapter import CreditAdapter, terminal_losses
from credit_hedging.environment import rollout
from credit_hedging.curve_market import simulate_curve_paths
from credit_hedging.learned_policy import TargetHedger


def main():
    jax.config.update('jax_enable_x64', True)
    assert jax.default_backend() == 'gpu'
    out = Path(os.environ['JOB_OUTPUT_DIR'])
    report = {'backend': jax.default_backend(), 'checks': {}, 'profiles': {}}

    def save():
        p = out / 'audit.partial.json'
        tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False))
        tmp.replace(p)

    def check(name, value):
        report['checks'][name] = bool(value)
        save()
        if not value:
            raise AssertionError(name)

    def finite(tree):
        return all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(tree)
                   if eqx.is_array(x))

    def error(tree1, tree2):
        return max(float(np.max(np.abs(np.asarray(x) - np.asarray(y))))
                   for x, y in zip(jax.tree.leaves(tree1), jax.tree.leaves(tree2))
                   if eqx.is_array(x) and x.size)

    # Nonlinear empirical entropic toy with nonzero carried Adam moments.
    m = eqx.nn.Linear(2, 1, key=jax.random.key(1), dtype=jnp.float64)
    opt = optax.chain(optax.clip_by_global_norm(1.), optax.adam(.03))
    losses = lambda model, batch: (jax.vmap(model)(batch).squeeze(-1) - .4) ** 2
    batches = .2 * jax.random.normal(jax.random.key(2), (3, 5, 2))
    weights = jnp.ones((3, 5))
    block = lambda student, batch, w: a.weighted_block(student, batch, w, losses, opt)
    student, _ = block((m, opt.init(eqx.filter(m, eqx.is_inexact_array))), batches, weights)
    target = lambda model: a.entropic(losses(model, batches[0] + .13))
    dpg = eqx.filter_jit(lambda s, b: a.dpg_attributions(s, b, block, target, weights.shape))
    post, vals, rewards, value = dpg(student, batches)
    jax.block_until_ready((post, vals, rewards, value))
    direct, _ = eqx.filter_jit(block)(student, batches, weights)
    check('toy_dpg_finite', finite((post, rewards)))
    check('toy_actual_post_state_equivalence', error(post, direct) < 1e-12)
    check('toy_per_minibatch_weight_derivative_sum', float(jnp.max(jnp.abs(rewards.sum(1)))) < 1e-10)
    step = 1e-4
    direction = jnp.zeros_like(weights).at[1, 2].set(1.)
    plus = target(block(student, batches, weights + step * direction)[0][0])
    minus = target(block(student, batches, weights - step * direction)[0][0])
    fd = float((plus - minus) / (2 * step))
    report['toy_dpg_finite_difference'] = {'finite_difference': fd, 'autodiff': -float(rewards[1, 2])}
    check('toy_finite_difference_nondegenerate', abs(fd) > 1e-8)
    check('toy_nonlinear_finite_difference', abs(fd + float(rewards[1, 2])) < 1e-7)
    prior = jnp.ones(3) / 3
    genopt = optax.adam(.01)
    gen = a.init_generator(prior, genopt)
    ids = jnp.arange(15).reshape(3, 5) % 3
    gen2, gm = a.generator_step(gen, genopt, ids, rewards, prior)
    check('generator_finite_floor', finite((gen2, gm)) and bool(jnp.all(gm['probabilities'] >= .05 * prior)))
    adv, ema, sm = a.soar_reward(student[0], post[0], target, jnp.array(.002))
    check('soar_baseline_timing', abs(float(adv - (sm['improvement'] - .002))) < 1e-12 and abs(float(ema - (.9 * .002 + .1 * sm['improvement']))) < 1e-12)
    ir, im = a.selfplay_rewards(student[0], m, student[1], batches, losses, .001, 1)
    check('selfplay_finite', finite((ir, im)))
    p, static = eqx.partition(student[0], eqx.is_inexact_array)
    old = eqx.filter(m, eqx.is_inexact_array)
    adam = a.find_adam_state(student[1])
    direction = jax.tree.map(lambda h, x, v: (h-x)*.001/(jnp.sqrt(v/(1.-.999**adam.count))+1e-8), old, p, adam.nu)
    toy_values, toy_jvp = eqx.filter_jvp(lambda candidate: losses(eqx.combine(candidate, static), batches[0]), (p,), (direction,))
    expected_rewards = jnp.abs(5*jax.nn.softmax(10.*toy_values)*toy_jvp)
    check('selfplay_entropic_contribution_identity', float(jnp.max(jnp.abs(ir[0]-expected_rewards))) < 1e-12)
    unchanged, _ = a.generator_step(gen, genopt, ids, ir, prior, selfplay=True, apply_update=False)
    check('selfplay_roundzero_preserves_optimizer', error(gen, unchanged) == 0.)
    cp = out / 'toy-checkpoint.eqx'
    eqx.tree_serialise_leaves(cp, (student, gen, jax.random.key_data(jax.random.key(8)), jnp.array(.002)))
    restored = eqx.tree_deserialise_leaves(cp, (student, gen, jax.random.key_data(jax.random.key(8)), jnp.array(.002)))
    replay = dpg(restored[0], batches)
    check('checkpoint_replay', error((post, vals, rewards, value), replay) == 0.)

    model = TargetHedger(jax.random.key(20261017))
    def ledger_equivalence(domain, config, tables):
        paths = eqx.filter_jit(simulate_curve_paths)(jax.random.key(71), tables, config, 8)
        def compare(policy, batch, full, cfg):
            original = rollout(policy, batch, full, cfg)['loss']
            new = terminal_losses(policy, batch, {n:full[n] for n in ('protection','annuity','par_spread')}, full['bond'][cfg.bond_days], cfg)
            return original, new
        original, new = eqx.filter_jit(compare)(model, paths, tables, config)
        check('production_terminal_ledger_'+domain, float(jnp.max(jnp.abs(original-new))) < 1e-12)
    adapter = CreditAdapter(Path(__file__).parent / 'credit-source', 3, ledger_equivalence)
    tick = time.monotonic()
    paths, ids = eqx.filter_jit(lambda market,key,q:market.draw(key,q,8,160))(adapter,jax.random.key(72),prior)
    jax.block_until_ready((paths, ids))
    report['profiles']['sampling_compile_seconds'] = time.monotonic() - tick
    feedback = eqx.filter_jit(lambda market,key:market.domain(key,'hard',160))(adapter,jax.random.key(73))
    save()

    for eps_root in (0., 1e-16):
        label = f'eps_root_{eps_root:g}'
        optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(.001, eps_root=eps_root))
        initial = (model, optimizer.init(eqx.filter(model, eqx.is_inexact_array)))
        def block_with_market(s,b,w,market):
            return a.weighted_block(s,b,w,market.losses,optimizer)
        def dpg_with_market(s,b,market,fb):
            fullblock=lambda ss,bb,ww:block_with_market(ss,bb,ww,market)
            feedback_fn=lambda candidate:a.entropic(market.losses(candidate,fb))
            return a.dpg_attributions(s,b,fullblock,feedback_fn,(8,160))
        full_dpg = eqx.filter_jit(dpg_with_market)
        tick = time.monotonic()
        result = full_dpg(initial, paths,adapter,feedback)
        jax.block_until_ready(result)
        profile = {'compile_and_first_seconds': time.monotonic() - tick, 'finite': finite(result)}
        report['profiles'][label] = profile
        save()
        print(json.dumps({label: profile}), flush=True)
        if not profile['finite']:
            continue
        tick = time.monotonic()
        repeated = full_dpg(initial, paths,adapter,feedback)
        jax.block_until_ready(repeated)
        profile['warm_block_seconds'] = time.monotonic() - tick
        profile['gpu_memory'] = jax.devices()[0].memory_stats()
        actual = eqx.filter_jit(block_with_market)(initial, paths, jnp.ones((8, 160)),adapter)
        jax.block_until_ready(actual)
        check(label + '_actual_post_state', error(result[0], actual[0]) < 1e-12)
        check(label + '_weight_derivative_sum', float(jnp.max(jnp.abs(result[2].sum(1)))) < 1e-8)
        teacher, logs = a.generator_step(gen, genopt, ids, result[2], prior)
        check(label + '_dpg_teacher_finite', finite((teacher, logs)))
        adv, ema, logs = eqx.filter_jit(lambda pre,post,market,fb:a.soar_reward(pre,post,lambda mm:a.entropic(market.losses(mm,fb)),jnp.array(0.)))(model,result[0][0],adapter,feedback)
        teacher, logs2 = a.generator_step(gen, genopt, ids, jnp.full(ids.shape, adv), prior)
        check(label + '_soar_teacher_finite', finite((teacher, ema, logs, logs2)))
        intrinsic = eqx.filter_jit(lambda m,h,s,b,market:a.selfplay_rewards(m,h,s,b,market.losses,.001,1))
        rr, mm = intrinsic(result[0][0], model, result[0][1], paths,adapter)
        jax.block_until_ready((rr, mm))
        teacher, logs = a.generator_step(gen, genopt, ids, rr, prior, selfplay=True)
        check(label + '_selfplay_teacher_finite', finite((teacher, logs)))
        from train_adaptive import save as trainer_save, load as trainer_load
        payload=dict(student=result[0],generator=teacher,baseline=ema,key_data=jax.random.key_data(jax.random.key(1)),active_seconds=1.2,outer_round=1,config_hash='audit',completed=[0],best='checkpoint.pkl')
        trainer_save(out/'actual-checkpoint.pkl',payload)
        restored=trainer_load(out/'actual-checkpoint.pkl')
        check(label+'_actual_trainer_checkpoint',error(payload['student'],restored['student']) == 0. and restored['config_hash']=='audit')
        replay=full_dpg(restored['student'],paths,adapter,feedback)
        same=full_dpg(result[0],paths,adapter,feedback)
        jax.block_until_ready((replay,same))
        check(label+'_actual_trainer_replay',error(replay,same)==0.)
        save()
        if eps_root == 0.:
            break
    report['status'] = 'passed' if any(v.get('finite') for v in report['profiles'].values() if isinstance(v, dict)) else 'failed_full_shape'
    save()
    (out / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
