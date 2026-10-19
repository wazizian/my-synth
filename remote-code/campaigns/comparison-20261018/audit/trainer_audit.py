"""Full production-shape actual trainer execution and completed-run replay."""
import json
import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import adaptive as algorithms
from credit_adapter import CreditAdapter
from credit_hedging.learned_policy import TargetHedger
from train_adaptive import run, load


def main():
    jax.config.update('jax_enable_x64', True)
    assert jax.default_backend() == 'gpu'
    out = Path(os.environ['JOB_OUTPUT_DIR'])
    source = Path(__file__).parent
    design = json.loads((source/'design.json').read_text())
    design.update(checkpoint_active_seconds=[0,1],selection_count=128,
                  adam_eps_root=1e-16,monitor_count=128,monitor_seed=1020261017)
    adapter = CreditAdapter(source/'credit-source',3)
    make_model = lambda seed: TargetHedger(jax.random.key(seed))
    report = dict(status='running',adam_eps_root=1e-16,arms={})
    arms = ['baseline','dpg_hard','selfplay','easy_reference','hard_reference',
            'soar_hard','dpg_mixed','soar_mixed']
    def save_report():
        (out/'trainer-audit.partial.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    def equal_trees(first,second):
        return all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in
                   zip(jax.tree.leaves(first),jax.tree.leaves(second)) if eqx.is_array(a))
    for arm in arms:
        folder=out/'trials'/arm
        trial_design=dict(design)
        if arm=='selfplay':
            trial_design['checkpoint_active_seconds']=[0,35]
        os.environ['JOB_SELFPLAY_HISTORY_DIR']=str(out/'history'/arm)
        run(adapter,make_model,folder,trial_design,arm,.001)
        before=load(folder/'latest.pkl')
        first=json.loads((folder/'selection.json').read_text())
        if not (folder/f"monitor-{trial_design['checkpoint_active_seconds'][-1]}.json").exists():
            raise AssertionError('Independent monitor stream did not run: '+arm)
        if arm=='selfplay':
            if before['outer_round']<3:
                raise AssertionError('Selfplay must exercise at least three outer rounds')
            teacher_applied=[json.loads(row).get('generator_update_applied') for row in (folder/'training.jsonl').read_text().splitlines()]
            if not any(teacher_applied):
                raise AssertionError('Selfplay never applied a nonzero-movement teacher update')
            @eqx.filter_jit
            def identity(student,historical,market):
                batches,_=market.draw(jax.random.key(901),jnp.ones(3)/3,8,160)
                model,state=student
                rewards,_=algorithms.selfplay_rewards(model,historical,state,batches,market.losses,.001,1,adam_eps_root=1e-16)
                params,static=eqx.partition(model,eqx.is_inexact_array)
                old=jax.tree.unflatten(jax.tree.structure(params),jax.tree.leaves(eqx.filter(historical,eqx.is_inexact_array)))
                adam=algorithms.find_adam_state(state)
                direction=jax.tree.map(lambda h,p,v:(h-p)*.001/(jnp.sqrt(v/(1.-.999**adam.count)+1e-16)+1e-8),old,params,adam.nu)
                batch=jax.tree.map(lambda x:x[0],batches)
                losses,jvp=eqx.filter_jvp(lambda p:market.losses(eqx.combine(p,static),batch),(params,),(direction,))
                expected=jnp.abs(160*jax.nn.softmax(10*losses)*jvp)
                return jnp.max(jnp.abs(expected-rewards[0]))
            identity_error=float(identity(before['student'],make_model(design['seed']),adapter))
            if identity_error>1e-12:
                raise AssertionError('Actual Adam eps_root selfplay identity mismatch')
            report['selfplay_actual_adam_identity_error']=identity_error
        if first['learner_updates'] <= 0:
            raise AssertionError('No actual learner update: '+arm)
        for boundary in trial_design['checkpoint_active_seconds']:
            payload=load(folder/f'active-{boundary}.pkl')
            if not all(name in payload for name in ('student','generator','baseline','key_data','active_seconds','outer_round','completed','config_hash')):
                raise AssertionError('Incomplete boundary checkpoint: '+arm)
        run(adapter,make_model,folder,trial_design,arm,.001)
        after=load(folder/'latest.pkl')
        for name in ('student','generator','baseline','key_data'):
            if not equal_trees(before[name],after[name]):
                raise AssertionError('Completed replay changed '+arm+' '+name)
        if before['active_seconds'] != after['active_seconds'] or before['outer_round'] != after['outer_round']:
            raise AssertionError('Completed replay consumed training budget: '+arm)
        replay=json.loads((folder/'resume.json').read_text())
        if replay['replay_error'] != 0.:
            raise AssertionError('Nonzero selection replay: '+arm)
        report['arms'][arm]=dict(status='passed',learner_updates=first['learner_updates'],
                                 active_seconds=first['active_seconds'],criterion=first['criterion'],
                                 selection_replay_error=replay['replay_error'],
                                 completed_replay_preserves_state=True)
        save_report()
        print(json.dumps({arm:report['arms'][arm]}),flush=True)
    report['status']='passed'
    save_report()
    (out/'trainer-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
