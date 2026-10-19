"""Full production-shape actual trainer execution and completed-run replay."""
import json
import os
from pathlib import Path

import equinox as eqx
import jax
import numpy as np

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
    arms = ['baseline']
    def save_report():
        (out/'monitor-audit.partial.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    def equal_trees(first,second):
        return all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in
                   zip(jax.tree.leaves(first),jax.tree.leaves(second)) if eqx.is_array(a))
    for arm in arms:
        folder=out/'trials'/arm
        os.environ['JOB_SELFPLAY_HISTORY_DIR']=str(out/'history'/arm)
        run(adapter,make_model,folder,design,arm,.001)
        before=load(folder/'latest.pkl')
        first=json.loads((folder/'selection.json').read_text())
        if first['learner_updates'] <= 0:
            raise AssertionError('No actual learner update: '+arm)
        for boundary in (0,1):
            payload=load(folder/f'active-{boundary}.pkl')
            if not all(name in payload for name in ('student','generator','baseline','key_data','active_seconds','outer_round','completed','config_hash')):
                raise AssertionError('Incomplete boundary checkpoint: '+arm)
        run(adapter,make_model,folder,design,arm,.001)
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
    (out/'monitor-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
