"""Retained seven-arm integration checks and real self-play continuation on GPU."""
import argparse
import datetime
import json
from pathlib import Path
import time
import traceback
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--active-seconds',type=float,default=1.)
    parser.add_argument('--campaign-start',type=float,
        default=datetime.datetime(2026,10,17,21,54,tzinfo=datetime.timezone.utc).timestamp())
    parser.add_argument('--deadline',type=float,default=None)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    from src.runtime import configure
    configure(root)
    import jax
    import numpy as np
    from src.checkpoint import atomic_json,load_checkpoint
    from src.protocol import ARMS
    from src.runner import train_one,_skeleton
    from src.teachers import find_adam_state

    deadline = args.deadline or args.campaign_start+8*3600-1800
    report_path = root/'results/integrated_smoke.json'
    report = dict(passed=False,started_at=time.time(),checks={},runs={},failures=[])
    def publish():
        atomic_json(report_path,report)
    def check(name,condition,details=None):
        report['checks'][name] = dict(passed=bool(condition),details=details)
        if not condition:
            report['failures'].append(name)
        publish()

    publish()
    for method in ARMS:
        config = dict(method=method,run_id='smoke_'+method,purpose='integration_smoke',
                      batch_size=8,block_steps=2,hidden_size=8,
                      campaign_start=args.campaign_start,deadline=deadline)
        result = train_one(root,config,args.active_seconds,resume=True)
        report['runs'][method] = {k:result.get(k) for k in
            ('run_id','status','last_checkpoint_id','active_training_seconds','learner_steps','outer_rounds','error')}
        check(method+'_completed',result['status']=='completed',result.get('error'))
        if result['status'] != 'completed':
            continue
        run_dir = root/'runs'/config['run_id']
        events = [json.loads(line) for line in (run_dir/'train_metrics.jsonl').read_text().splitlines()]
        training = [x for x in events if x['event_type'] in ('training','training_terminal')
                    and x.get('synthetic_loss') is not None]
        check(method+'_finite_learner',bool(training) and all(
            x.get('state_finite',False) and x.get('synthetic_loss') is not None
            and np.isfinite(x['synthetic_loss']) for x in training))
        evaluations = [json.loads(line) for line in (run_dir/'evaluation_metrics.jsonl').read_text().splitlines()]
        for role in ('monitor','selection'):
            domain_records = [x for x in evaluations if x.get('event_type')=='evaluation' and x.get('split_role')==role]
            check(method+'_'+role+'_three_domains',
                  {x['domain'] for x in domain_records}=={'base','easy','hard'} and
                  all(x.get('status')=='ok' and x.get('raw_mse') is not None for x in domain_records))
        if method.startswith('dpg_'):
            check(method+'_nontrivial_attribution',any((x.get('attribution_norm') or 0)>1e-12 for x in training))
        if method.startswith('soar_'):
            check(method+'_nontrivial_improvement',any(abs(x.get('raw_improvement') or 0)>1e-12 for x in training))
        if method.startswith(('dpg_','soar_')):
            check(method+'_probability_movement',any((x.get('distribution_movement_l1') or 0)>1e-10 for x in training))
        if method.endswith('_mixed'):
            check(method+'_both_target_components',all('easy_raw_before' in x and 'hard_raw_before' in x for x in training))
        if method.endswith('_hard'):
            check(method+'_no_easy_feedback',all('easy_raw_before' not in x for x in training))

    # Actual train_one resume must consume the saved moments, keys, and history.
    try:
        run_dir = root/'runs/smoke_selfplay'
        config = json.loads((run_dir/'config.json').read_text())
        skeleton,_,_ = _skeleton(config)
        before,metadata = load_checkpoint(run_dir/'checkpoints',skeleton)
        before_round = int(before['outer_round'])
        count_before = int(find_adam_state(before['optimizer']).count)
        keys_before = np.asarray(jax.random.key_data(before['regime_key']))
        old_history = metadata['history']
        status_before = json.loads((run_dir/'status.json').read_text())
        next_cap = status_before['active_training_seconds']+args.active_seconds
        resumed = train_one(root,config,next_cap,resume=True)
        after,after_metadata = load_checkpoint(run_dir/'checkpoints',skeleton)
        check('actual_resume_completed',resumed['status']=='completed',resumed.get('error'))
        check('actual_resume_round_advances',int(after['outer_round'])>before_round)
        check('actual_resume_adam_advances',int(find_adam_state(after['optimizer']).count)>count_before)
        check('actual_resume_key_advances',not np.array_equal(keys_before,np.asarray(jax.random.key_data(after['regime_key']))))
        check('actual_resume_preserves_history',after_metadata['history'][:len(old_history)]==old_history and
              len(after_metadata['history'])>=int(after['outer_round']))
        parameter_change = sum(float(jax.numpy.sum((a-b)**2)) for a,b in
            zip(jax.tree.leaves(after['model']),jax.tree.leaves(before['model'])) if isinstance(a,jax.Array))
        check('actual_resume_learner_changes',parameter_change>0,parameter_change)
        events = [json.loads(line) for line in (run_dir/'train_metrics.jsonl').read_text().splitlines()]
        intrinsic = [x for x in events if x.get('event_type') in ('training','training_terminal')]
        check('selfplay_nontrivial_alignment',any((x.get('reward_max') or 0)>1e-12 for x in intrinsic))
        check('selfplay_probability_movement',any((x.get('distribution_movement_l1') or 0)>1e-10 for x in intrinsic))
        check('selfplay_no_target_feedback',all('target_before' not in x and 'hard_raw_before' not in x and 'easy_raw_before' not in x for x in intrinsic))
        report['runs']['selfplay_resume'] = {k:resumed.get(k) for k in
            ('status','last_checkpoint_id','active_training_seconds','learner_steps','outer_rounds')}
    except BaseException:
        report['failures'].append('actual_resume_exception')
        report['resume_traceback'] = traceback.format_exc()
    report['passed'] = not report['failures']
    report['finished_at'] = time.time()
    publish()
    print(json.dumps({'passed':report['passed'],'failures':report['failures'],'path':str(report_path)}),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
