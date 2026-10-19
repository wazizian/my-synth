"""One worker, one sequential lease, independent persistent trial states."""
import argparse
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))


def selfplay_guard(config):
    """Freeze a generous measured-rate limit before the first production run."""
    profile_path=root/'results/profile.json'
    profile=json.loads(profile_path.read_text())
    rows=[row for row in profile.get('runs',[]) if row.get('run_id')=='profile_selfplay'
          and row.get('status')=='completed']
    if len(rows)!=1:raise RuntimeError('Completed production selfplay profile required for storage planning')
    row=rows[0]
    if any(row.get(key)!=config[key] for key in ('batch_size','block_steps','hidden_size')):
        raise RuntimeError('Selfplay storage profile shape mismatch')
    seconds,rounds,size=row.get('active_training_seconds'),row.get('outer_rounds'),row.get('selfplay_mean_snapshot_bytes')
    if not seconds or not rounds or not size or seconds<=0 or rounds<=0 or size<=0:
        raise RuntimeError('Selfplay profile lacks measured throughput or snapshot size')
    planned_seconds=float(config.get('final_active_seconds',2700))
    cap=max(2,math.ceil(2*planned_seconds*rounds/seconds))
    snapshot_bytes=math.ceil(size)
    storage_limit=12*1024**3
    projected=cap*snapshot_bytes
    reserve=1024**3
    if projected>storage_limit:raise RuntimeError('Measured selfplay block plan exceeds 12 GiB history limit')
    if shutil.disk_usage(root).free<projected+reserve:
        raise RuntimeError('Insufficient free disk for planned selfplay snapshots plus checkpoint/export reserve')
    return dict(selfplay_max_blocks=cap,selfplay_history_limit_bytes=storage_limit,
                selfplay_snapshot_estimate_bytes=snapshot_bytes,selfplay_storage_reserve_bytes=reserve,
                selfplay_planned_active_seconds=planned_seconds,
                selfplay_guard_profile={'run_id':row['run_id'],'outer_rounds':rounds,
                    'active_training_seconds':seconds,'snapshot_bytes':size,'rate_multiplier':2.,
                    'profile_gpu_model':row.get('gpu_model'),'projected_history_bytes':projected})


def final_complete(directory,checkpoint_id):
    path=directory/f'final_test_{checkpoint_id}.json'
    if not path.exists():return False
    outcome=json.loads(path.read_text())
    for domain in ('base','easy','hard'):
        row=outcome.get('domains',{}).get(domain,{})
        artifact=row.get('episode_metrics_path')
        if (row.get('status')!='ok' or row.get('checkpoint_id')!=checkpoint_id or
            row.get('raw_mse') is None or not artifact or not (root/artifact).is_file()):return False
    return True


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--seconds',type=float,default=600)
    p.add_argument('--evaluate',action='store_true')
    a=p.parse_args()
    if len(a.runs)!=len(set(a.runs)):raise ValueError('Duplicate run lease')
    for run_id in a.runs:
        if Path(run_id).name!=run_id:raise ValueError('Unsafe run ID')
    failures=[];completed=[];checkpoint_records=[]
    needs_sensitivity=not a.evaluate and set(a.runs)=={'baseline-lr001','hard_reference-lr001'}
    sensitivity_status='pending' if needs_sensitivity else 'not_applicable'
    stage='worker_preflight'
    def publish():
        (root/'group_status.json').write_text(json.dumps(dict(runs=a.runs,completed_runs=completed,
            failures=failures,checkpoints=checkpoint_records,evaluate=a.evaluate,
            pricing_sensitivity_status=sensitivity_status,
            status='complete' if len(completed)==len(a.runs) and not failures and
                sensitivity_status in ('complete','not_applicable') else 'running_or_failed',
            updated_at=time.time()),indent=2))
    publish()
    try:
        # Metadata-only local commit, no remote/push. Existing frozen source must
        # verify exactly; changes require explicit supervisor --update.
        subprocess.run([sys.executable,str(root/'scripts/commit_source.py'),'--root',str(root)],check=True)
        if needs_sensitivity:
            stage='resident_table_profiling'
            profile_arms=('baseline','hard_reference','dpg_hard','soar_hard','selfplay','dpg_mixed','soar_mixed')
            before_profile={}
            for arm in profile_arms:
                path=root/'runs'/('profile_'+arm)/'status.json'
                if not path.exists():raise RuntimeError('Existing retained production profile required: '+arm)
                row=json.loads(path.read_text())
                before_profile[arm]={key:row.get(key) for key in
                    ('active_training_seconds','outer_rounds','simulation_seconds','source_hash','code_commit')}
            subprocess.run([sys.executable,str(root/'scripts/profile_families.py'),
                '--active-seconds','15','--campaign-start','1792274040','--deadline','1792301040',
                '--gpu-csv','/workspace/job/outputs/gpu.csv'],cwd=root,check=True)
            gate=json.loads((root/'results/production_profile_gate.json').read_text())
            if not gate.get('passed') or set(gate.get('runs',{}))!=set(profile_arms):
                raise RuntimeError('Corrected resident-table production profiling did not pass all seven arms')
            note=dict(decision='Continue retained profile lineages to 15 cumulative active seconds before diagnostics',
                reason='Validate one-time GPU pricing-table residency at production shapes without discarding prior profile evidence',
                limitation='Cumulative profile rates include both original host-table and corrected resident-table segments; original events and code commits remain intact',
                before=before_profile,after={})
            for arm in profile_arms:
                row=json.loads((root/'runs'/('profile_'+arm)/'status.json').read_text())
                note['after'][arm]={key:row.get(key) for key in
                    ('active_training_seconds','outer_rounds','simulation_seconds','source_hash','code_commit')}
            (root/'results/residency_profile_transition.json').write_text(json.dumps(note,indent=2))
        from src.runtime import configure
        configure(root)
        from src.runner import train_one,evaluate_checkpoint
        stage='trial_execution'
        for run_id in a.runs:
            print('START',run_id,'evaluation' if a.evaluate else a.seconds,flush=True)
            try:
                directory=root/'runs'/run_id
                if a.evaluate:
                    checkpoints=sorted((directory/'checkpoints').glob('*/metadata.json'))
                    if not checkpoints:raise RuntimeError('No retained checkpoint to evaluate')
                    for path in checkpoints:
                        metadata=json.loads(path.read_text())
                        checkpoint_id=metadata['checkpoint_id']
                        if final_complete(directory,checkpoint_id):
                            checkpoint_records.append(dict(run_id=run_id,checkpoint_id=checkpoint_id,status='already_complete'))
                            continue
                        outcome=evaluate_checkpoint(root,run_id,path.parent,role='final_test')
                        valid=outcome.get('status')!='failed' and final_complete(directory,checkpoint_id)
                        checkpoint_records.append(dict(run_id=run_id,checkpoint_id=checkpoint_id,
                            status='complete' if valid else 'failed'))
                        if not valid:failures.append(dict(run_id=run_id,checkpoint=checkpoint_id,
                            reason=outcome.get('error','Incomplete final outcomes or episode artifacts')))
                        publish()
                else:
                    existing=directory/'config.json'
                    method=run_id.rsplit('-lr',1)[0]
                    config=json.loads(existing.read_text() if existing.exists() else (root/f'configs/methods/{method}.yaml').read_text())
                    config.update(run_id=run_id)
                    if not existing.exists() and run_id.endswith('-lr01'):config['generator_lr']=.01
                    if not existing.exists() and method=='selfplay':config.update(selfplay_guard(config))
                    config.setdefault('campaign_start',1792274040.)
                    config.setdefault('deadline',1792301040.)
                    result=train_one(root,config,a.seconds,resume=True)
                    if result.get('status')!='completed':raise RuntimeError('Trial did not complete: '+str(result.get('status')))
                    if result.get('active_training_seconds',0)<a.seconds:raise RuntimeError('Cumulative active cap not reached')
                if not any(item['run_id']==run_id for item in failures):completed.append(run_id)
            except Exception as exc:
                traceback.print_exc()
                failures.append(dict(run_id=run_id,reason=str(exc)))
                # Preserve the remaining independent runs; a failed arm cannot
                # overwrite prior retained statuses or silently skip the queue.
            publish()
            print('DONE',run_id,flush=True)
        if needs_sensitivity and not failures and len(completed)==len(a.runs):
            stage='pricing_sensitivity'
            before=set((root/'results/pricing_sensitivity').glob('*/report.json'))
            subprocess.run([sys.executable,str(root/'scripts/pricing_sensitivity.py'),
                '--runs',*a.runs,'--episodes','32','--checkpoint','best'],cwd=root,check=True)
            reports=set((root/'results/pricing_sensitivity').glob('*/report.json'))-before
            if len(reports)!=1:raise RuntimeError('Missing unique diagnostic pricing sensitivity report')
            report=json.loads(next(iter(reports)).read_text())
            if report.get('status')!='complete' or set(report.get('runs',{}))!=set(a.runs):
                raise RuntimeError('Pricing sensitivity diagnostics did not complete for both reference arms')
            sensitivity_status='complete'
    except Exception as exc:
        traceback.print_exc()
        failures.append(dict(run_id=None,reason=str(exc),stage=stage))
    publish()
    return 1 if failures or len(completed)!=len(a.runs) else 0


if __name__=='__main__':raise SystemExit(main())
