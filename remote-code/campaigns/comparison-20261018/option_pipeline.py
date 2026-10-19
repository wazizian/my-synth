"""Durable conditional option comparison controller; numerical selection/evaluation on GPUs."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
from rental_control import run_preallocation, resumable_preallocation
import shutil
import subprocess
import time
import traceback
from budget_admission import wait_for_admission

ROOT = Path(__file__).resolve().parent
ARMS = ('baseline', 'easy_reference', 'hard_reference', 'dpg_hard', 'soar_hard',
        'selfplay', 'dpg_mixed', 'soar_mixed')
DEADLINE = datetime(2026, 10, 19, 6, 5, tzinfo=timezone.utc).timestamp()


def write(name, value):
    path = ROOT / name
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def wait_for(predicate, stage):
    while not predicate():
        if time.time() > DEADLINE - 1800:
            raise RuntimeError('Deadline leaves no time for required final evaluation')
        write('option-pipeline-state.json', dict(stage=stage,
              utc=datetime.now(timezone.utc).isoformat()))
        time.sleep(30)


def gates_ready():
    campaign=ROOT.parent/'volatility-shocks-20261018'
    return (campaign/'option_freeze.json').exists() and all((campaign/f'gate-{bp}bp-results.json').exists() for bp in (1,3,10))


def complete_runs(namespace):
    runs = {}
    for arm in ARMS:
        launch = namespace / arm / 'launcher'
        if (launch / 'active.json').exists():
            return None
        run = next((r for r in sorted((launch / 'runs').glob('*'), reverse=True)
                    if all((r / p).exists() for p in
                           ('outputs/complete.json', 'outputs/selected.pkl',
                            'outputs/selection.json', 'artifacts.json'))), None)
        if run is None:
            return None
        runs[arm] = run
    return runs


def validated_evaluation_approval():
    path = ROOT / 'evaluation_approval.json'
    approval = json.loads(path.read_text())
    if approval.get('status') != 'GPU_validated':
        raise RuntimeError('Evaluator not validated')
    for name, expected in approval['source_hashes'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Evaluator source changed: ' + name)
    if hashlib.sha256((ROOT / 'package_evaluation.py').read_bytes()).hexdigest() != approval['packager_sha256']:
        raise RuntimeError('Final packager changed after proof')
    return approval, hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    lock = (ROOT / 'option-pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    wait_for(gates_ready, 'waiting_for_all_bounded_development_gates')
    wait_for(lambda: (ROOT / 'option_training_approval.json').exists(), 'waiting_for_GPU_training_validation')
    approval = json.loads((ROOT / 'option_training_approval.json').read_text())
    if approval.get('status') != 'GPU_validated':
        raise RuntimeError('Training implementation not approved by GPU evidence')
    for name, expected in approval['source_hashes'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Training source changed after GPU validation: ' + name)
    freeze_path=ROOT.parent/'volatility-shocks-20261018/option_freeze.json'
    freeze=json.loads(freeze_path.read_text())
    selected=freeze['selected']
    if selected is None:
        shutil.copy2(freeze_path,ROOT/'option_freeze.json')
        write('option-pipeline-state.json',dict(stage='skipped_no_cost_passed_specialist_gate',final_test_opened=False,freeze=str(freeze_path)))
        (ROOT/'option-skip-report.md').write_text('The option comparison was skipped because none of the three declared transaction costs passed both the specialist screen and independent confirmation. The useful-gain threshold remained 0.0005, and no final test stream was opened. Development diagnostics remain under volatility-shocks-20261018/gate-1bp-results.json, gate-3bp-results.json and gate-10bp-results.json.\n')
        for name in ['option-discovery-report.md','option-discovery-report.tex']:
            original=freeze_path.parent/name
            if original.exists():shutil.copy2(original,ROOT/name)
        return
    wait_for(lambda:(ROOT/'controller_handoff_approval.json').exists(),'waiting_for_validated_memory_safe_controller')
    handoff=json.loads((ROOT/'controller_handoff_approval.json').read_text())
    if handoff.get('status')!='GPU_validated':raise RuntimeError('Memory-safe handoff control not validated')
    for name,expected in handoff['source_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Controller source changed: '+name)
    wait_for(lambda:(ROOT/'history_archive_approval.json').exists(),'waiting_for_portable_history_wrapper_validation')
    history=json.loads((ROOT/'history_archive_approval.json').read_text())
    if history.get('status')!='GPU_validated':raise RuntimeError('History archive wrapper not validated')
    for name,expected in history['source_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('History wrapper changed: '+name)
    source=ROOT/'option-source'
    design_path = ROOT / 'design.option.frozen.json'
    if not design_path.exists():
        design = json.loads((ROOT / 'design.option.draft.json').read_text())
        design.update(status='frozen_for_training', scale_status='frozen_from_development',
                      scales=selected['scales'], adam_eps_root=approval['adam_eps_root'],
                      deadline_utc='2026-10-19T06:05:00+00:00',
                      selected_campaign='volatility-shocks-20261018', selected_bp=selected['bp'],
                      comparison_scope=freeze['comparison_scope'],
                      discovery_gate_passed=selected['passed'])
        write(design_path.name, design)
    namespace = ROOT / 'option-final'
    wait_for_admission('option', namespace)
    reservation=namespace/'budget-reservation.json'
    if not reservation.exists():
        def budget_ready():
            path=ROOT/'option_budget_approval.json'
            if not path.exists():return False
            row=json.loads(path.read_text())
            return (row.get('status')=='approved' and row.get('selected_bp')==selected['bp']
                    and row.get('reserve_usd',0)>0
                    and datetime.fromisoformat(row['valid_until_utc']).timestamp()>time.time())
        wait_for(budget_ready,'waiting_for_fresh_wallet_offer_projection_and_eight_arm_budget_reservation')
        namespace.mkdir(exist_ok=True)
        shutil.copy2(ROOT/'option_budget_approval.json',reservation)
    # The allocator adopts existing arms and fills only unattempted slots.
    with (ROOT / 'option-final-control.log').open('a') as log:
        subprocess.run(['python', str(ROOT / 'launch_arms.py'), '--benchmark', 'option',
                '--source', str(source), '--design', str(design_path), '--bp', str(selected['bp']),
                '--namespace', str(namespace)], stdout=log, stderr=subprocess.STDOUT, check=True)
    wait_for(lambda: complete_runs(namespace) is not None, 'waiting_for_eight_frozen_training_selections')
    runs = complete_runs(namespace)
    wait_for(lambda: (ROOT / 'evaluation_approval.json').exists(), 'waiting_for_GPU_evaluator_validation')
    evaluation_approval, evaluation_approval_sha256 = validated_evaluation_approval()
    manifest_path = ROOT / 'option-final-manifest.json'
    if not manifest_path.exists():
        models = {}
        curves = {}
        trial_curves = []
        for arm, run in runs.items():
            cp = run / 'outputs/selected.pkl'
            models[arm] = dict(checkpoint=str(cp), selection=str(run / 'outputs/selection.json'),
                               sha256=hashlib.sha256(cp.read_bytes()).hexdigest(), kind='pickle')
            curves[arm] = [str(p) for p in sorted((run / 'outputs/training').glob('lr-*/training.jsonl'))]
            for trial in sorted((run / 'outputs/training').glob('lr-*')):
                row = dict(arm=arm, lr=float(trial.name.removeprefix('lr-')))
                for category in ('selection', 'monitor'):
                    points = []
                    for path in sorted(trial.glob(category + '-*.json')):
                        boundary = float(path.stem.removeprefix(category + '-'))
                        points.append(dict(active_seconds=boundary, path=str(path)))
                    row[category] = sorted(points, key=lambda value: value['active_seconds'])
                trial_curves.append(row)

        write(manifest_path.name, dict(benchmark='option', source=str(source),
              bp=selected['bp'], frozen=True, models=models, curves=curves, trial_curves=trial_curves,
              seed=520261017, episodes=8192, bootstrap=1000,
              design=json.loads(design_path.read_text()), design_path=str(design_path),
              discovery_freeze=freeze, adam_eps_root=approval['adam_eps_root'],
              scales=json.loads(design_path.read_text())['scales'],
              comparison_scope=freeze['comparison_scope'], training_approval=approval,
              evaluation_approval=evaluation_approval,
              evaluation_approval_sha256=evaluation_approval_sha256,
              initial_premiums=json.loads((source/'initial_premiums.json').read_text()),
              static_addenda={name:str(ROOT/name) for name in ['OPTION_SCIENTIFIC_LIMITATIONS.md','OPTION_SCIENTIFIC_LIMITATIONS.tex']}))
    frozen_manifest=json.loads(manifest_path.read_text())
    if frozen_manifest.get('evaluation_approval_sha256') != evaluation_approval_sha256:
        raise RuntimeError('Evaluator approval changed after final manifest freeze; review before opening final stream')
    evaluation = namespace / 'evaluation'
    evaluation.mkdir(exist_ok=True)
    launch = evaluation / 'launcher'
    while (launch / 'active.json').exists():
        wait_for(lambda: not (launch / 'active.json').exists(), 'adopting_existing_final_evaluation')
    exported = any((r / 'outputs/evaluation/complete.json').exists() and (r / 'artifacts.json').exists() for r in (launch / 'runs').glob('*'))
    if not exported:
        if (evaluation / 'launcher.log').exists() and not resumable_preallocation(launch, evaluation / 'launcher.log'):
            raise RuntimeError('Failed existing final evaluation requires explicit recovery')
        launch.mkdir(exist_ok=True)
        template = ROOT.parent / 'credit-ordinary-long-20261018/launchers/baseline'
        for name in ('job.py', 'smoke.py', 'requirements-jax.txt', 'example_jax.py',
                     'USAGE.txt', 'AGENTS.md'):
            shutil.copy2(template / name, launch / name)
        subprocess.run(['python', str(ROOT / 'harden_export.py'), str(launch)], check=True)
        worker = evaluation / 'worker.py'
        validated_evaluation_approval()
        subprocess.run(['python', str(ROOT / 'package_evaluation.py'), str(manifest_path), str(worker)], check=True)
        command = ['uv', 'run', '--no-project', 'python', str(launch / 'job.py'),
            '--job', str(worker), '--requirements', str(ROOT.parent/'volatility-shocks-20261018/requirements.txt'),
            '--timeout-seconds', str(min(7200, int(DEADLINE - time.time() - 300))),
            '--max-hourly', '0.60', '--require-artifact', 'evaluation/complete.json',
            '--require-artifact', 'evaluation/results.json']
        validated_evaluation_approval()
        run_preallocation(command, launch, evaluation / 'launcher.log', DEADLINE)
    run = next((r for r in sorted((launch / 'runs').glob('*'), reverse=True)
                if (r / 'outputs/evaluation/complete.json').exists() and (r / 'artifacts.json').exists()), None)
    if run is None or (launch / 'active.json').exists():
        raise RuntimeError('Existing evaluation needs reattachment or verified cleanup')
    shutil.copy2(run / 'outputs/evaluation/results.json', ROOT / 'option-final-results.json')
    write('option-pipeline-state.json', dict(stage='eight_arm_training_and_final_evaluation_complete',
                                           run=str(run), report_pending=True))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        (ROOT / 'option-pipeline-failure.txt').write_text(traceback.format_exc())
        raise
