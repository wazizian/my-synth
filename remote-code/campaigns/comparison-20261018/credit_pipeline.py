"""Durable credit comparison controller; numerical selection/evaluation on GPUs."""
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
        write('credit-pipeline-state.json', dict(stage=stage,
              utc=datetime.now(timezone.utc).isoformat()))
        time.sleep(30)


def gates_ready():
    for campaign in ('credit-ordinary-long-20261018', 'credit-high-default-20261018'):
        for bp in (1, 3, 10):
            gate_launch = ROOT / f'{campaign}-gate-{bp}bp/launcher'
            if (gate_launch / 'active.json').exists():
                return False
            if not any((r / 'outputs/gate/complete.json').exists()
                       and (r / 'outputs/gate/results.json').exists()
                       and (r / 'artifacts.json').exists()
                       for r in (ROOT / f'{campaign}-gate-{bp}bp/launcher/runs').glob('*')):
                return False
    return True


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
        raise RuntimeError('Final evaluator is not approved by GPU evidence')
    for name, expected in approval['source_hashes'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Evaluator source changed after GPU validation: ' + name)
    if approval.get('packager_sha256') != hashlib.sha256((ROOT / 'package_evaluation.py').read_bytes()).hexdigest():
        raise RuntimeError('Final evaluation packager changed after GPU validation')
    return approval, hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    lock = (ROOT / 'credit-pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    wait_for(gates_ready, 'waiting_for_all_bounded_development_gates')
    wait_for(lambda: (ROOT / 'training_approval.json').exists(), 'waiting_for_GPU_training_validation')
    approval = json.loads((ROOT / 'training_approval.json').read_text())
    if approval.get('status') != 'GPU_validated':
        raise RuntimeError('Training implementation not approved by GPU evidence')
    for name, expected in approval['source_hashes'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Training source changed after GPU validation: ' + name)
    if not (ROOT / 'credit_freeze.json').exists():
        with (ROOT / 'credit-freeze-control.log').open('a') as log:
            subprocess.run(['python', str(ROOT / 'launch_freeze.py')],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    freeze = json.loads((ROOT / 'credit_freeze.json').read_text())
    selected = freeze['selected']
    source = ROOT.parent / selected['campaign'] / 'source'
    design_path = ROOT / 'design.credit.frozen.json'
    if not design_path.exists():
        design = json.loads((ROOT / 'design.credit.draft.json').read_text())
        design.update(status='frozen_for_training', scale_status='frozen_from_development',
                      scales=selected['scales'], adam_eps_root=approval['adam_eps_root'],
                      deadline_utc='2026-10-19T06:05:00+00:00',
                      selected_campaign=selected['campaign'], selected_bp=selected['bp'],
                      comparison_scope=freeze['comparison_scope'],
                      discovery_gate_passed=selected['passed'])
        write(design_path.name, design)
    namespace = ROOT / 'credit-final'
    from budget_admission import wait_for_admission
    write('credit-pipeline-state.json', dict(stage='checking_wallet_and_remaining_commitments',utc=datetime.now(timezone.utc).isoformat()))
    wait_for_admission('credit', namespace)
    # The allocator adopts existing arms and fills only unattempted slots.
    with (ROOT / 'credit-final-control.log').open('a') as log:
        subprocess.run(['python', str(ROOT / 'launch_arms.py'), '--benchmark', 'credit',
                '--source', str(source), '--design', str(design_path), '--bp', str(selected['bp']),
                '--namespace', str(namespace)], stdout=log, stderr=subprocess.STDOUT, check=True)
    wait_for(lambda: complete_runs(namespace) is not None, 'waiting_for_eight_frozen_training_selections')
    runs = complete_runs(namespace)
    wait_for(lambda: (ROOT / 'evaluation_approval.json').exists(), 'waiting_for_GPU_evaluator_validation')
    evaluation_approval, evaluation_approval_sha256 = validated_evaluation_approval()
    manifest_path = ROOT / 'credit-final-manifest.json'
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

        write(manifest_path.name, dict(benchmark='credit', source=str(source),
              bp=selected['bp'], frozen=True, models=models, curves=curves, trial_curves=trial_curves,
              seed=520261017, episodes=8192, bootstrap=1000,
              design=json.loads(design_path.read_text()), design_path=str(design_path),
              adam_eps_root=json.loads(design_path.read_text())['adam_eps_root'],
              scales=json.loads(design_path.read_text())['scales'],
              comparison_scope=freeze['comparison_scope'], discovery_freeze=freeze,
              training_approval=approval, evaluation_approval=evaluation_approval,
              evaluation_approval_sha256=evaluation_approval_sha256))
    frozen_manifest = json.loads(manifest_path.read_text())
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
            '--job', str(worker), '--requirements', str(source.parent / 'requirements.txt'),
            '--timeout-seconds', str(min(7200, int(DEADLINE - time.time() - 300))),
            '--max-hourly', '0.60', '--require-artifact', 'evaluation/complete.json',
            '--require-artifact', 'evaluation/results.json']
        validated_evaluation_approval()
        run_preallocation(command, launch, evaluation / 'launcher.log', DEADLINE)
    run = next((r for r in sorted((launch / 'runs').glob('*'), reverse=True)
                if (r / 'outputs/evaluation/complete.json').exists() and (r / 'artifacts.json').exists()), None)
    if run is None or (launch / 'active.json').exists():
        raise RuntimeError('Existing evaluation needs reattachment or verified cleanup')
    shutil.copy2(run / 'outputs/evaluation/results.json', ROOT / 'credit-final-results.json')
    write('credit-pipeline-state.json', dict(stage='eight_arm_training_and_final_evaluation_complete',
                                           run=str(run), report_pending=True))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        (ROOT / 'credit-pipeline-failure.txt').write_text(traceback.format_exc())
        raise
