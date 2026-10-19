"""Control-plane launch of a frozen eight-arm comparison, no numerical imports."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from offer_exclusions import exclusion_arguments
from campaign_deadlines import assert_design_deadline

ARMS = ('baseline', 'easy_reference', 'hard_reference', 'dpg_hard', 'soar_hard',
        'selfplay', 'dpg_mixed', 'soar_mixed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=('credit', 'option'), required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--bp', type=int, required=True)
    parser.add_argument('--namespace', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, action='append')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    approval = json.loads((root / 'controller_handoff_approval.json').read_text())
    if approval.get('status') != 'GPU_validated':
        raise RuntimeError('Controller handoff must pass worker validation before allocation')
    for name in ('controller_handoff.py', 'enable_handoff.py', 'launch_arms.py',
                 'recover_credit.py', 'package_trial.py', 'archive_history.py',
                 'harden_export.py', 'export_safety.py', 'external_artifacts.py',
                 'external_artifact_worker.py', 'watchdog.py',
                 'external_export_coordinator.py', 'offer_exclusions.py', 'wallet_guard.py', 'campaign_deadlines.py'):
        if approval.get('source_hashes', {}).get(name) != hashlib.sha256((root / name).read_bytes()).hexdigest():
            raise RuntimeError(f'Controller handoff source changed since GPU validation: {name}')
    args.namespace.mkdir(parents=True, exist_ok=True)
    lock = (args.namespace / 'launch.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Existing namespace controller owns allocation; adopt it.', flush=True)
        return
    design = json.loads(args.design.read_text())
    if design.get('status') != 'frozen_for_training':
        raise RuntimeError('Require scientific freeze and GPU validation before rental')
    if design.get('scale_status') != 'frozen_from_development':
        raise RuntimeError('Target scales not frozen')
    missing_arms = []
    for arm in args.arm or ARMS:
        folder = args.namespace / arm
        launch = folder / 'launcher'
        if (launch / 'active.json').exists():
            continue
        completed = any(all((run / name).exists() for name in
                        ('outputs/complete.json', 'outputs/selected.pkl',
                         'outputs/selection.json', 'artifacts.json'))
                        for run in (launch / 'runs').glob('*'))
        if completed:
            continue
        if (folder / 'launcher.log').exists():
            raise RuntimeError(f'Failed existing attempt needs explicit recovery: {folder}')
        missing_arms.append(arm)
    if not missing_arms:
        print('All requested arms already allocated or exported; no duplicate rentals.', flush=True)
        return
    # Leave 30 minutes for final evaluation/export before the user's deadline.
    assert_design_deadline(args.namespace, design['deadline_utc'])
    deadline = datetime.fromisoformat(design['deadline_utc']).timestamp()
    remaining = deadline - datetime.now(timezone.utc).timestamp()
    if remaining < 9000:
        raise RuntimeError('Insufficient time for original trial budgets and final evaluation')
    timeout = min(12000, int(remaining - 1800))
    template = root.parent / 'credit-ordinary-long-20261018/launchers/baseline'
    for arm in missing_arms:
        folder = args.namespace / arm
        launch = folder / 'launcher'
        launch.mkdir(parents=True, exist_ok=True)
        if (launch / 'active.json').exists() or (folder / 'launcher.log').exists():
            raise RuntimeError(f'Existing attempt must be inspected/reattached: {folder}')
        for name in ('job.py', 'smoke.py', 'requirements-jax.txt', 'example_jax.py',
                     'USAGE.txt', 'AGENTS.md'):
            shutil.copy2(template / name, launch / name)
        subprocess.run([sys.executable, str(root / 'harden_export.py'), str(launch)], check=True)
        subprocess.run([sys.executable, str(root / 'enable_handoff.py'), str(launch)], check=True)
        worker = folder / 'worker.py'
        subprocess.run([sys.executable, str(root / 'package_trial.py'),
            '--benchmark', args.benchmark, '--source', str(args.source),
            '--design', str(args.design), '--arm', arm, '--all-learning-rates',
            '--bp', str(args.bp), '--output', str(worker)], check=True)
        requirements = root.parent / 'credit-ordinary-long-20261018/requirements.txt'
        if args.benchmark == 'option':
            requirements = root.parent / 'volatility-shocks-20261018/requirements.txt'
        command = [sys.executable, str(launch / 'job.py'),
            '--job', str(worker), '--requirements', str(requirements),
            '--timeout-seconds', str(timeout), '--max-hourly', '0.60',
            '--sync-timeout-seconds', '180', '--controller-handoff',
            '--handoff-deadline-utc', design['deadline_utc'], '--require-artifact', 'complete.json',
            '--require-artifact', 'selected.pkl', '--require-artifact', 'selection.json']
        if arm == 'selfplay':
            for rate in design['learning_rates']:
                archive = f'histories/lr-{rate:g}.tar.gz'
                command += ['--require-artifact', archive,
                            '--require-artifact', archive + '.manifest.json']
        command += exclusion_arguments()
        with (folder / 'launcher.log').open('w') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
        (folder / 'control.json').write_text(json.dumps(dict(
            arm=arm, benchmark=args.benchmark, bp=args.bp, command=command,
            design_sha256=hashlib.sha256(args.design.read_bytes()).hexdigest(),
            deadline_utc=design['deadline_utc'], control_pid=child.pid), indent=2))
        # Set up one worker at a time; remote numerical jobs run concurrently.
        code = child.wait()
        (folder / 'controller-exit.json').write_text(json.dumps(dict(
            returncode=code, utc=datetime.now(timezone.utc).isoformat())))
        if code:
            raise RuntimeError(f'Launcher failed before acknowledged handoff: {arm}')
        if not any((run / 'handoff.accepted.json').exists()
                   for run in (launch / 'runs').glob('*')):
            raise RuntimeError(f'Launcher exited without durable ownership handoff: {arm}')


if __name__ == '__main__':
    main()
