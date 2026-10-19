"""Control-only packaging of six development summaries for GPU selection."""
from pathlib import Path
from datetime import datetime, timezone
from rental_control import run_preallocation, resumable_preallocation
import fcntl
import json
import shutil
import subprocess
import time


def main():
    root = Path(__file__).resolve().parent
    folder = root / 'credit-freeze'
    folder.mkdir(exist_ok=True)
    lock = (folder / 'control.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'credit_freeze.json').exists():
        return
    inputs = {}
    for campaign in ('credit-ordinary-long-20261018', 'credit-high-default-20261018'):
        for bp in (1, 3, 10):
            gate = root / f'{campaign}-gate-{bp}bp'
            runs = sorted((gate / 'launcher/runs').glob('*'), reverse=True)
            run = next((r for r in runs if (r / 'outputs/gate/complete.json').exists()
                        and (r / 'outputs/gate/results.json').exists()
                        and (r / 'artifacts.json').exists()), None)
            if run is None:
                raise RuntimeError(f'Development gate incomplete: {campaign}/{bp}')
            inputs[f'{campaign}-{bp}bp.json'] = dict(
                campaign=campaign, results=json.loads((run / 'outputs/gate/results.json').read_text()))
    launch = folder / 'launcher'
    launch.mkdir(exist_ok=True)
    while (launch / 'active.json').exists():
        time.sleep(30)
    completed = next((r for r in sorted((launch / 'runs').glob('*'), reverse=True) if (r / 'outputs/credit_freeze.json').exists() and (r / 'artifacts.json').exists()), None)
    if completed is not None:
        shutil.copy2(completed / 'outputs/credit_freeze.json', root / 'credit_freeze.json')
        return
    if (folder / 'launcher.log').exists() and not resumable_preallocation(launch, folder / 'launcher.log'):
        raise RuntimeError('Failed selection attempt requires explicit recovery')
    template = root.parent / 'credit-ordinary-long-20261018/launchers/baseline'
    for name in ('job.py', 'smoke.py', 'requirements-jax.txt', 'example_jax.py',
                 'USAGE.txt', 'AGENTS.md'):
        shutil.copy2(template / name, launch / name)
    subprocess.run(['python', str(root / 'harden_export.py'), str(launch)], check=True)
    worker = folder / 'worker.py'
    worker.write_text('INPUTS=' + repr(inputs) + '\nSOURCE=' + repr((root / 'freeze_credit.py').read_text()) + '''
import os,pathlib,sys
root=pathlib.Path(os.environ['JOB_OUTPUT_DIR'])
inputs=root/'inputs';inputs.mkdir(exist_ok=True)
import json
for name,value in INPUTS.items():(inputs/name).write_text(json.dumps(value))
(root/'freeze_credit.py').write_text(SOURCE)
sys.path.insert(0,str(root))
from freeze_credit import main
main(inputs,root/'credit_freeze.json')
''')
    command = ['uv', 'run', '--no-project', 'python', str(launch / 'job.py'),
        '--job', str(worker), '--timeout-seconds', '600', '--max-hourly', '0.60',
        '--require-artifact', 'credit_freeze.json']
    deadline = datetime(2026, 10, 19, 6, 5, tzinfo=timezone.utc).timestamp()
    run_preallocation(command, launch, folder / 'launcher.log', deadline)
    run = next(r for r in sorted((launch / 'runs').glob('*'), reverse=True)
               if (r / 'outputs/credit_freeze.json').exists() and (r / 'artifacts.json').exists())
    if (launch / 'active.json').exists():
        raise RuntimeError('Selection worker cleanup unconfirmed')
    shutil.copy2(run / 'outputs/credit_freeze.json', root / 'credit_freeze.json')


if __name__ == '__main__':
    main()
