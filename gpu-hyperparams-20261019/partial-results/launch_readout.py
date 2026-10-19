"""Control-plane IO only: wait for frozen DPG checkpoints and detach GPU readout."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def main():
    """Wait at most 30 minutes, snapshot supplemental source, launch once and record exit."""
    root = Path(__file__).resolve().parent
    run = root / 'runs/20261019T174738Z-f7c0546f'
    ssh = json.loads((run / 'metadata.json').read_text())['ssh']
    default = '/workspace/hyperparameter-trials/confirm-dpg-default'
    selected = '/workspace/hyperparameter-trials/confirm-dpg-selected'
    deadline = min(time.time() + 1800,
                   json.loads((root / 'budget.json').read_text())['created_at'] + 12700)
    while time.time() < deadline:
        try:
            status = subprocess.run(ssh + [f'test -f {default}/status.json && test -f {selected}/status.json'],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            if status.returncode == 0:
                break
        except (OSError, subprocess.TimeoutExpired):
            pass
        print('Waiting for completed DPG confirmation checkpoints', flush=True)
        time.sleep(30)
    else:
        raise TimeoutError('DPG checkpoints unavailable within readout launch deadline')
    prep = ('set -eu\n'
            'test ! -e /workspace/job/outputs/readout/submitted.json\n'
            'cp -a /workspace/neural-sde-source /workspace/readout-supplement\n'
            'mkdir -p /workspace/job/outputs/readout/inputs\n'
            f'cp {default}/latest.pkl /workspace/job/outputs/readout/inputs/default.pkl\n'
            f'cp {selected}/latest.pkl /workspace/job/outputs/readout/inputs/selected.pkl\n'
            'cat > /workspace/readout-supplement/neural_sde_hedging/teacher_readout.py\n')
    subprocess.run(ssh + [prep], input=(root / 'teacher_readout.py').read_bytes(), check=True, timeout=60)
    command = ('timeout --signal=TERM --kill-after=30 900 env -u LD_LIBRARY_PATH '
               'JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false '
               'PYTHONPATH=/workspace/readout-supplement '
               '/opt/jax/bin/python -u -m neural_sde_hedging.teacher_readout '
               '--default-checkpoint /workspace/job/outputs/readout/inputs/default.pkl '
               '--selected-checkpoint /workspace/job/outputs/readout/inputs/selected.pkl '
               '--output /workspace/job/outputs/readout')
    record = {'submitted_at': time.time(), 'command': command,
              'source_sha256': hashlib.sha256((root / 'teacher_readout.py').read_bytes()).hexdigest(),
              'mode': 'same GPU worker, detached supplemental diagnostic; original training source unchanged'}
    subprocess.run(ssh + ['cat > /workspace/job/outputs/readout/submitted.json'],
                   input=json.dumps(record, indent=2).encode(), timeout=30, check=True)
    subprocess.run(ssh + ['cd /workspace/readout-supplement && sha256sum neural_sde_hedging/*.py > /workspace/job/outputs/readout/source-sha256.txt && sha256sum /workspace/job/outputs/readout/inputs/*.pkl > /workspace/job/outputs/readout/input-sha256.txt'],
                   timeout=30, check=True)
    wrapper = command + '; result=$?; printf "%s\\n" "$result" > /workspace/job/outputs/readout/exit-code'
    detached = 'nohup bash -c ' + shlex.quote(wrapper) + ' >/workspace/job/outputs/readout/console.log 2>&1 < /dev/null &'
    subprocess.run(ssh + [detached], timeout=30, check=True)
    (root / 'readout-submitted.json').write_text(json.dumps(record, indent=2))
    print('Detached GPU readout submitted', flush=True)
    result_deadline = min(time.time() + 960,
                          json.loads((root / 'budget.json').read_text())['created_at'] + 12700)
    while time.time() < result_deadline:
        try:
            result = subprocess.run(ssh + ['if test -f /workspace/job/outputs/readout/exit-code; then cat /workspace/job/outputs/readout/exit-code; else echo pending; fi'],
                                    capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            time.sleep(15)
            continue
        status = result.stdout.strip()
        if result.returncode == 0 and status != 'pending':
            print('Readout exit:', status, flush=True)
            (root / 'readout-exit-code').write_text(status + '\n')
            if status != '0':
                raise RuntimeError('Readout failed; inspect exported console.log')
            return
        time.sleep(15)
    raise TimeoutError('Readout result unavailable at deadline')


if __name__ == '__main__':
    main()
