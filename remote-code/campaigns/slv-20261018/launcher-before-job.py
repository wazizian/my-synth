#!/usr/bin/env python3
"""Run a standalone Python job on a rented RTX 4090 and collect its outputs."""
import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import threading
import sys
import uuid
import smoke

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, default=ROOT / 'example_jax.py')
    parser.add_argument('--requirements', type=Path, help='Additional requirements file, installed by uv')
    parser.add_argument('--max-hourly', type=float, default=.60)
    parser.add_argument('--timeout-seconds', type=int, default=1800, help='Job runtime limit; setup has a separate 15 minute limit')
    parser.add_argument('--boot-timeout-seconds', type=int, default=600, help='Maximum wait for container and SSH readiness')
    parser.add_argument('--disk', type=int, default=40)
    parser.add_argument('--search-only', action='store_true')
    parser.add_argument('--sync-interval', type=int, default=30, help='Fetch metrics/checkpoints every N seconds (minimum 10)')
    parser.add_argument('--sync-timeout-seconds', type=int, default=300, help='Maximum duration of each incremental/final transfer')
    parser.add_argument('--require-artifact', action='append', default=[], help='Fail if this output-relative glob is absent after final sync (repeatable)')
    parser.add_argument('job_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if shutil.which('rsync') is None:
        parser.error('Install rsync on this host before launching a job.')
    if any(Path(pattern).is_absolute() or '..' in Path(pattern).parts for pattern in args.require_artifact):
        parser.error('Artifact patterns must be relative paths inside outputs/.')
    if args.timeout_seconds <= 0 or args.boot_timeout_seconds <= 0 or args.disk < 30 or args.max_hourly <= 0 or args.sync_interval < 10 or args.sync_timeout_seconds < 10:
        parser.error('Timeout and price must be positive; disk >=30 GB; sync interval and transfer timeout >=10 seconds.')
    for p in [args.job, ROOT / 'requirements-jax.txt', ROOT / 'example_jax.py'] + ([args.requirements] if args.requirements else []):
        if not p.is_file():
            parser.error(f'Missing file: {p}')
    args.image = 'pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime'
    args.lifetime = args.boot_timeout_seconds + 900 + args.timeout_seconds + 4 * args.sync_timeout_seconds + 120
    run_dir = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    def workload(ssh):
        run_dir.mkdir(parents=True)
        (run_dir / 'submitted-job.py').write_bytes(args.job.read_bytes())
        (run_dir / 'requirements-jax.txt').write_bytes((ROOT / 'requirements-jax.txt').read_bytes())
        if args.requirements:
            (run_dir / 'extra-requirements.txt').write_bytes(args.requirements.read_bytes())
        (run_dir / 'metadata.json').write_text(json.dumps({
            'job': str(args.job.resolve()), 'timeout_seconds': args.timeout_seconds,
            'image': args.image, 'max_hourly': args.max_hourly,
            'instance': json.loads(smoke.STATE.read_text()), 'ssh': ssh}, indent=2))
        print(f'Local logs and outputs: {run_dir}', flush=True)

        def remote(command, timeout, **kwargs):
            return subprocess.run(ssh + [command], timeout=timeout, check=True, **kwargs)

        def upload(local, name):
            remote('mkdir -p /workspace/job/outputs/{metrics,checkpoints,preflight}; cat > /workspace/job/' + name,
                   120, input=local.read_bytes())

        stop_sync = threading.Event()
        sync_thread = None
        sync_status = {'successes': 0, 'failures': 0, 'final_sync_complete': False}
        output_dir = run_dir / 'outputs'
        output_dir.mkdir()

        def fetch_outputs():
            command = ['rsync', '-rlt', '--safe-links', '--partial', '--delay-updates',
                       '--exclude=*.tmp*', '--timeout=60', '-e', shlex.join(ssh[:-1]),
                       ssh[-1] + ':/workspace/job/outputs/', str(output_dir) + '/']
            with (run_dir / 'sync.log').open('a') as sync_log:
                result = subprocess.run(command, stdout=sync_log, stderr=sync_log, timeout=args.sync_timeout_seconds)
            if result.returncode != 0:
                sync_status['failures'] += 1
                raise RuntimeError(f'Output sync failed (rsync {result.returncode}); see sync.log')
            sync_status['successes'] += 1
            sync_status['last_success_utc'] = datetime.now(timezone.utc).isoformat()
            (run_dir / 'sync-status.json').write_text(json.dumps(sync_status, indent=2))

        def sync_loop():
            while not stop_sync.is_set():
                try:
                    fetch_outputs()
                except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                    print(f'Periodic sync: {exc}', file=sys.stderr, flush=True)
                stop_sync.wait(args.sync_interval)

        def record_inventory():
            inventory = []
            for file in sorted(output_dir.rglob('*')):
                if file.is_file() and not file.is_symlink() and '.tmp' not in file.name:
                    digest = hashlib.sha256()
                    with file.open('rb') as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b''):
                            digest.update(block)
                    inventory.append({'path': str(file.relative_to(output_dir)),
                                      'bytes': file.stat().st_size, 'sha256': digest.hexdigest()})
            (run_dir / 'artifacts.json').write_text(json.dumps(inventory, indent=2))

        with (run_dir / 'console.log').open('w') as log:
            try:
                upload(args.job, 'job.py')
                upload(ROOT / 'example_jax.py', 'verify.py')
                upload(ROOT / 'requirements-jax.txt', 'requirements.txt')
                if args.requirements:
                    upload(args.requirements, 'extra.txt')
                setup = '''set -eu
cd /workspace/job
apt-get update
apt-get install -y --no-install-recommends curl rsync
curl -LsSf https://astral.sh/uv/0.13.1/install.sh -o /tmp/install-uv.sh
UV_UNMANAGED_INSTALL=/opt/uv sh /tmp/install-uv.sh
export PATH=/opt/uv:$PATH
uv python install 3.12
uv venv --python 3.12 /opt/jax
uv pip compile --python /opt/jax/bin/python requirements.txt -o outputs/requirements.lock
uv pip sync --python /opt/jax/bin/python outputs/requirements.lock
'''
                if args.requirements:
                    setup = setup.replace('requirements.txt -o outputs/', 'requirements.txt extra.txt -o outputs/')
                setup += '''uv pip check --python /opt/jax/bin/python
uv pip freeze --python /opt/jax/bin/python > outputs/environment.txt
uv --version > outputs/uv-version.txt
nvidia-smi > outputs/nvidia-smi.txt
'''
                remote('bash -c ' + shlex.quote(setup), 900, stdout=log, stderr=subprocess.STDOUT)
                sync_thread = threading.Thread(target=sync_loop, daemon=True)
                sync_thread.start()
                env = 'env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false'
                user_env = env + ' JOB_OUTPUT_DIR=/workspace/job/outputs JOB_METRICS_DIR=/workspace/job/outputs/metrics JOB_CHECKPOINT_DIR=/workspace/job/outputs/checkpoints'
                job_args = args.job_args[1:] if args.job_args[:1] == ['--'] else args.job_args
                command = ('set -eu\ncd /workspace/job\n'
                           'nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv -l 5 > outputs/gpu.csv &\n'
                           'monitor=$!\ntrap \'kill "$monitor" 2>/dev/null || true\' EXIT\n' +
                           env + ' JOB_OUTPUT_DIR=/workspace/job/outputs/preflight /opt/jax/bin/python -u verify.py\n' +
                           user_env + ' /opt/jax/bin/python -u job.py ' + shlex.join(job_args))
                remote('bash -c ' + shlex.quote(command), args.timeout_seconds, stdout=log, stderr=subprocess.STDOUT)
            finally:
                stop_sync.set()
                if sync_thread:
                    sync_thread.join(timeout=args.sync_timeout_seconds + 5)
                # Retry final metrics/checkpoint sync before instance destruction.
                try:
                    for attempt in range(3):
                        try:
                            fetch_outputs()
                            sync_status['final_sync_complete'] = True
                            (run_dir / 'sync-status.json').write_text(json.dumps(sync_status, indent=2))
                            break
                        except (OSError, subprocess.SubprocessError, RuntimeError):
                            if attempt == 2:
                                raise
                except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                    print(f'Output retrieval failed: {exc}; instance cleanup will proceed.', file=sys.stderr)
                    raise
                finally:
                    record_inventory()
        print((run_dir / 'console.log').read_text()[-6000:], flush=True)
        required = args.require_artifact or (['metrics/train.jsonl', 'checkpoints/step-000100.eqx']
                    if args.job.resolve() == (ROOT / 'example_jax.py').resolve() else [])
        missing = [pattern for pattern in required if not any(
            f.is_file() and not f.is_symlink() for f in output_dir.glob(pattern))]
        if missing:
            raise RuntimeError(f'Required artifacts were not retrieved: {missing}')
        print(f'Metrics and checkpoints saved to {output_dir}', flush=True)

    try:
        smoke.run(args, workload)
    except (Exception, KeyboardInterrupt) as exc:
        print(f'ERROR: {exc}. Logs: {run_dir}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
