"""GPU-only smoke workload; generated source is embedded by the IO packager."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def main():
    """Extract pinned source, run GPU checks and full-grid short trials; save evidence."""
    output = Path(os.environ['JOB_OUTPUT_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    source = Path('/workspace/neural-sde-source')
    source.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode('SOURCE_ZIP_BASE64'))) as archive:
        archive.extractall(source)
    manifest = json.loads((source / 'SOURCE_SHA256.json').read_text())
    for relative, expected in manifest.items():
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Source checksum mismatch: ' + relative)
    (output / 'SOURCE_SHA256.json').write_text(json.dumps(manifest, indent=2))
    environment = os.environ | {'NEURAL_SDE_GPU_TESTS': '1', 'PYTHONPATH': str(source)}
    summary = {'started': time.time(), 'source_verified': True, 'stages': []}
    commands = [('pytest', [sys.executable, '-m', 'pytest', '-q', '-ra',
                           str(source / 'tests/test_neural_sde_gpu.py'),
                           '--junitxml=' + str(output / 'pytest.xml')])]
    commands += [(method + '-126day', [sys.executable, '-m', 'neural_sde_hedging.training',
                 '--method', method, '--output', str(output / (method + '-126day')),
                 '--seconds', '300', '--rounds', '2', '--groups', '2', '--items', '4',
                 '--inner-steps', '2', '--n-steps', '126', '--feedback-size', '16',
                 '--payoff-cap', '1', '--warmup-steps', '2'])
                 for method in ('dpg', 'soar', 'selfplay')]
    for name, command in commands:
        start = time.monotonic()
        print('START', name, flush=True)
        with (output / (name + '.log')).open('w') as stream:
            result = subprocess.run(command, cwd=source, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=600)
        record = {'name': name, 'exit_code': result.returncode, 'seconds': time.monotonic() - start}
        if result.returncode == 0 and name != 'pytest':
            status = json.loads((output / name / 'status.json').read_text())
            record['status'] = status
            if status['status'] != 'complete' or status['rounds'] != 2:
                record['exit_code'] = 2
            # A different interpreter entry point must load a CLI-created bank;
            # reconstruction functions may never be recorded under __main__.
            checkpoint = str(output / name / 'latest.pkl')
            check = 'from pathlib import Path; from neural_sde_hedging.training import load_checkpoint; s=load_checkpoint(Path(' + repr(checkpoint) + ')); assert s["round_index"]==2; print("cross-entrypoint checkpoint PASS")'
            with (output / (name + '-checkpoint.log')).open('w') as stream:
                restored = subprocess.run([sys.executable, '-c', check], cwd=source,
                                          env=environment, stdout=stream,
                                          stderr=subprocess.STDOUT, timeout=90)
            record['cross_entrypoint_checkpoint_exit'] = restored.returncode
            if restored.returncode:
                record['exit_code'] = restored.returncode
        summary['stages'].append(record)
        summary['finished'] = time.time()
        summary['passed'] = all(row['exit_code'] == 0 for row in summary['stages'])
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('END', json.dumps(record), flush=True)
        print((output / (name + '.log')).read_text()[-8000:], flush=True)
        if record['exit_code']:
            return record['exit_code']
    summary['complete'] = True
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
