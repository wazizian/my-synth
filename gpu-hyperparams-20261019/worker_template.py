"""GPU validation worker, embedded source supplied by file-IO packager."""
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
    """Verify source, run GPU tests, then the bounded hyperparameter experiment."""
    output = Path(os.environ['JOB_OUTPUT_DIR'])
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
    commands = [
        ('pytest', [sys.executable, '-m', 'pytest', '-q', '-ra',
                    str(source / 'tests/test_neural_sde_gpu.py'),
                    '--junitxml=' + str(output / 'pytest.xml')], 600),
        ('hyperparameters', [sys.executable, '-u', '-m',
                             'neural_sde_hedging.hyperparameter_validation'], 10000),
    ]
    stages = []
    for name, command, timeout in commands:
        began = time.monotonic()
        print('START', name, flush=True)
        with (output / (name + '.log')).open('w') as stream:
            result = subprocess.run(command, cwd=source, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
        stages.append({'name': name, 'exit_code': result.returncode,
                       'seconds': time.monotonic() - began})
        (output / 'worker-stages.json').write_text(json.dumps(stages, indent=2))
        print('END', stages[-1], flush=True)
        print((output / (name + '.log')).read_text()[-8000:], flush=True)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    sys.exit(main())
