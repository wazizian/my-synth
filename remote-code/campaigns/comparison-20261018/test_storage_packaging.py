"""GPU-worker-only import/compile checks for fresh copied storage launchers."""
import importlib.util
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('candidate_harden', root / 'harden_export.storage_candidate.py')
harden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harden)
from enable_handoff import enable

with tempfile.TemporaryDirectory() as temporary:
    folder = Path(temporary)
    # The candidate expects canonical filenames beside itself.
    shutil.copy2(root / 'export_safety.storage_candidate.py', root / 'export_safety.py')
    for handoff in (False, True):
        launch = folder / str(handoff)
        launch.mkdir()
        for name in ('job.py', 'smoke.py', 'USAGE.txt'):
            shutil.copy2(root / 'test-launcher-template' / name, launch / name)
        harden.harden(launch)
        if handoff:
            enable(launch)
        for name in ('job.py', 'smoke.py', 'export_safety.py', 'controller_handoff.py',
                     'external_artifacts.py', 'external_artifact_worker.py'):
            py_compile.compile(str(launch / name), doraise=True)
        subprocess.run([sys.executable, '-c',
                        'import job, smoke, export_safety, controller_handoff, external_artifacts'],
                       cwd=launch, check=True)
print('fresh hardened launcher dependencies import and compile with/without handoff', flush=True)
