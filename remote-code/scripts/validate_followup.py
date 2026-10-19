"""Reuse the validated immutable table; validate fallback/data/integration."""
import json,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
subprocess.run([sys.executable,'scripts/commit_source.py'],cwd=root,check=True,stdout=subprocess.DEVNULL)
from src.runtime import configure
configure(root)
results=[]
for script in ('validate_fallback.py','prepare_data.py','smoke.py','check_termination.py'):
    started=time.time();print('RUN',script,flush=True)
    result=subprocess.run([sys.executable,'scripts/'+script],cwd=root)
    results.append({'script':script,'returncode':result.returncode,'seconds':time.time()-started})
    (root/'validation_followup.json').write_text(json.dumps(results,indent=2))
    if result.returncode:raise SystemExit(result.returncode)
started=time.time()
result=subprocess.run([sys.executable,'scripts/profile_families.py','--campaign-start','1792274040','--deadline','1792301040','--gpu-csv','/workspace/job/outputs/gpu.csv'],cwd=root)
results.append({'script':'profile_families.py','returncode':result.returncode,'seconds':time.time()-started})
(root/'validation_followup.json').write_text(json.dumps(results,indent=2))
if result.returncode:raise SystemExit(result.returncode)
print('FOLLOWUP_VALIDATION_COMPLETE',flush=True)
