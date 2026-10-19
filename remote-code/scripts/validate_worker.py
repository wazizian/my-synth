"""Initial rented-GPU correctness, market preparation and profiling gate."""
import os,sys,json,time,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from src.runtime import configure
configure(root)
os.environ['PYTHONPATH']=str(root)
results=[]
for command in ([sys.executable,'-m','pytest','tests','-q'],[sys.executable,'scripts/prepare_market.py'],[sys.executable,'scripts/prepare_data.py'],[sys.executable,'scripts/smoke.py']):
    start=time.time();print('RUN',command,flush=True)
    result=subprocess.run(command,cwd=root)
    results.append({'command':command,'seconds':time.time()-start,'returncode':result.returncode})
    (root/'validation.json').write_text(json.dumps(results,indent=2))
    if result.returncode:raise SystemExit(result.returncode)
print('VALIDATION_AND_DATA_COMPLETE',flush=True)
