from pathlib import Path
import hashlib,json,textwrap
c=Path(__file__).resolve().parent
files={str(p.relative_to(c/'source')):p.read_text() for p in sorted((c/'source').rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json')}
(c/'source_hashes.json').write_text(json.dumps({n:hashlib.sha256(v.encode()).hexdigest() for n,v in files.items()},indent=2)+'\n')
body='''import os,pathlib,sys,json,hashlib,subprocess
root=pathlib.Path(os.environ['JOB_OUTPUT_DIR']);src=root/'source';src.mkdir(exist_ok=True)
FILES=PAYLOAD
for n,v in FILES.items():
 p=src/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(v)
(root/'source_hashes.json').write_text(json.dumps({n:hashlib.sha256(v.encode()).hexdigest() for n,v in FILES.items()},indent=2))
import jax
jax.config.update('jax_enable_x64',True)
assert jax.default_backend()=='gpu'
sys.path.insert(0,str(src))
with (root/'pytest.log').open('w') as log:
 result=subprocess.run([sys.executable,'-m','pytest','-q',str(src/'tests'),'--junitxml='+str(root/'tests.xml')],cwd=src,env=dict(os.environ,JAX_ENABLE_X64='true',PYTHONPATH=str(src)),stdout=log,stderr=subprocess.STDOUT)
print((root/'pytest.log').read_text(),flush=True)
(root/'validation.json').write_text(json.dumps({'exit_code':result.returncode,'backend':jax.default_backend()}))
if result.returncode:raise RuntimeError('GPU validation failed')
from credit_hedging.learned_run import main
from dataclasses import replace
import gc
summary=json.loads((src/'numerical_summary.json').read_text())
original=json.loads(json.dumps(summary))
for cost in (.0001,.0003,.001):
 for domain in summary:
  summary[domain]['production_config']['transaction_cost']=cost
 (src/'numerical_summary.json').write_text(json.dumps(summary,indent=2))
 destination=root/('cost-'+str(round(cost*10000))+'bp')
 destination.mkdir(exist_ok=True)
 (destination/'configuration.json').write_text(json.dumps(summary,indent=2))
 main(src,destination/'learned',sys.argv[1])
 jax.clear_caches();gc.collect()
(root/'complete.json').write_text(json.dumps(dict(arm=sys.argv[1],costs=[.0001,.0003,.001],status='complete')))

'''.replace('PAYLOAD',repr(files))
worker='import traceback,pathlib,os,json\ndef main():\n'+textwrap.indent(body,'    ')+'''\ntry:
 main()
except BaseException:
 p=pathlib.Path(os.environ['JOB_OUTPUT_DIR'])/'failure.json'
 p.write_text(json.dumps({'traceback':traceback.format_exc()},indent=2))
 raise
'''
(c/'worker.py').write_text(worker)
compile(worker,str(c/'worker.py'),'exec')
print('Packaged',len(files),'source/config files',hashlib.sha256(worker.encode()).hexdigest())
