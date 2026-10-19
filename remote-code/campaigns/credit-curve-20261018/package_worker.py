"""Package versioned project sources into the launcher's single-file interface."""
from pathlib import Path
import hashlib
import json
import textwrap

campaign = Path(__file__).resolve().parent
source = campaign.parent.parent/'credit-hedging'
files = {str(p.relative_to(source)):p.read_text() for p in source.rglob('*.py')}
files['pyproject.toml'] = (source/'pyproject.toml').read_text()
files['design.json'] = (campaign/'design.json').read_text()
worker = '''import os,json,pathlib,sys,subprocess,hashlib
root=pathlib.Path(os.environ['JOB_OUTPUT_DIR'])
src=root/'source'
src.mkdir(exist_ok=True)
FILES = '''+repr(files)+'''
for name,body in FILES.items():
 p=src/name
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(body)
(root/'source_hashes.json').write_text(json.dumps({n:hashlib.sha256(b.encode()).hexdigest() for n,b in FILES.items()},indent=2))
import jax
jax.config.update('jax_enable_x64', True)
assert jax.default_backend() == 'gpu'
env=dict(os.environ,JAX_ENABLE_X64='true',PYTHONPATH=str(src))
with (root/'pytest.log').open('w') as log:
 r=subprocess.run([sys.executable,'-m','pytest','-q',str(src/'tests'),'--junitxml='+str(root/'tests.xml')],cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT)
(root/'validation.json').write_text(json.dumps({'exit_code':r.returncode,'backend':jax.default_backend(),'jax':jax.__version__}))
print((root/'pytest.log').read_text(),flush=True)
if r.returncode: sys.exit(r.returncode)
sys.path.insert(0,str(src))
from credit_hedging.numerical_validation import run as numerical_run
numerical=numerical_run(src/'design.json',root/'numerical')
if not all(v['passed'] for v in numerical.values()):
 raise RuntimeError('Numerical acceptance failed; economic outcomes remain unopened')
from credit_hedging.economic_screen import run as economic_run
result=economic_run(src/'design.json',root/'numerical/numerical_summary.json',root/'economic')
from credit_hedging.audit_economic import run as audit_run
audit_run(root/'economic')
from credit_hedging.economic_report import run as report_run
report_run(root/'economic')
(root/'economic_complete.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
'''
worker = ('import os,json,pathlib,sys,traceback\n'
          'def main():\n'+textwrap.indent(worker,'    ')+
          '\ntry:\n    main()\nexcept BaseException:\n'
          "    root=pathlib.Path(os.environ['JOB_OUTPUT_DIR'])\n"
          "    (root/'failure.json').write_text(json.dumps({'traceback':traceback.format_exc()},indent=2))\n"
          "    print(traceback.format_exc(),flush=True)\n"
          "    if not (root/'export_ready.json').exists():\n"
          "        sys.path.insert(0,str(root/'source'))\n"
          "        from credit_hedging.economic_report import await_export\n"
          "        await_export(root)\n"
          "    raise\n")
(campaign/'economic_worker.py').write_text(worker)
(campaign/'economic_source_hashes.json').write_text(json.dumps({n:hashlib.sha256(b.encode()).hexdigest() for n,b in files.items()},indent=2)+'\n')
print(hashlib.sha256(worker.encode()).hexdigest())
