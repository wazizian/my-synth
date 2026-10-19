"""Freeze all selected checkpoints, then package evaluation without opening its paths."""
from pathlib import Path
import base64,hashlib,json,shutil,datetime
c=Path(__file__).resolve().parent
selected=c/'selected';selected.mkdir(exist_ok=True)
choices={}
for arm in ('baseline','easy','hard'):
    runs=[p for p in (c/'launchers'/arm/'runs').glob('*') if (p/'outputs/learned/complete.json').exists()]
    if len(runs)!=1:raise RuntimeError(f'Need exactly one completed {arm} run; got {len(runs)}')
    run=runs[0];root=run/'outputs/learned';selection=json.loads((root/'selection.json').read_text())
    shutil.copyfile(root/'selected.eqx',selected/(arm+'.eqx'));shutil.copyfile(root/'selection.json',selected/(arm+'-selection.json'))
    for p in root.glob('*/progress.json'):shutil.copyfile(p,selected/(arm+'-'+p.parent.name+'-progress.json'))
    for p in root.glob('*/selection-*.npz'):shutil.copyfile(p,selected/(arm+'-'+p.parent.name+'-'+p.name))
    shutil.copyfile(root/'initial-selection.npz',selected/(arm+'-initial-selection.npz'))
    actual=root/Path(selection['selected']['checkpoint']).parent.name/'best.eqx'
    if actual.read_bytes()!=(selected/(arm+'.eqx')).read_bytes():raise RuntimeError('Selected checkpoint byte mismatch')
    choices[arm]=dict(run=str(run),selection=selection,checkpoint_sha256=hashlib.sha256((selected/(arm+'.eqx')).read_bytes()).hexdigest())
freeze=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),evaluation_opened=False,choices=choices)
(c/'frozen_selection.json').write_text(json.dumps(freeze,indent=2)+'\n')
files={str(p.relative_to(c/'source')):p.read_text() for p in sorted((c/'source').rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json')}
blobs={p.name:base64.b64encode(p.read_bytes()).decode() for p in sorted(selected.iterdir()) if p.is_file()}
worker='''import os,pathlib,sys,json,base64,hashlib,traceback
root=pathlib.Path(os.environ['JOB_OUTPUT_DIR']);src=root/'source';src.mkdir(exist_ok=True);selected=root/'selected';selected.mkdir(exist_ok=True)
FILES=PAYLOAD
BLOBS=BINARY
for n,v in FILES.items():
 p=src/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(v)
for n,v in BLOBS.items():(selected/n).write_bytes(base64.b64decode(v))
(root/'source_hashes.json').write_text(json.dumps({n:hashlib.sha256(v.encode()).hexdigest() for n,v in FILES.items()},indent=2))
(root/'frozen_selection.json').write_text(FROZEN)
sys.path.insert(0,str(src))
try:
 from credit_hedging.learned_evaluation import main
 main(src,root/'evaluation',selected)
 from credit_hedging.learned_report import main as report
 report(root/'evaluation',selected)
 manifest=[]
 for p in sorted(root.rglob('*')):
  if p.is_file() and (p.relative_to(root).parts[0] in ('evaluation','selected','source') or p.name in ('source_hashes.json','frozen_selection.json')):
   h=hashlib.sha256()
   with p.open('rb') as f:
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
   manifest.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=h.hexdigest()))
 (root/'scientific_manifest.json').write_text(json.dumps(manifest,indent=2))
 (root/'complete.json').write_text(json.dumps({'status':'evaluation_and_report_complete','files':len(manifest)}))
 print('Evaluation, report and scientific manifest complete',flush=True)
except BaseException:
 (root/'failure.json').write_text(json.dumps({'traceback':traceback.format_exc()}))
 raise
'''.replace('PAYLOAD',repr(files)).replace('BINARY',repr(blobs)).replace('FROZEN',repr(json.dumps(freeze,indent=2)))
(c/'evaluation_worker.py').write_text(worker);compile(worker,str(c/'evaluation_worker.py'),'exec')
launcher=c/'launchers/evaluation';launcher.mkdir(exist_ok=True)
for name in ('job.py','smoke.py','requirements-jax.txt','example_jax.py'):shutil.copyfile(Path('/home/jump/vast-tools')/name,launcher/name)
print(json.dumps(dict(ready=True,worker_sha256=hashlib.sha256(worker.encode()).hexdigest(),choices={k:v['checkpoint_sha256'] for k,v in choices.items()})))
