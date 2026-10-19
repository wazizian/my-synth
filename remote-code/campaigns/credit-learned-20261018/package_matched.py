from pathlib import Path
import json,base64,hashlib,datetime,shutil
c=Path(__file__).resolve().parent
prior=json.loads((c/'frozen_selection.json').read_text());lr=prior['choices']['baseline']['selection']['selected']['lr'];blobs={};choices={}
for arm in ('baseline','easy','hard'):
    run=Path(prior['choices'][arm]['run']);s=json.loads((run/'outputs/learned/selection.json').read_text());trial=next(t for t in s['trials'] if t['lr']==lr);p=run/'outputs/learned'/f'lr-{lr:g}'/'best.eqx'
    blobs[arm+'.eqx']=base64.b64encode(p.read_bytes()).decode();choices[arm]=dict(checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),trial=trial)
freeze=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),common_learning_rate=lr,chosen_from='Baseline development selection frozen before original evaluation',original_selection_frozen_at=prior['frozen_at'],analysis_scope='Additional matched-rate protocol check on same evaluation paths. No retraining or evaluation-driven hyperparameter selection.',choices=choices)
(c/'matched_freeze.json').write_text(json.dumps(freeze,indent=2)+'\n');blobs['matched_freeze.json']=base64.b64encode(json.dumps(freeze).encode()).decode()
p=next((c/'launchers/evaluation/runs').glob('*/outputs/evaluation/terminal_and_static_selection.npz'));blobs['original_terminal.npz']=base64.b64encode(p.read_bytes()).decode()
files={str(p.relative_to(c/'source')):p.read_text() for p in (c/'source').rglob('*') if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in p.parts}
worker='''import os,pathlib,sys,json,hashlib,base64,traceback
root=pathlib.Path(os.environ['JOB_OUTPUT_DIR']);src=root/'source';src.mkdir(exist_ok=True);checkpoints=root/'selected';checkpoints.mkdir(exist_ok=True)
FILES=PAYLOAD
BLOBS=BINARY
for n,v in FILES.items():
 p=src/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(v)
for n,v in BLOBS.items():(checkpoints/n).write_bytes(base64.b64decode(v))
(root/'source_hashes.json').write_text(json.dumps({n:hashlib.sha256(v.encode()).hexdigest() for n,v in FILES.items()},indent=2))
sys.path.insert(0,str(src))
try:
 from credit_hedging.matched_evaluation import main
 main(src,root/'matched',checkpoints)
 manifest=[]
 for p in sorted(root.rglob('*')):
  if p.is_file() and p.relative_to(root).parts[0] in ('matched','selected','source'):
   manifest.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 (root/'scientific_manifest.json').write_text(json.dumps(manifest,indent=2))
 (root/'complete.json').write_text(json.dumps({'status':'matched_evaluation_complete'}))
except BaseException:
 (root/'failure.json').write_text(json.dumps({'traceback':traceback.format_exc()}));raise
'''.replace('PAYLOAD',repr(files)).replace('BINARY',repr(blobs))
(c/'matched_worker.py').write_text(worker);compile(worker,'matched_worker.py','exec')
p=c/'launchers/matched';p.mkdir(exist_ok=True)
for n in ('job.py','smoke.py','requirements-jax.txt','example_jax.py'):shutil.copyfile(Path('/home/jump/vast-tools')/n,p/n)
print(json.dumps(dict(common_lr=lr,checkpoints={k:v['checkpoint_sha256'] for k,v in choices.items()})))
