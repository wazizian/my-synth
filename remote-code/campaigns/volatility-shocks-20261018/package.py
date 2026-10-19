from pathlib import Path
import hashlib,json
c=Path(__file__).resolve().parent
files={p.name:p.read_text() for p in (c/'model.py',c/'diagnostic.py',c/'evaluate.py',c/'DESIGN.md')}
manifest={n:hashlib.sha256(s.encode()).hexdigest() for n,s in files.items()}
(c/'source_hashes.json').write_text(json.dumps(manifest,indent=2))
body='''import os,sys,pathlib,json,traceback,hashlib
root=pathlib.Path(os.environ['JOB_OUTPUT_DIR']);src=root/'source';src.mkdir(exist_ok=True)
FILES=PAYLOAD
for name,text in FILES.items():(src/name).write_text(text)
(root/'source_hashes.json').write_text(json.dumps({n:hashlib.sha256(s.encode()).hexdigest() for n,s in FILES.items()},indent=2))
sys.path.insert(0,str(src))
import jax
assert jax.default_backend()=='gpu'
mode=sys.argv[1]
try:
 if mode=='diagnostic':
  import diagnostic;diagnostic.main(root/'diagnostic')
 elif mode in ('baseline','easy','hard'):
  import model;model.train(root/'training',mode)
 elif mode=='evaluation':
  import base64
  MODELS=MODEL_PAYLOAD
  for name,data in MODELS.items():
   p=root/'models'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(data))
  import evaluate;evaluate.main(root/'evaluation',root/'models',DIAGNOSTIC_PAYLOAD)
 else:raise ValueError(mode)
except BaseException:
 (root/'failure.json').write_text(json.dumps(dict(traceback=traceback.format_exc())))
 raise
'''.replace('PAYLOAD',repr(files),1)
# Evaluation payload inserted by supervisor after training completes.
body=body.replace('MODEL_PAYLOAD','{}').replace('DIAGNOSTIC_PAYLOAD','{}')
(c/'worker.py').write_text(body)
compile(body,str(c/'worker.py'),'exec')
print('Packaged',manifest)
