"""Control-plane packaging only: frozen source/checkpoint bytes, no project imports."""
from pathlib import Path
import base64
import gzip
import hashlib
import json
import sys

C=Path(__file__).resolve().parent

def package(manifest_path,destination):
    manifest=json.loads(Path(manifest_path).read_text());payload={}
    source=Path(manifest['source']);manifest['source']='INPUT/source'
    if manifest['benchmark']=='credit':
        for p in (source/'credit_hedging').rglob('*.py'):payload['source/'+str(p.relative_to(source))]=p.read_bytes()
        for name in ['numerical_summary.json','learned_design.json']:
            if (source/name).exists():payload['source/'+name]=(source/name).read_bytes()
    else:
        payload['source/model.py']=(source/'model.py').read_bytes()
        payload['source/initial_premiums.json']=json.dumps(manifest['initial_premiums']).encode()
    for name in ['final_evaluation.py','option_adapter.py','adaptive.py']:payload['code/'+name]=(C/name).read_bytes()
    for arm,spec in manifest['models'].items():
        for key in ['checkpoint','selection']:
            original=Path(spec[key]);relative='models/'+arm+'/'+original.name
            payload[relative]=original.read_bytes();spec[key]='INPUT/'+relative
    new_curves={}
    for arm,paths in manifest.get('curves',{}).items():
        new_curves[arm]=[]
        for i,p in enumerate(paths):
            relative=f'curves/{arm}/{i}.jsonl';payload[relative]=Path(p).read_bytes();new_curves[arm].append('INPUT/'+relative)
    manifest['curves']=new_curves
    for trial in manifest.get('trial_curves',[]):
        for category in ['selection','monitor']:
            for i,row in enumerate(trial.get(category,[])):
                original=Path(row['path']);relative=f"trial-curves/{trial['arm']}/lr-{trial['lr']}/{category}/{original.name}"
                payload[relative]=original.read_bytes();row['path']='INPUT/'+relative
    if isinstance(manifest.get('discovery_freeze'),str):
        payload['discovery-freeze.json']=Path(manifest['discovery_freeze']).read_bytes();manifest['discovery_freeze']='INPUT/discovery-freeze.json'
    for name,path in manifest.get('static_addenda',{}).items():
        payload['addenda/'+name]=Path(path).read_bytes()
    source_hashes={k:hashlib.sha256(v).hexdigest() for k,v in payload.items()}
    raw_bytes=sum(map(len,payload.values()))
    if raw_bytes>100_000_000:raise RuntimeError('Evaluation payload exceeds100MB bounded artifact policy')
    encoded={k:base64.b64encode(gzip.compress(v)).decode() for k,v in payload.items()}
    worker=r'''import os,pathlib,json,gzip,base64,sys,traceback,hashlib
root=pathlib.Path(os.environ['JOB_OUTPUT_DIR']);root.mkdir(exist_ok=True)
inputs=pathlib.Path('/workspace/evaluation-input');inputs.mkdir(exist_ok=True)
PAYLOAD=__PAYLOAD__
MANIFEST=__MANIFEST__
for name,blob in PAYLOAD.items():
 p=inputs/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(gzip.decompress(base64.b64decode(blob)))
manifest=json.loads(json.dumps(MANIFEST).replace('INPUT/',str(inputs)+'/'))
p=inputs/'manifest.json';p.write_text(json.dumps(manifest,indent=2))
(root/'evaluation-source-hashes.json').write_text(json.dumps({k:hashlib.sha256((inputs/k).read_bytes()).hexdigest() for k in PAYLOAD},indent=2))
sys.path[:0]=[str(inputs/'code'),str(inputs/'source')]
try:
 import final_evaluation
 final_evaluation.main(p,root/'evaluation')
 if manifest.get('static_addenda'):
  import shutil,subprocess
  for name in manifest['static_addenda']:
   shutil.copy2(inputs/'addenda'/name,root/'evaluation'/name)
   if name.endswith('.md'):
    report=root/'evaluation/report.md';report.write_text(report.read_text()+chr(10)+'[Scientific limitations]('+name+')'+chr(10))
   if name.endswith('.tex'):
    report=root/'evaluation/report.tex';report.write_text(report.read_text().replace(chr(92)+'end{document}',chr(92)+'input{'+name+'}'+chr(10)+chr(92)+'end{document}'))
  latex=shutil.which('pdflatex')
  if latex:
   with (root/'evaluation/latex-build.log').open('a') as log:subprocess.run([latex,'-interaction=nonstopmode','-halt-on-error','report.tex'],cwd=root/'evaluation',stdout=log,stderr=log,timeout=60,check=False)
except BaseException:
 (root/'failure.json').write_text(json.dumps({'traceback':traceback.format_exc()}));raise
'''.replace('__PAYLOAD__',repr(encoded)).replace('__MANIFEST__',repr(manifest))
    Path(destination).write_text(worker)
    Path(str(destination)+'.manifest.json').write_text(json.dumps(dict(source_hashes=source_hashes,uncompressed_bytes=raw_bytes,frozen_manifest=manifest),indent=2))
    print(json.dumps(dict(worker=str(destination),uncompressed_bytes=raw_bytes,files=len(payload))))

if __name__=='__main__':package(sys.argv[1],sys.argv[2])
