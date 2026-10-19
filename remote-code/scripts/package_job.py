"""Control-plane only: embed a source snapshot in one standalone rental job."""
import argparse,base64,io,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--entry',required=True);p.add_argument('--output',required=True);a=p.parse_args()
r=Path(__file__).resolve().parents[1]; buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as tf:
    for folder in ('src','tests','scripts','configs'):
        for f in sorted((r/folder).rglob('*')):
            if f.is_file() and '__pycache__' not in str(f) and f.resolve()!=Path(a.output).resolve():tf.add(f,arcname=str(f.relative_to(r)))
    tf.add(r/'pyproject.toml',arcname='pyproject.toml')
blob=base64.b64encode(buf.getvalue()).decode()
Path(a.output).write_text('import os,base64,io,tarfile,subprocess,sys\nfrom pathlib import Path\nr=Path(os.environ["JOB_OUTPUT_DIR"])/"project"\nr.mkdir(exist_ok=True)\nwith tarfile.open(fileobj=io.BytesIO(base64.b64decode('+repr(blob)+')),mode="r:gz") as t:t.extractall(r,filter="data")\nos.chdir(r)\nsubprocess.run([sys.executable,"-u",'+repr(a.entry)+']+sys.argv[1:],check=True)\n')
print(a.output)
