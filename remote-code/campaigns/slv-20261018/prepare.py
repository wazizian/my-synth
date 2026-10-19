"""Control-plane packaging only; does not run numerical work."""
from pathlib import Path
import json, hashlib, ast
root=Path(__file__).resolve().parent
source=(root/'worker.py').read_text(); design=(root/'design.json').read_text()
source=source.replace("json.loads('__DESIGN_JSON__')",'json.loads('+repr(design)+')')
ast.parse(source)
(root/'worker_packaged.py').write_text(source)
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'design.json',root/'worker.py',root/'worker_packaged.py',Path('/home/jump/plan.md'),Path('/home/jump/deep-hedging/PLAN.md')]}
(root/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
print(json.dumps(hashes,indent=2))
