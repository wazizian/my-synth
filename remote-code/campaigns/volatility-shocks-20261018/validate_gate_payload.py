"""GPU-host-only stdlib validation of generated discovery payload; no simulator import."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
tree = ast.parse(text)
compile(tree, str(path), 'exec')
values = {}
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in ('source', 'MODELS', 'expected_hashes'):
                values[target.id] = ast.literal_eval(node.value)
assert set(values['MODELS']) == {'baseline', 'easy', 'hard'}
compile(values['source'], 'model.py', 'exec')
rows = {}
for arm, encoded in values['MODELS'].items():
    raw = base64.b64decode(encoded, validate=True)
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == values['expected_hashes'][arm], arm
    rows[arm] = {'bytes': len(raw), 'sha256': digest}
result = {'status': 'passed', 'worker_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
          'checkpoint_bytes_exact': rows, 'no_market_or_numerical_imports': True}
output = path.with_name('payload-validation.json')
output.write_text(json.dumps(result, indent=2))
print(json.dumps(result))
