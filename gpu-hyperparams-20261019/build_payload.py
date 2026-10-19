"""Package source using only file IO/compression; do not import numerical code."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
files = sorted((root / 'neural_sde_hedging').glob('*.py')) + [root / 'tests/test_neural_sde_gpu.py']
manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, str(path.relative_to(root)))
    archive.writestr('SOURCE_SHA256.json', json.dumps(manifest, indent=2))
template = (Path(__file__).parent / 'worker_template.py').read_text()
template = template.replace('SOURCE_ZIP_BASE64', base64.b64encode(buffer.getvalue()).decode())
(Path(__file__).parent / 'worker.py').write_text(template)
(Path(__file__).parent / 'SOURCE_SHA256.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'source_files': len(files), 'payload_bytes': len(template)}))
