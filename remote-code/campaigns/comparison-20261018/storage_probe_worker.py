"""Tiny storage-control fixture; execute only on an existing GPU worker."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('action', choices=('create', 'verify'))
parser.add_argument('--restored', type=Path)
args = parser.parse_args()
root = Path('/workspace/artifact-storage-probe')
output = Path('/workspace/job/outputs/storage-proof')
output.mkdir(exist_ok=True)
source = root / 'source'
if args.action == 'create':
    source.mkdir(parents=True, exist_ok=True)
    data = (b'deep-hedging-private-volume-proof-20261018\n' * 30)[:1024]
    (source / 'fixture.bin').write_bytes(data)
    (source / 'manifest.json').write_text(json.dumps(dict(
        bytes=len(data), sha256=hashlib.sha256(data).hexdigest())))
    print((source / 'manifest.json').read_text())
else:
    restored = (args.restored or (root / 'restored')) / 'fixture.bin'
    expected = json.loads((source / 'manifest.json').read_text())
    actual = dict(bytes=restored.stat().st_size,
                  sha256=hashlib.sha256(restored.read_bytes()).hexdigest())
    if actual != expected:
        raise RuntimeError('Volume roundtrip SHA256 mismatch')
    proof = dict(passed=True, volume_id=56629013, source_instance_id=56617036,
                 source=str(source), restored=str(restored), **actual)
    (output / 'roundtrip.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof, indent=2))
