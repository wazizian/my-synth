"""Perform one bounded actual external export; control-plane orchestration only."""
import argparse
import json
from pathlib import Path
from external_artifacts import ensure_external

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--rows', type=Path, required=True)
parser.add_argument('--storage-fixture', action='store_true')
args = parser.parse_args()
metadata = json.loads((args.run / 'metadata.json').read_text())
rows = json.loads(args.rows.read_text())
ensure_external(args.run, metadata['ssh'], rows, fixture=args.storage_fixture)
print('Verified external export recorded:', args.run / 'external-artifacts.json')
