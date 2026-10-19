"""Stream an immutable worker handoff; never load numerical arrays locally."""
import argparse,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');p.add_argument('--include-runs',action='store_true');a=p.parse_args();r=Path(a.source)
with tarfile.open(a.destination,'w:gz') as t:
    for name in ['data','cache/pricing','objective_scales.json']+(['runs','results/frozen_selection.json'] if a.include_runs else []):
        path=r/name
        if path.exists():t.add(path,arcname=name)
print(a.destination)
