"""Metadata-only duplicate export proposal; never reads numerical payloads."""
from pathlib import Path
import collections,json,stat
root=Path(__file__).resolve().parent
sources=[(Path('/home/jump/vast-tools/runs/20261018T132207Z-ecc52952'),'artifacts.json'),(Path('/home/jump/vast-tools/runs/20261018T135331Z-dd6fa036'),'recovery_artifacts.json')]
groups=collections.defaultdict(list);excluded=[]
for run,name in sources:
 manifest=run/name;rows=json.loads(manifest.read_text());created=manifest.stat().st_mtime_ns
 for row in rows:
  file=run/'outputs'/row['path']
  try:info=file.lstat()
  except FileNotFoundError:continue
  if not stat.S_ISREG(info.st_mode) or info.st_size!=row['bytes'] or info.st_mtime_ns>created:
   excluded.append(str(file));continue
  groups[(row['sha256'],row['bytes'])].append(dict(path=str(file),manifest=str(manifest),inode=info.st_ino,device=info.st_dev,mtime_ns=info.st_mtime_ns,bytes=info.st_size,blocks=info.st_blocks))
operations=[];savings=0
for (sha,size),files in groups.items():
 if len(files)<2:continue
 canonical=files[0];seen={(canonical['device'],canonical['inode'])}
 for duplicate in files[1:]:
  inode=(duplicate['device'],duplicate['inode'])
  if inode in seen or duplicate['device']!=canonical['device']:continue
  seen.add(inode);operations.append(dict(canonical=canonical,duplicate=duplicate,sha256=sha,bytes=size));savings+=duplicate['blocks']*512
proposal={'basis':'Matching preverified SHA256 and byte-size export manifests; current regular-file sizes match; current mtimes not newer than verification manifests; immutable completed run exports. No payload read/hash locally.','operations':operations,'eligible_files':len(operations),'estimated_freed_bytes':savings,'excluded_metadata_mismatch':excluded,'unique_files_untouched':True,'requires_parent_review':True}
(root/'archive-hardlink-proposal.json').write_text(json.dumps(proposal,indent=2));print(json.dumps({k:v for k,v in proposal.items() if k!='operations'}))
