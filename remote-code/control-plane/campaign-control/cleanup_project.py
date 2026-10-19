"""User-authorized artifact cleanup; retain code and all canonical metrics."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess
R=Path('/home/jump/deep-hedging');C=Path('/home/jump/campaign-control')
G='/home/jump/.local/bin/git'
assert not list((R/'workers').glob('*/active.json'))
tracked=[n for n in subprocess.check_output([G,'ls-files','-z'],cwd=R).decode().split('\0') if n]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
# Preserve compact numerical validation evidence from obsolete worker exports.
for w in sorted((R/'workers').glob('validation*')):
 for project in w.glob('runs/*/outputs/project'):
  for sub in ('results','cache/pricing'):
   for p in (project/sub).glob('*.json'):
    dest=R/'results/validation_history'/w.name/project.parents[1].name/p.relative_to(project)
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
for p in (R/'cache/pricing').glob('*.validation.json'):
 dest=R/'results/pricing_validation'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
# All tracked bytes and canonical result/run metric bytes must survive unchanged.
protected={R/n for n in tracked}
protected.update(p for p in (R/'results').rglob('*') if p.is_file() and 'worker_environments' not in p.parts)
for run in (R/'runs').iterdir():
 if run.is_dir():
  protected.update(p for p in run.iterdir() if p.is_file())
  protected.update(p for p in (run/'episode_metrics').rglob('*') if p.is_file())
protected.update(p for p in (R/'source_snapshots').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
before={str(p):digest(p) for p in sorted(protected)}
paths=[R/n for n in ('workers','cache','.venv','.pytest_cache','results/worker_environments')]
paths += [p for run in (R/'runs').iterdir() if run.is_dir() for p in (run/'checkpoints',run/'history')]
paths += list((R/'data').glob('*.npz'))+list((R/'queue').glob('*.tar.gz'))+list((R/'queue').glob('*.log'))+list((R/'queue').glob('*.lock'))+list(R.glob('*_job.py'))
paths += list(R.glob('**/__pycache__'))
paths += [R/'budget/lock',R/'budget/offers.json',R/'budget/instances_current.json',R/'budget/provider_charges_latest.json']+list((R/'budget').glob('account*'))
paths += [Path('/home/jump/campaign-link-base'),C/'thin_queue_archives',Path('/tmp/git-core.rpm')]
paths=[p for p in paths if p.exists() or p.is_symlink()]
paths=sorted(set(paths),key=lambda p:(len(p.parts),str(p)))
paths=[p for i,p in enumerate(paths) if not any(parent in p.parents for parent in paths[:i])]
for p in paths:
 assert p not in protected and not any(p in f.parents for f in protected),str(p)
 assert p!=R and p!=R/'.git'
files=0;logical=0;seen=set();unique=0
for p in paths:
 for f in ([p] if not p.is_dir() else p.rglob('*')):
  if f.is_file() and not f.is_symlink():
   st=f.stat();files+=1;logical+=st.st_size
   key=(st.st_dev,st.st_ino)
   if key not in seen:seen.add(key);unique+=st.st_size
free_before=shutil.disk_usage(R).free
manifest={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization':'User requested cleanup retaining Git code and metrics','status':'deleting','removed_paths':[str(p) for p in paths],'candidate_files':files,'candidate_logical_bytes':logical,'candidate_unique_inode_bytes':unique,'retained_hashed_file_count':len(before),'retained_content_hashes':before,'scope':'Keep Git/code including historical source snapshots, reports, figures, compact manifests and all canonical per-episode metrics. Historical checkpoint recoverability audits describe pre-cleanup state.'}
mp=R/'results/cleanup_manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
for p in paths:
 if p.is_dir() and not p.is_symlink():shutil.rmtree(p)
 else:p.unlink()
for p,sha in before.items():assert Path(p).is_file() and digest(Path(p))==sha,p
manifest.update(status='complete',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),free_bytes_before=free_before,free_bytes_after=shutil.disk_usage(R).free,retained_hashes_verified=True)
manifest['freed_bytes']=manifest['free_bytes_after']-free_before
mp.write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:manifest[k] for k in ('status','candidate_files','freed_bytes','retained_hashed_file_count','retained_hashes_verified')}))
