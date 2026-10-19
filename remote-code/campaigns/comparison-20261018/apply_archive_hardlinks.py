"""Apply approved metadata-verified immutable archive deduplication."""
from pathlib import Path
import datetime,json,os,stat
root=Path(__file__).resolve().parent
proposal=json.loads((root/'archive-hardlink-proposal.json').read_text());audit=root/'archive-hardlink-audit.jsonl'
if audit.exists():raise RuntimeError('An audit already exists; inspect before resuming')
def snapshot(path):
 s=path.lstat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,mode=stat.S_IMODE(s.st_mode),nlink=s.st_nlink,blocks=s.st_blocks,regular=stat.S_ISREG(s.st_mode))
def matches(actual,expected):return actual['regular'] and all(actual[k]==expected[k] for k in ('device','inode','bytes','mtime_ns'))
before=os.statvfs(root);count=0;skipped=0
with audit.open('x') as stream:
 for op in proposal['operations']:
  source=Path(op['canonical']['path']);target=Path(op['duplicate']['path']);first=snapshot(source);second=snapshot(target);row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=str(source),target=str(target),sha256_from_preverified_manifests=op['sha256'],source_before=first,target_before=second)
  if not matches(first,op['canonical']) or not matches(second,op['duplicate']) or first['device']!=second['device']:
   row['status']='skipped_changed_metadata';skipped+=1
  else:
   temporary=target.with_name(target.name+'.approved-hardlink.tmp')
   if temporary.exists():raise RuntimeError('Unexpected temporary hardlink:'+str(temporary))
   os.link(source,temporary)
   try:
    # Recheck directly before replacing the only duplicate inode link.
    if not matches(snapshot(source),op['canonical']) or not matches(snapshot(target),op['duplicate']):
     row['status']='skipped_changed_before_replace';skipped+=1
    else:
     os.replace(temporary,target);after=snapshot(target)
     if after['inode']!=first['inode'] or after['device']!=first['device'] or after['bytes']!=second['bytes']:raise RuntimeError('Atomic hardlink verification failed')
     row.update(status='linked_identical_preverified_content',target_after=after);count+=1
   finally:temporary.unlink(missing_ok=True)
  stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
after=os.statvfs(root)
summary={'approved_by':'Parent/root review of exact836-operation proposal','linked':count,'skipped':skipped,'free_bytes_before':before.f_bavail*before.f_frsize,'free_bytes_after':after.f_bavail*after.f_frsize,'freed_bytes':after.f_bavail*after.f_frsize-before.f_bavail*before.f_frsize,'scientific_payload_read_or_hashed':False,'unique_files_changed':False,'immutability_rule':'Linked legacy archives are immutable. Before any future edit, copy the individual path to a fresh inode and atomically replace it; never edit linked content in place.','audit':str(audit)}
(root/'archive-hardlink-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
