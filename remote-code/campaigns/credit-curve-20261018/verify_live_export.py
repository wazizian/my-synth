"""Verify immutable local exports, then release the worker's shutdown guard."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

campaign=Path(__file__).resolve().parent
run=Path('/home/jump/vast-tools/runs/20261018T132207Z-ecc52952')
ssh=json.loads((run/'metadata.json').read_text())['ssh']
verified={}
deadline=time.monotonic()+8000
manifest=None
while time.monotonic()<deadline:
    if manifest is None:
        result=subprocess.run(ssh+['test -f /workspace/job/outputs/export_ready.json && cat /workspace/job/outputs/export_ready.json'],capture_output=True,text=True,timeout=30)
        if result.returncode==0:
            manifest=json.loads(result.stdout)
            (campaign/'export_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            print('Manifest received:',len(manifest['files']),'files',flush=True)
        else:
            time.sleep(30)
            continue
    missing=[]
    for row in manifest['files']:
        path=run/'outputs'/row['path']
        if not path.exists() or path.stat().st_size!=row['bytes']:
            missing.append(row['path']);continue
        stamp=(path.stat().st_size,path.stat().st_mtime_ns)
        if verified.get(row['path'])==stamp:
            continue
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):
                digest.update(block)
        if digest.hexdigest()==row['sha256']:
            verified[row['path']]=stamp
        else:
            missing.append(row['path'])
    record=dict(checked_at=time.time(),verified_files=len(verified),total_files=len(manifest['files']),
                pending_files=missing,complete=not missing)
    tmp=campaign/'export_verification.tmp.json'
    tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(campaign/'export_verification.json')
    print('Verified',len(verified),'of',len(manifest['files']),'pending',len(missing),flush=True)
    if not missing:
        result=subprocess.run(ssh+['cat > /workspace/job/outputs/export_ack.tmp.json && mv /workspace/job/outputs/export_ack.tmp.json /workspace/job/outputs/export_ack.json'],input=json.dumps(record),text=True,capture_output=True,timeout=30)
        if result.returncode:
            raise RuntimeError('Failed to acknowledge verified export: '+result.stderr)
        print('Export acknowledged; normal launcher completion and shutdown released',flush=True)
        break
    time.sleep(30)
else:
    raise RuntimeError('Export verification deadline exceeded; no acknowledgement sent')
