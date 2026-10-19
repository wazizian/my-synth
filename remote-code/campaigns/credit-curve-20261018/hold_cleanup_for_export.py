"""Bounded supervisor hold of this campaign's failed launcher during export."""
from pathlib import Path
import json
import os
import signal
import time

root=Path(__file__).resolve().parent
target='/home/jump/deep-hedging/campaigns/credit-curve-20261018/economic_worker.py'
found=[]
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():continue
    try:
        argv=(entry/'cmdline').read_bytes().split(b'\0')
    except (OSError,PermissionError):continue
    if (argv and Path(os.fsdecode(argv[0])).name.startswith('python')
        and b'/home/jump/vast-tools/job.py' in argv and target.encode() in argv):
        found.append(int(entry.name))
if len(found)!=1:
    raise RuntimeError(f'Expected exactly one specific campaign launcher, found {found}')
pid=found[0]
record=dict(pid=pid,started_at=time.time(),reason='Failed hard price-table build; hold cleanup while completed raw datasets export',maximum_seconds=1800)
(root/'cleanup_hold.json').write_text(json.dumps(record,indent=2))
os.kill(pid,signal.SIGSTOP)
print('Holding launcher cleanup; existing rsync child continues:',pid,flush=True)
try:
    deadline=time.monotonic()+1800
    while time.monotonic()<deadline:
        p=root/'export_verification.json'
        if p.exists() and json.loads(p.read_text()).get('complete'):
            record['export_verified']=True
            break
        time.sleep(10)
finally:
    os.kill(pid,signal.SIGCONT)
    record['resumed_at']=time.time()
    (root/'cleanup_hold.json').write_text(json.dumps(record,indent=2))
    print('Launcher resumed for final sync and confirmed shutdown',flush=True)
