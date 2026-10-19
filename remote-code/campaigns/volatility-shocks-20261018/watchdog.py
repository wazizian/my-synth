"""Independent systemd-owned rental deadline, with bounded export recovery."""
import datetime,json,subprocess,time,shlex,sys,hashlib
sys.path.insert(0,"/home/jump/vast-tools");import smoke
from pathlib import Path
C=Path(__file__).resolve().parent
DEADLINE=datetime.datetime(2026,10,19,6,5,tzinfo=datetime.timezone.utc).timestamp()
while True:
 now=time.time()
 for l in (C/'launchers').iterdir():
  active=l/'active.json'
  if not active.exists():continue
  saved=json.loads(active.read_text());matches=[]
  for r in (l/'runs').glob('*'):
   m=r/'metadata.json'
   if m.exists() and json.loads(m.read_text()).get('instance',{}).get('instance_id')==saved['instance_id']:matches.append(r)
  due=DEADLINE
  if matches:
   r=max(matches);metadata=json.loads((r/'metadata.json').read_text());start=datetime.datetime.strptime(r.name[:16],'%Y%m%dT%H%M%SZ').replace(tzinfo=datetime.timezone.utc).timestamp();due=min(due,start+metadata['timeout_seconds']+1200)
  if now<due:continue
  print('Deadline cleanup',l.name,saved['instance_id'],flush=True)
  exported=False
  if matches:
   ssh=metadata['ssh'];out=r/'outputs';out.mkdir(exist_ok=True)
   cmd=['rsync','-rlt','--safe-links','--partial','--delay-updates','--exclude=*.tmp*','--timeout=30','-e',shlex.join(ssh[:-1]),ssh[-1]+':/workspace/job/outputs/',str(out)+'/']
   try:
    result=subprocess.run(cmd,timeout=120,check=False)
    exported=result.returncode==0
    if exported:
     inventory=[]
     for p in out.rglob('*'):
      if p.is_file() and '.tmp' not in p.name:inventory.append({'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()})
     (r/'deadline-artifacts.json').write_text(json.dumps(inventory,indent=2))
   except subprocess.TimeoutExpired:pass
  if not exported:
   print('Export incomplete; stop compute and preserve worker disk',saved['instance_id'],flush=True)
   try:smoke.api('PUT',f"instances/{saved['instance_id']}/",{'state':'stopped'})
   except Exception as exc:print('STOP FAILED',str(exc),flush=True)
   continue
  subprocess.run(['/home/jump/.local/bin/uv','run','--no-project','python',str(l/'smoke.py'),'--cleanup'],timeout=120,check=False)
 if now>DEADLINE+600:break
 time.sleep(30)
