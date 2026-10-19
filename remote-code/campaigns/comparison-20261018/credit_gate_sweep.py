"""Control-only bounded gate schedule and pre-workload infrastructure recovery."""
from pathlib import Path
import time,subprocess,json,fcntl,re,sys
sys.path.insert(0,'/home/jump/vast-tools')
import smoke
root=Path(__file__).resolve().parent;base=root.parent
lock=(root/'gate-sweep.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
pending=[('credit-high-default-20261018',3),('credit-ordinary-long-20261018',10),('credit-high-default-20261018',10)];children=[];blocked=[]
def recover_command(campaign,bp,gate):
 launch=gate/'launcher'
 if (launch/'active.json').exists():return None
 runs=list((launch/'runs').glob('*'))
 if any((r/'workload-launch-intent.json').exists() for r in runs):
  blocked.append(dict(campaign=campaign,bp=bp,reason='Workload was launched; preserve scientific failure and require explicit review'))
  return None
 state=gate/'infrastructure-retries.json';record=json.loads(state.read_text()) if state.exists() else {'attempts':0}
 if record['attempts']>=2:
  blocked.append(dict(campaign=campaign,bp=bp,reason='Two bounded pre-workload infrastructure retries exhausted'))
  return None
 previous_ids=set()
 exclusions={53061818,48448052,55917102,56219704,55701183}
 for run in runs:
  meta=run/'metadata.json'
  if meta.exists():
   instance=json.loads(meta.read_text()).get('instance',{});offer=instance.get('offer')
   if instance.get('instance_id'):previous_ids.add(instance['instance_id'])
   if offer:exclusions.add(int(offer))
 for log in gate.glob('*launcher.log'):
  content=log.read_text();exclusions.update(int(value) for value in re.findall(r'Offer (\d+):',content));previous_ids.update(int(value) for value in re.findall(r'Created instance (\d+)',content))
 if previous_ids:
  remaining=[row for row in smoke.list_instances() if row.get('id') in previous_ids and row.get('actual_status')!='destroyed']
  if remaining:
   blocked.append(dict(campaign=campaign,bp=bp,reason='Provider still reports prior instance; no duplicate rental',instance_ids=[row['id'] for row in remaining]));return None
 record['attempts']+=1;record['excluded_offers']=sorted(exclusions);state.write_text(json.dumps(record,indent=2))
 cmd=['/home/jump/.local/bin/uv','run','--no-project','python',str(launch/'job.py'),'--job',str(gate/'worker.py'),'--requirements',str(base/campaign/'requirements.txt'),'--timeout-seconds','7200','--max-hourly','0.60','--require-artifact','gate/complete.json','--require-artifact','gate/results.json']
 for offer in sorted(exclusions):cmd.extend(['--exclude-offer',str(offer)])
 return cmd,gate/f"infrastructure-retry-{record['attempts']}-launcher.log"
while pending or children:
 for campaign,bp in pending[:]:
  ready=all(any((r/f'outputs/cost-{bp}bp/learned/selection.json').exists() and (r/f'outputs/cost-{bp}bp/learned/selected.eqx').exists() for r in (base/campaign/'launchers'/arm/'runs').glob('*')) for arm in ('baseline','easy','hard'))
  if not ready:continue
  gate=root/(campaign+f'-gate-{bp}bp')
  if gate.exists():
   complete=any((r/'outputs/gate/complete.json').exists() and (r/'artifacts.json').exists() for r in (gate/'launcher/runs').glob('*'))
   if complete or (gate/'launcher/active.json').exists():pending.remove((campaign,bp));continue
   recovery=recover_command(campaign,bp,gate)
   if recovery is None:pending.remove((campaign,bp));continue
   command,logpath=recovery
  else:
   command=['/usr/bin/python',str(root/'launch_credit_gates.py'),'--bp',str(bp),'--campaign',campaign];logpath=root/(campaign+f'-gate-{bp}bp-orchestration.log')
  log=logpath.open('a');p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);log.close();children.append((campaign,bp,p));pending.remove((campaign,bp))
 for item in children[:]:
  campaign,bp,p=item
  if p.poll() is not None:
   print(campaign,bp,p.returncode,flush=True);children.remove(item)
   if p.returncode:pending.append((campaign,bp))
 (root/'gate_sweep_status.json').write_text(json.dumps({'pending':pending,'running':[{'campaign':c,'bp':bp,'pid':p.pid} for c,bp,p in children],'blocked':blocked},indent=2))
 time.sleep(30)
