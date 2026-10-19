"""Real GPU-worker SIGTERM/checkpoint check on the retained baseline smoke."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from src.runtime import configure
configure(root)
run=root/'runs/smoke_baseline';before=json.loads((run/'status.json').read_text())
old=json.loads((run/'checkpoints/latest.json').read_text())
command=[sys.executable,'scripts/run_trial.py','--root',str(root),'--config',str(run/'config.json'),'--active-seconds',str(before['active_training_seconds']+90)]
with (run/'termination_check.log').open('w') as log:
    child=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    sent=False;end=time.time()+240
    while child.poll() is None and time.time()<end:
        status=json.loads((run/'status.json').read_text())
        if status['learner_steps']>before['learner_steps'] and status['status']=='running':
            child.send_signal(signal.SIGTERM);sent=True;break
        time.sleep(.5)
    try:code=child.wait(timeout=180)
    except subprocess.TimeoutExpired:
        child.kill();child.wait();raise RuntimeError('SIGTERM checkpoint deadline exceeded')
after=json.loads((run/'status.json').read_text());latest=json.loads((run/'checkpoints/latest.json').read_text())
from src.runner import _skeleton
from src.checkpoint import load_checkpoint
config=json.loads((run/'config.json').read_text());skeleton,_,_=_skeleton(config)
state,metadata=load_checkpoint(run/'checkpoints',skeleton)
passed=bool(sent and code==0 and after['status']=='stopped' and latest['checkpoint_id']!=old['checkpoint_id'] and after['learner_steps']>before['learner_steps'])
report={'passed':passed,'signal_sent':sent,'returncode':code,'before_steps':before['learner_steps'],'after_steps':after['learner_steps'],'checkpoint_id':metadata['checkpoint_id'],'status':after['status']}
(root/'results/termination_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if not passed:raise SystemExit('Graceful termination validation failed')
