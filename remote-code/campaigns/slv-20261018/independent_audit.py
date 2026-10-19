"""Run only on GPU worker host; read-only scientific outputs, atomic audit reports."""
import os,time,json,hashlib,pathlib,subprocess
import numpy as np
from scipy.special import logsumexp
ROOT=pathlib.Path(os.environ.get('JOB_OUTPUT_DIR','/workspace/job/outputs'))
subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],check=True)
seen={}; failures=[]; maximum={}; checks=0

def check(name,error,tolerance,path):
    global checks
    checks+=1;error=float(error);maximum[name]=max(maximum.get(name,0),error)
    if not np.isfinite(error) or error>tolerance: failures.append(dict(file=str(path.relative_to(ROOT)),check=name,error=error,tolerance=tolerance))

def inspect(path):
    global checks
    with np.load(path) as f:
        if not {'loss','daily','positions','costs'}.issubset(f.files):return
        a={name:f[name] for name in f.files};x=a['loss'];daily=a['daily'];n=len(x)
        check('finite',sum(np.size(v)-np.isfinite(v).sum() for v in a.values()),0,path)
        check('position_bound',max(0,float(np.max(np.abs(a['positions'])))-2),1e-12,path)
        check('telescoping',np.max(np.abs(daily.sum(1)-x+a['v0'])),1e-10,path)
        if 'spot' in a and 'marks' in a:
            normalized=a['spot'][:,32:]/a['spot'][:,32,None];h=a['positions'].astype(np.float64)
            delta=np.diff(np.column_stack([np.zeros(n),h]),axis=1);costs=.001*np.abs(delta)*normalized[:,:-1];costs[:,-1]+=.001*np.abs(h[:,-1])*normalized[:,-1]
            gains=h*np.diff(normalized,axis=1);loss=np.maximum(normalized[:,-1]-1,0)-gains.sum(1)+costs.sum(1)
            check('loss_reconstruction',np.max(np.abs(loss-x)),1e-10,path)
            check('daily_reconstruction',np.max(np.abs(np.diff(a['marks'],axis=1)-gains+costs-daily)),1e-10,path)
            check('cost_reconstruction',np.max(np.abs(costs-a['costs'])),1e-10,path)
            check('payoff_mark',np.max(np.abs(a['marks'][:,-1]-np.maximum(normalized[:,-1]-1,0))),1e-12,path)
            check('turnover',np.max(np.abs(np.abs(delta).sum(1)+np.abs(h[:,-1])-a['turnover'])),1e-10,path)
        mp=path.parent/'metrics.json'
        if mp.exists() and path.stem in ['base','easy','hard']:
            metrics=json.loads(mp.read_text())[path.stem]
            entropic=(logsumexp(10*x)-np.log(n))/10; sortedx=np.sort(x)[::-1]; mass=.05*n;count=int(mass);tail=(sortedx[:count].sum()+(mass-count)*sortedx[count])/mass
            reconstructed={'entropic':entropic,'cvar95':tail,'mean':x.mean(),'terminal_sd':x.std(ddof=1),'centered_entropic':entropic-x.mean(),'centered_cvar95':tail-x.mean(),'daily_error_std':np.sqrt(daily.var(0,ddof=1).mean()),'cost_mean':a['costs'].sum(1).mean(),'turnover_mean':a['turnover'].mean()}
            for name,value in reconstructed.items():check('metric_'+name,abs(value-metrics[name]),1e-10,path)
            check('sample_count',abs(metrics['sample_count']-n),0,path)
            check('finite_count',abs(metrics['finite_count']-n),0,path)

def snapshot(complete):
    result=dict(complete=complete,passed=not failures,checks=checks,files=len(seen),max_errors=maximum,failures=failures,file_sha256=seen,scope='Independent NumPy/SciPy accounting and metric reconstruction on the same GPU worker host; does not by itself certify scientific design or continuous-time pricing.')
    tmp=ROOT/'independent_audit.json.tmp';tmp.write_text(json.dumps(result,indent=2));tmp.replace(ROOT/'independent_audit.json')

start=time.monotonic()
while time.monotonic()-start<11000:
    for p in sorted(ROOT.rglob('*.npz')):
        name=str(p.relative_to(ROOT))
        if name in seen:continue
        # Metrics may be committed immediately after arrays: defer until available.
        if p.stem in ['base','easy','hard'] and not (p.parent/'metrics.json').exists():continue
        inspect(p); seen[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    complete=(ROOT/'artifact_hashes.json').exists()
    snapshot(complete)
    if complete or (ROOT/'failure.json').exists():break
    time.sleep(3)
print(json.dumps(dict(checks=checks,files=len(seen),failures=len(failures))),flush=True)
raise SystemExit(bool(failures))
