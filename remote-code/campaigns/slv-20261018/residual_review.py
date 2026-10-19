"""Supplementary plan-required residual review, only on GPU worker host."""
import json,pathlib,numpy as np,subprocess
from scipy.special import logsumexp
subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],check=True)
root=pathlib.Path('/workspace/job/outputs');reports=[]
def metrics(x):
    mean=x.mean();ordered=np.sort(x)[::-1];mass=.05*len(x);count=int(mass)
    return np.array([x.std(ddof=1),(logsumexp(10*x)-np.log(len(x)))/10-mean,(ordered[:count].sum()+(mass-count)*ordered[count])/mass-mean])
for family in ['A','B']:
    for phase in ['screen','confirmation']:
        directory=root/family/phase
        if not (directory/'gate.json').exists():continue
        easy=np.load(directory/'easy_reference'/'easy.npz');hard=np.load(directory/'hard_reference'/'hard.npz')
        ex=easy['loss']-easy['v0'];hx=hard['loss']-hard['v0'];point=metrics(hx)-metrics(ex)
        rng=np.random.default_rng(np.random.SeedSequence([20261017,12,0 if phase=='screen' else 1]));draws=[]
        for _ in range(400):draws.append(metrics(hx[rng.integers(0,len(hx),len(hx))])-metrics(ex[rng.integers(0,len(ex),len(ex))]))
        intervals=np.quantile(draws,[.025,.975],axis=0);details={name:dict(difference=float(point[i]),low=float(intervals[0,i]),high=float(intervals[1,i])) for i,name in enumerate(['marked_sd','marked_centered_entropic','marked_centered_cvar95'])}
        report=dict(family=family,phase=phase,material_contradiction=bool(np.any(intervals[1]<0)),differences=details,easy_marked=metrics(ex).tolist(),hard_marked=metrics(hx).tolist(),easy_v0_mean=float(easy['v0'].mean()),hard_v0_mean=float(hard['v0'].mean()),easy_v0_sd=float(easy['v0'].std(ddof=1)),hard_v0_sd=float(hard['v0'].std(ddof=1)),scope='Supplementary residual review, not a rewrite of the frozen primary gate; conditional episode intervals only')
        path=directory/'residual_review.json';tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(path);reports.append(report)
print(json.dumps(reports,indent=2))
