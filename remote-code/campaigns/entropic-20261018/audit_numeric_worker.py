"""Independent NumPy reconstruction; execute only on the GPU worker host."""
import hashlib,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(sys.argv[1]);design=json.loads((ROOT/'design.json').read_text())
failures=[];count=0;maxima={};datasets=[]
def check(error,tolerance,name):
    global count
    count+=1;maxima[name]=max(maxima.get(name,0.),float(error))
    if not np.isfinite(error) or error>tolerance:failures.append(dict(test=name,error=float(error),tolerance=tolerance))
def risk(x):
    z=10*x;m=z.max();return (m+np.log(np.exp(z-m).mean()))/10
for candidate in design['candidates']:
    base=ROOT/candidate['name']
    if not base.exists():continue
    C,H=32,20;a=candidate['log_step']
    q=np.full((9,C+H),candidate['base_q']);q[0]=candidate['easy_q']
    for i,t in enumerate(range(6,13),2):q[i,C+t:]=candidate['hard_q']
    up=q/(1+np.exp(a));emission=np.stack([up*np.exp(a),1-q,up],axis=-1)
    table=np.zeros((9,H+1,2*H+1));table[:,H]=np.maximum(np.exp(a*np.arange(-H,H+1))-1,0)
    for t in reversed(range(H)):
        table[:,t,1:-1]=emission[:,C+t,0,None]*table[:,t+1,:-2]+emission[:,C+t,1,None]*table[:,t+1,1:-1]+emission[:,C+t,2,None]*table[:,t+1,2:]
    for path in sorted(base.rglob('metrics.json')):
        scores=json.loads(path.read_text())
        for domain,m in scores.items():
            npz=path.parent/(domain+'.npz')
            try:r=np.load(npz);raw={k:r[k] for k in r.files}
            except (OSError,ValueError):continue # A writer may still be closing a different scope.
            x=raw['loss'];n=len(x);s=raw['spot'];h=raw['positions'];marks=raw['marks'];cost=raw['costs'];daily=raw['daily']
            tag=str(npz.relative_to(ROOT));datasets.append(tag)
            if not all(np.isfinite(v).all() for v in raw.values()):failures.append(dict(test='finite arrays',path=tag))
            rel=s[:,C:]/s[:,C,None];payoff=np.maximum(rel[:,-1]-1,0);gain=h*np.diff(rel,axis=1)
            changes=np.diff(np.concatenate([np.zeros((n,1)),h],axis=1),axis=1)
            expected_cost=.001*np.abs(changes)*rel[:,:-1];expected_cost[:,-1]+=.001*np.abs(h[:,-1])*rel[:,-1]
            check(np.max(np.abs(cost-expected_cost)),1e-7,'cost accounting')
            check(np.max(np.abs(x-(payoff-gain.sum(1)+expected_cost.sum(1)))),1e-7,'terminal loss accounting')
            check(np.max(np.abs(daily-(np.diff(marks,axis=1)-gain+expected_cost))),1e-7,'daily error accounting')
            check(np.max(np.abs(daily.sum(1)-x+raw['v0'])),1e-7,'telescoping')
            check(np.max(np.abs(raw['turnover']-np.abs(changes).sum(1)-np.abs(h[:,-1]))),1e-7,'turnover')
            check(max(0,np.abs(h).max()-2),1e-7,'position bounds')
            check(np.max(np.abs(marks[:,-1]-payoff)),1e-12,'terminal liability mark')
            moves=np.rint(np.log(s[:,1:]/s[:,:-1])/a).astype(int)
            check(np.max(np.abs(np.log(s[:,1:]/s[:,:-1])/a-moves)),1e-10,'trinomial lattice')
            check(max(0,np.abs(moves).max()-1),0,'increment support')
            post=np.broadcast_to(design['base_probabilities'],(n,9)).copy()
            relative=np.rint(np.log(rel)/a).astype(int)
            reconstructed=np.zeros_like(marks)
            for t in range(C+H+1):
                if t>=C:reconstructed[:,t-C]=(post*table[:,t-C,relative[:,t-C]+H].T).sum(1)
                if t<C+H:
                    post*=emission[:,t,moves[:,t]+1].T;post/=post.sum(1,keepdims=True)
            check(np.max(np.abs(reconstructed-marks)),1e-10,'independent Bayesian valuation')
            check(abs(risk(x)-m['entropic']),1e-7,'entropic metric')
            mass=.05*n;k=int(mass);order=np.sort(x)[::-1];cvar=(order[:k].sum()+(mass-k)*order[k])/mass
            check(abs(cvar-m['cvar95']),1e-7,'fractional CVaR metric')
            check(abs(np.sqrt(daily.var(0,ddof=1).mean())-m['daily_error_std']),1e-7,'daily dispersion metric')
            check(np.max(np.abs(daily.mean(0)-m['daily_mean'])),1e-10,'daily mean metric')
            check(np.max(np.abs(daily.std(0,ddof=1)-m['daily_sd'])),1e-10,'daily SD curve')
            expected=hashlib.sha256(raw['spot'].tobytes()+raw['marks'].tobytes()+raw['switch'].tobytes()).hexdigest()
            if expected!=m['dataset_sha256']:failures.append(dict(test='dataset hash',path=tag))
residual_difficulty={}
for candidate in design['candidates']:
    name=candidate['name'];base=ROOT/name/'screen';comparisons={}
    for arm,domain in [('easy_reference','easy'),('hard_reference','hard')]:
        p=base/arm/(domain+'.npz')
        if not p.exists():continue
        r=np.load(p);x=r['loss'];center=x-x.mean();v=r['loss']-r['v0'];mass=.05*len(x);k=int(mass)
        def cvar(z):
            z=np.sort(z)[::-1];return float((z[:k].sum()+(mass-k)*z[k])/mass)
        comparisons[domain]=dict(terminal_std=float(x.std(ddof=1)),entropic_above_mean=float(risk(center)),cvar_above_mean=cvar(center),marked_residual_entropic=float(risk(v)),marked_residual_cvar=cvar(v),marked_residual_mean=float(v.mean()),marked_residual_std=float(v.std(ddof=1)))
    if len(comparisons)==2:
        comparisons['hard_above_easy']={field:comparisons['hard'][field]>comparisons['easy'][field] for field in comparisons['easy']}
    residual_difficulty[name]=comparisons
result=dict(passed=not failures,check_count=count,dataset_count=len(datasets),datasets=datasets,residual_difficulty=residual_difficulty,max_errors=maxima,failures=failures,scope='All complete evaluation datasets present at audit time; independent NumPy calculation on GPU worker host')
print(json.dumps(result,indent=2,allow_nan=False))
raise SystemExit(bool(failures))
