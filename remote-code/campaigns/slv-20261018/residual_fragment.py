def marked_review(path):
    from scipy.special import logsumexp as scipy_logsumexp
    easy=np.load(path/'easy_reference'/'easy.npz');hard=np.load(path/'hard_reference'/'hard.npz')
    ex=easy['loss']-easy['v0'];hx=hard['loss']-hard['v0']
    def summaries(x):
        mean=x.mean();ordered=np.sort(x)[::-1];mass=.05*len(x);count=int(mass)
        return np.array([x.std(ddof=1),(scipy_logsumexp(10*x)-np.log(len(x)))/10-mean,(ordered[:count].sum()+(mass-count)*ordered[count])/mass-mean])
    point=summaries(hx)-summaries(ex);rng=np.random.default_rng(np.random.SeedSequence([20261017,12,0 if path.name=='screen' else 1]));draws=[]
    for _ in range(400):draws.append(summaries(hx[rng.integers(0,len(hx),len(hx))])-summaries(ex[rng.integers(0,len(ex),len(ex))]))
    intervals=np.quantile(draws,[.025,.975],axis=0);margins=np.array([2e-4,4e-4,4e-4])
    details={name:dict(difference=float(point[i]),low=float(intervals[0,i]),high=float(intervals[1,i]),numerical_margin=float(margins[i])) for i,name in enumerate(['marked_sd','marked_centered_entropic','marked_centered_cvar95'])}
    result=dict(material_contradiction=bool(np.any(intervals[1]<-margins)),statistical_contradiction=bool(np.any(intervals[1]<0)),differences=details,easy_marked=summaries(ex).tolist(),hard_marked=summaries(hx).tolist(),easy_v0_mean=float(easy['v0'].mean()),hard_v0_mean=float(hard['v0'].mean()),easy_v0_sd=float(easy['v0'].std(ddof=1)),hard_v0_sd=float(hard['v0'].std(ddof=1)),scope='Supplementary plan-required residual review declared before training; primary gate reported separately. Conservative margins reflect fixed mark-error tolerance. Conditional episode uncertainty only.')
    save(path/'residual_review.json',result);return result
