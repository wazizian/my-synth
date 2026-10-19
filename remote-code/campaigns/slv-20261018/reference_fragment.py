# Independent validation routines, appended before main during packaging.
def numpy_filter_reference(pricer,spot,prob):
    from scipy.special import logsumexp as scipy_lse
    spot=np.asarray(spot); prob=np.asarray(prob); vv=np.asarray(pricer.v); grid=np.asarray(pricer.x)
    params={k:np.asarray(v) for k,v in pricer.params.items()}; trans=np.asarray(pricer.transitions); table=np.asarray(pricer.tables)
    n=len(spot); post=np.zeros((n,12,len(vv)))
    for s in range(12):
        hi=np.clip(np.searchsorted(vv,params['v0'][s]),1,len(vv)-1); w=(params['v0'][s]-vv[hi-1])/(vv[hi]-vv[hi-1])
        post[:,s,hi-1]=prob[s]*(1-w); post[:,s,hi]=prob[s]*w
    answer=[]
    for t in range(C+H):
        x=np.log(spot[:,t]/(spot[:,C] if t>=C else 1))
        scale=((1+params['leverage'][None,:]*np.tanh(-2*x[:,None]))**2)[:,:,None]*vv[None,None,:]*DT
        returns=np.log(spot[:,t+1]/spot[:,t])[:,None,None]
        ll=-.5*(np.log(2*np.pi*scale)+(returns+scale/2)**2/scale)
        with np.errstate(divide='ignore'): lp=np.log(post)+ll
        post=np.exp(lp-scipy_lse(lp,axis=(1,2),keepdims=True))
        for s in range(12):
            z=0 if s<2 else ((s-2)//5 if t<C+6+(s-2)%5 else 1-(s-2)//5)
            post[:,s,:]=post[:,s,:]@trans[s,z]
        if t+1>=C:
            x=np.log(spot[:,t+1]/spot[:,C]); marks=np.zeros(n)
            for i in range(n):
                ix=np.clip(np.searchsorted(grid,x[i])-1,0,len(grid)-2)
                f=(np.exp(x[i])-np.exp(grid[ix]))/(np.exp(grid[ix+1])-np.exp(grid[ix]))
                val=(1-f)*table[:,t+1-C,ix,:]+f*table[:,t+1-C,ix+1,:]
                marks[i]=np.sum(post[i]*val)
            if t+1==C+H: marks=np.maximum(spot[:,-1]/spot[:,C]-1,0)
            answer.append(marks)
    return np.array(answer).T

def independent_prices(family,pricer,path):
    params=scenario_parameters(family); results=[]
    # Latent-state prices: independent direct Monte Carlo, no table/filter calls.
    # It validates the backward solver; a separate NumPy filter validates updates.
    @eqx.filter_jit
    def mc(k,scenario,start,n):
        v=jnp.full((n,),.04); s=jnp.ones(n)
        def step(carry,inputs):
            s,v=carry; t,k=inputs; z=regime(jnp.int32(scenario),C+t); a,b=jax.random.split(k)
            local=1+params['leverage'][scenario]*jnp.tanh(-2*jnp.log(s)); w=local**2*v*DT
            nexts=s*jnp.exp(-w/2+jnp.sqrt(w)*jax.random.normal(a,(n,),dtype=jnp.float64))
            nextv=cir_step(b,v,params['kappa'][scenario],params['theta'][scenario,z],params['xi'][scenario,z],DT)
            return (nexts,nextv),None
        (s,_),_=jax.lax.scan(step,(s,v),(jnp.arange(start,H),jax.random.split(k,H-start)))
        # Deterministic half-delta control variate: E[S_T-1]=0 exactly.
        return jnp.maximum(s-1,0)-.5*(s-1)
    for scenario in [0,1,2,7]:
        for start in [0,10]:
            samples=sync(mc(key(11,scenario*20+start),scenario,start,65536))
            values=pricer.tables[scenario,start,pricer.x.size//2]
            table_value=float(jnp.interp(.04,pricer.v,values)); mean=float(samples.mean()); se=float(samples.std(ddof=1)/jnp.sqrt(samples.size))
            diff=abs(table_value-mean); passed=diff<=4*se+2e-5
            results.append(dict(scenario=scenario,date=start,table=table_value,mc_mean=mean,standard_error=se,difference=diff,passed=passed))
            save(path/'independent_pricing_progress.json',dict(latent_price_reference=results))
    sim=make_sim(family); filters=[]
    for domain in ['base','hard']:
        prob=probabilities(family,domain); spot=sim(key(11,100),prob,8)['spot']; actual=np.asarray(pricer.marks(spot,prob))
        reference=numpy_filter_reference(pricer,spot,prob); error=float(np.max(np.abs(actual-reference)))
        filters.append(dict(domain=domain,max_error=error,passed=error<1e-10))
    cir_checks=[]
    for scenario,z in [(1,0),(2,0),(2,1)]:
        kap=float(params['kappa'][scenario]); theta=float(params['theta'][scenario,z]); xi=float(params['xi'][scenario,z]); decay=np.exp(-kap*DT); scale=xi**2*(1-decay)/(4*kap)
        expected_mean,expected_variance=ncx2.stats(4*kap*theta/xi**2,.04*decay/scale,moments='mv'); expected_mean*=scale; expected_variance*=scale**2
        draws=sync(cir_step(key(11,900+scenario*2+z),jnp.full((131072,),.04),kap,theta,xi,DT))
        zz=abs(float(draws.mean())-expected_mean)/np.sqrt(expected_variance/draws.size)
        cir_checks.append(dict(scenario=scenario,regime=z,mean_z=float(zz),expected_mean=float(expected_mean),observed_mean=float(draws.mean()),expected_variance=float(expected_variance),observed_variance=float(draws.var()),passed=bool(zz<4)))
    report=dict(passed=all(x['passed'] for x in results+filters+cir_checks),cir_sampler=cir_checks,latent_price_reference=results,independent_filter=filters,scope='Independent forward continuous-variance MC checks backward pricing; NumPy likelihood/transition recursion checks JAX filter. MC tolerances include reported sampling uncertainty, not a claim of exact pricing.')
    save(path/'independent_pricing.json',report); return report
