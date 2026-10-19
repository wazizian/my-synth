"""Price-history-only adapter for the frozen stochastic-option benchmark.

Import and execution belong on GPU workers. Domain menu is easy/base/hard.
The batch carries causal log-price paths and an external initial-premium ledger
constant. Only log prices reach the policy; premium and domain never become
features. Latent factors stay internal.
"""
from pathlib import Path
import importlib.util
import sys
import json
import jax
import jax.numpy as jnp
import equinox as eqx

class OptionAdapter(eqx.Module):
    model: object = eqx.field(static=True)
    cost: float
    decay: object
    noise: object
    prob: object
    premiums: object
    names=('easy','base','hard')
    def __init__(self,source,bp):
        source=Path(source)
        path=source/'model.py' if source.is_dir() else source
        spec=importlib.util.spec_from_file_location('option_benchmark',path)
        self.model=importlib.util.module_from_spec(spec);sys.modules[spec.name]=self.model;spec.loader.exec_module(self.model)
        self.cost=float(bp)/10000
        values=json.loads((path.parent/'initial_premiums.json').read_text())
        self.premiums=jnp.asarray([values[n] for n in self.names])
        half=jnp.asarray([self.model.P_HALF[n] for n in self.names])
        self.decay,self.noise=jax.vmap(self.model.coefficients)(half)
        self.prob=jnp.asarray([self.model.PROB[n] for n in self.names])
    def make_model(self,seed=20261017):
        return self.model.Hedger(jax.random.key(seed))
    def losses(self,model,batch):
        return self.model.rollout(model,batch['logs'],self.cost)['loss']-batch['initial_premium']
    def domain(self,key,domain,count):
        if isinstance(domain,int):domain=self.names[domain]
        if domain=='baseline':domain='base'
        index=self.names.index(domain)
        return {'logs':self.model.simulate(key,domain,count)['logs'],'initial_premium':jnp.full(count,self.premiums[index])}
    def draw(self,key,probabilities,block_size=8,batch_size=256):
        ik,zk,uk=jax.random.split(key,3)
        block_ids=jax.random.categorical(ik,jnp.log(jnp.asarray(probabilities)),shape=(block_size,batch_size))
        ids=block_ids.reshape(-1)
        total=block_size*batch_size
        z=jax.random.normal(zk,(self.model.HORIZON,total,3))
        u=jax.random.uniform(uk,(self.model.HORIZON,total))
        decay=self.decay[ids];noise=self.noise[ids];prob=self.prob[ids]
        def step(carry,inputs):
            x,log_s=carry;zz,uu=inputs
            v=self.model.variance(x)
            r=-.5*v*self.model.DT+jnp.sqrt(v*self.model.DT)*zz[:,2]
            next_x=decay*x+noise*zz[:,:2]+(uu<prob)[:,None]*jnp.log(2.)
            return (next_x,log_s+r),log_s
        (_,last),logs=jax.lax.scan(step,(jnp.zeros((total,2)),jnp.zeros(total)),(z,u))
        return {'logs':jnp.concatenate((logs,last[None]),axis=0).reshape(self.model.HORIZON+1,block_size,batch_size).transpose(1,0,2),'initial_premium':self.premiums[ids].reshape(block_size,batch_size)},block_ids
    def validate_on_gpu(self,key):
        """Numerical smoke for mixed law shape, observation discipline and gradients."""
        if jax.default_backend()!='gpu':raise RuntimeError('GPU required')
        model=self.make_model();batch,ids=self.draw(key,jnp.asarray([0.,1.,0.]))
        assert batch['logs'].shape==(8,127,256) and ids.shape==(8,256)
        first=jax.tree.map(lambda x:x[0],batch)
        loss=self.losses(model,first)
        value,gradient=eqx.filter_value_and_grad(lambda m:jnp.mean(self.losses(m,first)))(model)
        leaves=jax.tree_util.tree_leaves(eqx.filter(gradient,eqx.is_inexact_array))
        assert bool(jnp.isfinite(loss).all()) and bool(jnp.isfinite(value))
        assert all(bool(jnp.isfinite(x).all()) for x in leaves)
        assert bool((ids==1).all())
        return {'shape':list(batch['logs'].shape),'blocks':int(ids.size),'finite_loss_and_gradient':True,'price_history_only':True}
