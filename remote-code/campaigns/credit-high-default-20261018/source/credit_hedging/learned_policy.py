"""Common recurrent target/retention policy initialized exactly at replenishment."""
import equinox as eqx
import jax
import jax.numpy as jnp

class TargetHedger(eqx.Module):
    cell: eqx.nn.GRUCell
    head: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)
    n_vintages: int = eqx.field(static=True)
    max_protection: float = eqx.field(static=True)
    action_parameterization: str = eqx.field(static=True, default='target_retention_v1')

    def __init__(self,key,n_vintages=126,hidden_size=32,max_protection=2.):
        a,b=jax.random.split(key)
        self.hidden_size=hidden_size;self.n_vintages=n_vintages;self.max_protection=max_protection
        self.cell=eqx.nn.GRUCell(4+3*n_vintages,hidden_size,key=a,dtype=jnp.float64)
        self.head=eqx.nn.Linear(hidden_size,n_vintages+1,key=b,dtype=jnp.float64)
        self.head=eqx.tree_at(lambda h:(h.weight,h.bias),self.head,
                              (jnp.zeros_like(self.head.weight),jnp.zeros_like(self.head.bias)))

    def __call__(self,features,hidden):
        hidden=self.cell(features,hidden)
        logits=self.head(hidden)
        target=self.max_protection*jax.nn.sigmoid(logits[0])
        retain=jnp.clip(1.+1.02*jnp.tanh(logits[1:]),0.,1.)
        retained_total=jnp.sum(features[4:4+self.n_vintages]*self.max_protection*retain)
        buy=jax.nn.relu(target-retained_total)
        return hidden,dict(buy=buy,retain=retain)
