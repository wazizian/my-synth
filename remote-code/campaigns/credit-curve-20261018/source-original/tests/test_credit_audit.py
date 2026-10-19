"""Independent audit regressions, including a contract-list NumPy cash account."""
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from credit_hedging.environment import book_step, rollout
from credit_hedging.market import MarketConfig, build_tables, simulate_paths
from credit_hedging.metrics import entropic_risk, evaluate_metrics
from credit_hedging.policy import Hedger
from credit_hedging.train import validate_evaluation


@pytest.mark.parametrize('kwargs', [dict(rate=np.nan), dict(latent_vol=np.nan),
    dict(transaction_cost=np.nan), dict(max_protection=np.inf), dict(bond_days=3.5),
    dict(cds_days=True), dict(grid_size=5.2)])
def test_nonfinite_and_noninteger_configuration_rejected(kwargs):
    """Reject invalid config scalars before tracing; input malformed keyword dict, no output."""
    with pytest.raises(ValueError):
        MarketConfig(**kwargs)


@pytest.mark.parametrize('gamma', [np.nan, np.inf, -1.0, 0.0])
def test_nonfinite_gamma_rejected(gamma):
    """Reject a nonfinite or nonpositive risk aversion before computing a scalar metric."""
    with pytest.raises(ValueError):
        entropic_risk(jnp.array([0.,1.]), gamma)


def test_small_kappa_does_not_freeze_float32_dynamics():
    """Check the almost-Brownian OU limit in float32 against its analytical variance."""
    with jax.experimental.enable_x64(False):
        config = MarketConfig(bond_days=2, cds_days=1, grid_min=-4, grid_max=-3,
                              grid_size=501, mean_reversion=1e-12)
        tables = build_tables(config)
        row = np.asarray(tables['transition'][tables['initial_state']], dtype=np.float64)
        grid = np.asarray(tables['grid'], dtype=np.float64)
        mean = row @ grid
        actual_variance = row @ ((grid-mean)**2)
        np.testing.assert_allclose(actual_variance, config.latent_vol**2/config.trading_days, rtol=.001)


def test_policy_can_hold_without_churn_and_fully_close():
    """Use finite logits to represent exact buy-zero/retain-one and the opposite extremes."""
    model = Hedger(jax.random.key(9), n_vintages=3, hidden_size=4)
    model = eqx.tree_at(lambda m:m.head.weight, model, jnp.zeros_like(model.head.weight))
    model = eqx.tree_at(lambda m:m.head.bias, model, jnp.array([-8.,8.,8.,8.]))
    _, action = model(jnp.zeros(13),jnp.zeros(4))
    np.testing.assert_array_equal(action['buy'],0)
    np.testing.assert_array_equal(action['retain'],1)
    model = eqx.tree_at(lambda m:m.head.bias, model, jnp.array([8.,-8.,-8.,-8.]))
    _, action = model(jnp.zeros(13),jnp.zeros(4))
    np.testing.assert_array_equal(action['buy'],2)
    np.testing.assert_array_equal(action['retain'],0)


def test_nonfinite_evaluation_fails_closed():
    """Ensure NaN scalars and hidden NaN daily arrays cannot pass the export validator."""
    result = dict(loss=jnp.array([0.,0.]), defaulted=jnp.array([False,False]),
                  records=dict(daily_loss=jnp.zeros((2,2)), discounted_cost=jnp.zeros((2,2))))
    metrics = evaluate_metrics(result)
    np.testing.assert_allclose(metrics['exponential_ess'],2)
    np.testing.assert_allclose(metrics['max_exponential_weight'],.5)
    validate_evaluation(result,metrics)
    with pytest.raises(RuntimeError,match='Nonfinite'):
        validate_evaluation(result,dict(metrics,max_accounting_error=jnp.array(jnp.nan)))
    bad = dict(result,records=dict(result['records'],discounted_cost=jnp.array([[0.,jnp.nan],[0.,0.]])))
    with pytest.raises(RuntimeError,match='Nonfinite'):
        validate_evaluation(bad,metrics)


def test_legacy_action_parameterization_remains_restorable():
    """Check the explicit legacy mode preserves pre-audit checkpoint action semantics."""
    model=Hedger(jax.random.key(1),3,4,action_parameterization='sigmoid_v1')
    model=eqx.tree_at(lambda m:m.head.weight,model,jnp.zeros_like(model.head.weight))
    _,action=model(jnp.zeros(13),jnp.zeros(4))
    np.testing.assert_allclose(action['buy'],2*jax.nn.sigmoid(-5.))
    np.testing.assert_allclose(action['retain'],jax.nn.sigmoid(5.))


def scripted_book(buy, retain, paths, tables, config):
    """Run known action arrays [time,batch,...] through scan; return time-major book records."""
    n_time,batch = buy.shape
    book = dict(notional=jnp.zeros((batch,config.cds_days)),coupon=jnp.zeros((batch,config.cds_days)))

    def step(book,day):
        """Settle scripted day from the current book; return next book and daily cash records."""
        market = dict(day=day,state=paths['state'][:,day],next_state=paths['state'][:,day+1],
                      alive=paths['alive'][:,day],default=paths['default'][:,day])
        return book_step(book,dict(buy=buy[day],retain=retain[day]),market,tables,config)

    return jax.lax.scan(step,book,jnp.arange(n_time))


def test_independent_contract_list_cash_account():
    """Compare the vectorized book with a NumPy list of dated contracts and funded bank balance."""
    c=MarketConfig(bond_days=9,cds_days=3,grid_min=-5,grid_max=-2,grid_size=31)
    t=build_tables(c)
    tables=jax.tree.map(np.asarray,t)
    batch=4
    defaults=[None,0,3,8]
    rng=np.random.default_rng(81)
    states=rng.integers(10,21,size=(batch,10))
    alive=np.ones((batch,10),dtype=bool)
    default=np.zeros((batch,9),dtype=bool)
    for b,day in enumerate(defaults):
        if day is not None:
            default[b,day]=True
            alive[b,day+1:]=False
    buy=rng.uniform(0,1.5,(9,batch))
    retain=rng.uniform(0,1,(9,batch,3))
    # Explicitly include hold, full close and cap-saturation cases.
    retain[1]=1
    retain[2]=0
    buy[0]=3
    paths=dict(state=jnp.asarray(states),alive=jnp.asarray(alive),default=jnp.asarray(default))
    book,records=eqx.filter_jit(scripted_book)(jnp.asarray(buy),jnp.asarray(retain),paths,t,c)
    records=jax.tree.map(np.asarray,records)
    dt=1/c.trading_days
    for b in range(batch):
        cash=-tables['bond'][9,states[b,0]]
        previous_discounted_wealth=0.
        contracts=[]
        for day in range(9):
            if alive[b,day]:
                sale=0.
                sold_notional=0.
                for contract in contracts:
                    age=day-contract['issue']
                    remaining=contract['expiry']-day
                    amount=contract['notional']*(1-retain[day,b,age])
                    mark=tables['protection'][remaining,states[b,day]]-contract['coupon']*tables['annuity'][remaining,states[b,day]]
                    sale+=amount*mark
                    sold_notional+=amount
                    contract['notional']-=amount
                purchase=min(buy[day,b],max(c.max_protection-sum(x['notional'] for x in contracts),0.))
                cash+=sale-c.transaction_cost*(sold_notional+purchase)
                contracts.append(dict(issue=day,expiry=day+c.cds_days,notional=purchase,
                                      coupon=tables['par_spread'][states[b,day]]))
                np.testing.assert_allclose(records['sale_cash'][day,b],sale,atol=1e-13)
            cash*=np.exp(c.rate*dt)
            if alive[b,day]:
                cash-=sum(x['notional']*x['coupon']*dt for x in contracts)
                if default[b,day]:
                    cash+=c.recovery+(1-c.recovery)*sum(x['notional'] for x in contracts)
                    contracts=[]
                else:
                    contracts=[x for x in contracts if x['expiry']>day+1]
                    if day==8:
                        cash+=1.
                        for contract in contracts:
                            n=contract['expiry']-(day+1)
                            mark=tables['protection'][n,states[b,day+1]]-contract['coupon']*tables['annuity'][n,states[b,day+1]]
                            cash+=contract['notional']*(mark-c.transaction_cost)
                        contracts=[]
            asset_value=0.
            if alive[b,day+1] and day<8:
                asset_value=tables['bond'][8-day,states[b,day+1]]
                for contract in contracts:
                    n=contract['expiry']-(day+1)
                    asset_value+=contract['notional']*(tables['protection'][n,states[b,day+1]]-contract['coupon']*tables['annuity'][n,states[b,day+1]])
            wealth=np.exp(-c.rate*(day+1)*dt)*(cash+asset_value)
            np.testing.assert_allclose(records['daily_loss'][day,b],previous_discounted_wealth-wealth,atol=1e-12)
            previous_discounted_wealth=wealth
        np.testing.assert_allclose(records['daily_loss'][:,b].sum(),-cash*np.exp(-c.rate*9*dt),atol=1e-12)
    np.testing.assert_array_equal(book['notional'],0)


def test_cds_and_bond_expected_self_financing_gain_zero():
    """Enumerate all next states and default for a mixed-coupon book; cost-free mean daily loss is zero."""
    c=MarketConfig(bond_days=8,cds_days=3,grid_size=31,grid_min=-5,grid_max=-2,transaction_cost=0)
    t=build_tables(c)
    idx=int(t['initial_state'])
    batch=c.grid_size+1
    book=dict(notional=jnp.tile(jnp.array([0.,.3,.2]),(batch,1)),
              coupon=jnp.tile(jnp.array([0.,.01,.05]),(batch,1)))
    market=dict(day=jnp.array(2),state=jnp.full(batch,idx),
                next_state=jnp.concatenate((jnp.arange(c.grid_size),jnp.array([idx]))),
                alive=jnp.ones(batch,dtype=bool),default=jnp.arange(batch)==c.grid_size)
    action=dict(buy=jnp.full(batch,.4),retain=jnp.tile(jnp.array([1.,.5,.8]),(batch,1)))
    _,records=book_step(book,action,market,t,c)
    q=float(t['survival'][idx])
    probabilities=np.concatenate((q*np.asarray(t['transition'][idx]),[1-q]))
    np.testing.assert_allclose(probabilities@np.asarray(records['daily_loss']),0,atol=1e-13)


def test_autodiff_agrees_with_finite_difference():
    """Check a buy-logit derivative through the entire book/rollout against centered finite differences."""
    c=MarketConfig(bond_days=8,cds_days=3,grid_size=31,grid_min=-5,grid_max=-2)
    t=build_tables(c)
    paths=simulate_paths(jax.random.key(8),t,c,2)
    paths=dict(paths,alive=jnp.ones((2,9),dtype=bool).at[1,5:].set(False),
               default=jnp.zeros((2,8),dtype=bool).at[1,4].set(True))
    model=Hedger(jax.random.key(4),3,8)

    def objective(shift):
        """Shift one output bias and return a scalar risk on fixed default/no-default paths."""
        changed=eqx.tree_at(lambda m:m.head.bias,model,model.head.bias.at[0].add(shift))
        return entropic_risk(rollout(changed,paths,t,c)['loss'])

    derivative=jax.grad(objective)(0.)
    finite_difference=(objective(1e-5)-objective(-1e-5))/(2e-5)
    np.testing.assert_allclose(derivative,finite_difference,rtol=2e-5,atol=1e-9)
