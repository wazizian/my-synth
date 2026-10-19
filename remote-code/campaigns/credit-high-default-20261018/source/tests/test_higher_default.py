import jax
import jax.numpy as jnp
import numpy as np
import pytest
from credit_hedging.curve_market import CurveConfig,build_curve_tables
from credit_hedging.numerical_validation import sampled_prices

@pytest.mark.parametrize('multiplier',[1.,3.])
def test_scaled_intensity_matches_recursion_and_independent_reference(multiplier):
 c=CurveConfig(bond_days=12,cds_days=4,grid_size=13,fast_grid_size=21,hazard_multiplier=multiplier)
 t=build_curve_tables(c)
 x=np.asarray(t['x_grid']);y=np.asarray(t['y_grid'])
 expected=np.exp(-multiplier*np.logaddexp(0,x[:,None]+y[None,:])/252).ravel()
 np.testing.assert_allclose(t['survival'],expected,atol=1e-14)
 joint=np.kron(np.asarray(t['transition_x']),np.asarray(t['transition_y']))
 for day in range(1,13):
  price=np.exp(-c.rate/252)*((1-expected)*c.recovery+expected*(joint@np.asarray(t['bond'][day-1])))
  np.testing.assert_allclose(t['bond'][day],price,atol=1e-12)
 ref=sampled_prices(c)
 ix=np.abs(x[:,None]-np.asarray(ref['x'])).argmin(0);iy=np.abs(y[:,None]-np.asarray(ref['y'])).argmin(0)
 indices=ix[:,None]*len(y)+iy[None,:]
 for k,name in enumerate(('bond','protection','annuity')):
  days=c.bond_days if k==0 else c.cds_days
  np.testing.assert_allclose(t[name][1:days+1,indices],ref['prices'][:days,k],atol=1e-12,rtol=1e-12)

def test_invalid_hazard_multiplier():
 for v in (0.,-1.,float('nan')):
  with pytest.raises(ValueError):CurveConfig(hazard_multiplier=v)
