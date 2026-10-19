"""Reconstruct screen metrics from exported episode records on the worker."""
from pathlib import Path
import json
import numpy as np


def run(root):
    root=Path(root)
    metrics=json.loads((root/'all_metrics.json').read_text())
    checks=[]
    for domain in ('easy','ordinary','hard'):
        for policy,expected in metrics[domain].items():
            losses=[];total=np.zeros(1260);squares=np.zeros(1260);cost=0.;n=0;max_error=0.
            for path in sorted((root/domain).glob(policy+'-[0-9][0-9][0-9][0-9][0-9].npz')):
                with np.load(path) as a:
                    if not all(np.isfinite(a[k]).all() for k in a.files):
                        raise RuntimeError(f'Nonfinite exported array: {path}')
                    loss=a['loss'];daily=a['daily_loss'];n+=len(loss)
                    residual1=np.max(np.abs(daily.sum(axis=1)-loss))
                    residual2=np.max(np.abs(a['initial_price']-a['discounted_cashflow'].sum(axis=1)-loss))
                    max_error=max(max_error,float(residual1),float(residual2))
                    losses.append(loss);total+=daily.sum(axis=0);squares+=(daily*daily).sum(axis=0)
                    cost+=a['discounted_cost'].sum()
                    if (a['notional']<0).any() or (a['notional']>2+1e-10).any():
                        raise RuntimeError('Position bounds violated')
                    suffix=path.stem[-5:]
                    with np.load(root/domain/f'paths-{suffix}.npz') as p:
                        if not np.array_equal(p['default'].any(axis=1),a['defaulted']):
                            raise RuntimeError('Default labels mismatch')
                        if np.any(daily[~p['alive'][:,:-1]] != 0):
                            raise RuntimeError('Nonzero absorbing daily loss')
            if n!=8192 or max_error>1e-10:
                raise RuntimeError(f'Incomplete or inconsistent records {domain} {policy}')
            loss=np.concatenate(losses)
            maximum=loss.max()
            entropic=maximum+np.log(np.exp(10*(loss-maximum)).sum()/n)/10
            # Independent fractional-boundary tail implementation.
            tail=.05*n;k=int(np.floor(tail));ordered=np.sort(loss)
            cvar=(ordered[-k:].sum()+(tail-k)*ordered[-k-1])/tail
            daily_std=np.sqrt(np.maximum(squares/n-(total/n)**2,0))
            actual=dict(entropic=entropic,cvar95=cvar,daily_error_std=np.sqrt(np.mean(daily_std**2)),mean_cost=cost/n)
            errors={key:float(abs(value-expected[key])) for key,value in actual.items()}
            if max(errors.values())>1e-10:
                raise RuntimeError('Independent metric reconstruction disagrees')
            checks.append(dict(domain=domain,policy=policy,episodes=n,max_accounting_error=max_error,metric_errors=errors))
    result=dict(passed=True,checks=checks,scope='All 18 component/policy cells, exported terminal/daily cash records, fractional tail reconstruction, absorbing defaults and position bounds. Base weighted mixture and gate intervals are separately retained; this audit does not certify continuous dynamics.')
    (root/'independent_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result
