"""Paired full-calendar feasible-policy screen, executed only on a GPU worker."""
from pathlib import Path
from dataclasses import asdict
import gc
import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .curve_market import CurveConfig, build_curve_tables, simulate_curve_paths
from .environment import rollout
from .diagnostic_policies import FixedProtection, ObservableSensitivity


def risk(loss, gamma=10., axis=-1):
    maximum = np.max(loss, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis)+np.log(np.mean(np.exp(gamma*(loss-maximum)), axis=axis))/gamma


def summary(loss, daily_sum, daily_square_sum, n, costs, turnover, defaulted, initial, gamma):
    daily_mean = daily_sum/n
    daily_std = np.sqrt(np.maximum(daily_square_sum/n-daily_mean**2, 0))
    descending = np.sort(loss)[::-1]
    tail = .05*n
    weights = np.clip(tail-np.arange(n), 0, 1)
    expweights = np.exp(gamma*(loss-loss.max()))
    expweights /= expweights.sum()
    return dict(entropic=float(risk(loss, gamma)), cvar95=float(weights@descending/tail),
                var95=float(np.quantile(loss, .95)), mean=float(loss.mean()),
                terminal_std=float(loss.std()), centered_entropic=float(risk(loss-loss.mean(), gamma)),
                centered_cvar95=float(weights@descending/tail-loss.mean()),
                daily_error_std=float(np.sqrt(np.mean(daily_std**2))), daily_mean=daily_mean.tolist(),
                daily_std=daily_std.tolist(), mean_cost=float(costs/n), mean_turnover=float(turnover/n),
                default_count=int(defaulted.sum()), survival_count=int((~defaulted).sum()),
                default_path_mean_loss=float(loss[defaulted].mean()) if defaulted.any() else None,
                survival_path_mean_loss=float(loss[~defaulted].mean()) if (~defaulted).any() else None,
                initial_bond_price=float(initial), n=n, tail_mass=tail,
                entropic_ess=float(1/(expweights@expweights)), max_exponential_weight=float(expweights.max()))


def run(design_path, numerical_path, output):
    assert jax.default_backend() == 'gpu'
    jax.config.update('jax_enable_x64', True)
    design = json.loads(Path(design_path).read_text())
    numerical = json.loads(Path(numerical_path).read_text())
    if not all(v['passed'] for v in numerical.values()):
        raise RuntimeError('Pricing refinement gate has not passed')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    gamma = design['gamma']
    count = design['diagnostics']['episodes_per_component']
    batch = 128
    if count % batch:
        raise ValueError('Episode count must be divisible by evaluation batch size')
    ordinary = CurveConfig(**numerical['ordinary']['production_config'])
    ordinary_tables = eqx.filter_jit(build_curve_tables)(ordinary)
    sensitivity = ObservableSensitivity(ordinary_tables, ordinary)
    jax.block_until_ready(sensitivity)
    if not np.all(np.diff(np.asarray(ordinary_tables['par_spread'])) > 0):
        raise RuntimeError('Quote-implied ordinary state is not monotone')
    del ordinary_tables
    policies = {'unhedged':None, 'replenish_0.5':FixedProtection(.5),
                'replenish_1.0':FixedProtection(1.), 'replace_daily_0.5':FixedProtection(.5, True),
                'replace_daily_1.0':FixedProtection(1., True), 'observable_sensitivity':sensitivity}
    evaluator = eqx.filter_jit(rollout)
    simulator = eqx.filter_jit(simulate_curve_paths)
    timings = []
    all_losses, all_summaries = {}, {}
    for domain_index, domain in enumerate(('easy', 'ordinary', 'hard')):
        cfg = CurveConfig(**numerical[domain]['production_config'])
        root = output/domain
        root.mkdir(exist_ok=True)
        (root/'config.json').write_text(json.dumps(asdict(cfg), indent=2))
        tables = eqx.filter_jit(build_curve_tables)(cfg)
        jax.block_until_ready(tables)
        if not all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree.leaves(tables)):
            raise RuntimeError('Nonfinite production price table')
        initial = float(tables['bond'][cfg.bond_days, tables['initial_state']])
        initial_bonds = np.asarray(tables['bond'][:,tables['initial_state']])
        survival_curve = [1.]
        discounted_defaults = 0.
        for day in range(1,cfg.bond_days+1):
            discount = np.exp(-cfg.rate*day/cfg.trading_days)
            q = (initial_bonds[day]-cfg.recovery*discounted_defaults-
                 cfg.recovery*discount*survival_curve[-1])/((1-cfg.recovery)*discount)
            discounted_defaults += discount*(survival_curve[-1]-q)
            survival_curve.append(q)
        survival_curve = np.asarray(survival_curve)
        if np.any(np.diff(survival_curve)>1e-10) or np.any(survival_curve < -1e-10):
            raise RuntimeError('Inconsistent survival curve')
        np.savez_compressed(root/'initial_curves.npz',bond=initial_bonds,survival=survival_curve,
                            cds_protection=np.asarray(tables['protection'][:,tables['initial_state']]),
                            cds_annuity=np.asarray(tables['annuity'][:,tables['initial_state']]),
                            initial_par_spread=np.asarray(tables['par_spread'][tables['initial_state']]))
        accum = {name:dict(loss=[], daily_sum=np.zeros(cfg.bond_days), daily_square_sum=np.zeros(cfg.bond_days),
                          inventory_sum=np.zeros((cfg.bond_days,cfg.cds_days)),costs=0.,turnover=0.,max_accounting=0.,
                          default_day_loss_sum=0.) for name in policies}
        defaults = []
        for start in range(0, count, batch):
            started = time.monotonic()
            key = jax.random.fold_in(jax.random.key(design['seed']), 110000+domain_index*10000+start)
            paths = simulator(key, tables, cfg, batch)
            host_paths = jax.device_get(paths)
            np.savez_compressed(root/f'paths-{start:05}.npz', **host_paths)
            defaulted = host_paths['default'].any(axis=1)
            defaults.append(defaulted)
            for name, policy in policies.items():
                result = evaluator(policy, paths, tables, cfg, record_inventory=True)
                host = jax.device_get(result)
                if not all(np.isfinite(v).all() for v in jax.tree.leaves(host)):
                    raise RuntimeError('Nonfinite economic-screen output')
                r = host['records']; a = accum[name]
                residual = float(np.max(np.abs(r['daily_loss'].sum(axis=1)-host['loss'])))
                if residual > design['thresholds']['accounting_max_error']:
                    raise RuntimeError('Economic-screen accounting failed')
                a['max_accounting'] = max(a['max_accounting'],residual)
                a['loss'].append(host['loss'])
                a['daily_sum'] += r['daily_loss'].sum(axis=0)
                a['daily_square_sum'] += (r['daily_loss']**2).sum(axis=0)
                a['inventory_sum'] += r['inventory'].sum(axis=0)
                a['costs'] += r['discounted_cost'].sum()
                a['turnover'] += (r['buy']+r['sold']).sum()
                a['default_day_loss_sum'] += r['daily_loss'][host_paths['default']].sum()
                # Full paths and deterministic rule definitions reconstruct the
                # vintage ledger. Retain all daily cash/position records plus
                # aggregate vintage holdings and a representative full trace.
                arrays = {k:v for k,v in r.items() if k not in ('inventory','coupons')}
                arrays.update(loss=host['loss'],initial_price=host['initial_price'],defaulted=defaulted)
                np.savez_compressed(root/f'{name}-{start:05}.npz', **arrays)
                if start == 0:
                    np.savez_compressed(root/f'{name}-inventory-traces.npz',inventory=r['inventory'][:8],coupons=r['coupons'][:8])
                del result,host,r
            timings.append(dict(domain=domain, first_episode=start, episodes=batch,
                                wall_seconds_including_compile_io=time.monotonic()-started))
            (output/'timing.json').write_text(json.dumps(timings,indent=2))
            print(f'economic {domain}: {start+batch}/{count}', flush=True)
        defaulted = np.concatenate(defaults)
        domain_summary = {}
        all_losses[domain] = {}
        for name,a in accum.items():
            loss = np.concatenate(a['loss'])
            all_losses[domain][name] = loss
            metrics = summary(loss,a['daily_sum'],a['daily_square_sum'],count,a['costs'],a['turnover'],defaulted,initial,gamma)
            metrics.update(max_accounting_error=a['max_accounting'], default_day_mean_loss=float(a['default_day_loss_sum']/max(defaulted.sum(),1)))
            domain_summary[name] = metrics
            np.savez_compressed(root/f'{name}-inventory-mean.npz',mean_inventory=a['inventory_sum']/count)
        (root/'metrics.json').write_text(json.dumps(domain_summary,indent=2,allow_nan=False))
        all_summaries[domain] = domain_summary
        del tables,accum,paths,host_paths
        jax.clear_caches();gc.collect()
    names = list(policies)
    # Exact empirical P0 mixture: component samples receive declared weights.
    # Ordinary component metrics are retained separately, never mislabeled base.
    base_metrics = {}
    for name in names:
        loss = np.concatenate([all_losses[d][name] for d in design['P0']])
        weights = np.concatenate([np.full(count,p/count) for p in design['P0'].values()])
        mean = float(weights@loss)
        exps = weights*np.exp(gamma*(loss-loss.max()))
        expnorm = exps/exps.sum()
        order = np.argsort(loss)[::-1]
        before = np.cumsum(weights[order])-weights[order]
        tail_weights = np.minimum(weights[order],np.maximum(.05-before,0))
        daily_mean = sum(p*np.asarray(all_summaries[d][name]['daily_mean']) for d,p in design['P0'].items())
        daily_second = sum(p*(np.asarray(all_summaries[d][name]['daily_std'])**2+
                              np.asarray(all_summaries[d][name]['daily_mean'])**2) for d,p in design['P0'].items())
        daily_std = np.sqrt(np.maximum(daily_second-daily_mean**2,0))
        entropic = float(loss.max()+np.log(exps.sum())/gamma)
        cvar = float(tail_weights@loss[order]/.05)
        base_metrics[name] = dict(entropic=entropic,cvar95=cvar,mean=mean,
                                 centered_entropic=entropic-mean,centered_cvar95=cvar-mean,
                                 terminal_std=float(np.sqrt(weights@((loss-mean)**2))),
                                 daily_error_std=float(np.sqrt(np.mean(daily_std**2))),
                                 daily_mean=daily_mean.tolist(),daily_std=daily_std.tolist(),
                                 mean_cost=sum(p*all_summaries[d][name]['mean_cost'] for d,p in design['P0'].items()),
                                 mean_turnover=sum(p*all_summaries[d][name]['mean_turnover'] for d,p in design['P0'].items()),
                                 initial_bond_price=sum(p*all_summaries[d][name]['initial_bond_price'] for d,p in design['P0'].items()),
                                 default_frequency=sum(p*all_summaries[d][name]['default_count']/count for d,p in design['P0'].items()),
                                 n=len(loss),weighted_tail_mass=.05,entropic_ess=float(1/(expnorm@expnorm)),
                                 max_exponential_weight=float(expnorm.max()),estimator='Exact declared mixture of component empirical measures')
        np.savez_compressed(output/f'base-{name}-weighted-terminal.npz',loss=loss,weight=weights)
    all_summaries['base'] = base_metrics
    intervals = {}
    for domain in ('easy','hard'):
        losses = np.stack([all_losses[domain][n] for n in names])
        rng = np.random.default_rng(120000+design['seed']+(domain=='hard'))
        boot = []
        for _ in range(design['diagnostics']['paired_bootstrap_replicates']):
            ix = rng.integers(count,size=count)
            risks = risk(losses[:,ix],gamma)
            boot.append(risks-risks.min())
        boot = np.asarray(boot)
        observed = risk(losses,gamma)
        intervals[domain] = {n:dict(regret=float(observed[i]-observed.min()),
                                  ci95=np.quantile(boot[:,i],[.025,.975]).tolist()) for i,n in enumerate(names)}
    equivalent = [n for n in names if all(intervals[d][n]['ci95'][1] < .001 for d in ('easy','hard'))]
    result = dict(stop_uninformative=bool(equivalent), common_near_best_policies=equivalent,
                  intervals=intervals, scope='Development-only feasible-policy comparison; one frozen bounded design; no neural or final-test data',
                  base_definition='P0 is .05 easy + .90 ordinary + .05 hard; ordinary component is not P0',
                  inventory_scope='All daily trades/notional/cashflows and scenario paths retained; full vintage traces for first eight paths, mean vintage surfaces for all paths; deterministic rules permit reconstruction')
    (output/'gate.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (output/'all_metrics.json').write_text(json.dumps(all_summaries,indent=2,allow_nan=False))
    return result
