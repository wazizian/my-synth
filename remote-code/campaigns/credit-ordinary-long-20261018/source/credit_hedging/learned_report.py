"""Produce scientific figures and report on the evaluation GPU host."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .learned_run import write


def main(root,training):
    root=Path(root);training=Path(training);m=json.loads((root/'metrics.json').read_text());g=json.loads((root/'paired_gains.json').read_text());s=json.loads((root/'static_selection.json').read_text());sm=json.loads((root/'static_metrics.json').read_text())
    figures=root/'figures';figures.mkdir(exist_ok=True)
    names=['baseline','easy','hard','replenish_1.0'];labels=['Base-trained','Easy specialist','Hard specialist','Fixed one-unit'];colors=['#334155','#16a34a','#dc2626','#a855f7']
    for domain in ('easy','ordinary','hard','base'):
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for ax,key,label in zip(axes,('entropic','cvar95','mse'),('Entropic risk','CVaR95','MSE')):
            ax.bar(np.arange(4),[m[domain][n][key] for n in names],color=colors);ax.set_xticks(np.arange(4),labels,rotation=25,ha='right');ax.set_title(label);ax.set_ylabel('Face units' if key!='mse' else 'Squared face units');ax.grid(axis='y',alpha=.2)
        fig.suptitle(f'{domain}: frozen selected policies, fresh paired evaluation');fig.savefig(figures/f'{domain}-risk.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
        for name,label,color in zip(names,labels,colors):
            a=np.load(root/f'{domain}-{name}-daily-curve.npz');ax.plot(np.arange(1260)+1,a['std'],label=label,color=color,alpha=.8)
        ax.set(xlabel='Trading day',ylabel='Cross-episode daily loss SD, face units',title=domain+'; fixed calendar including absorbing zeros');ax.legend();ax.grid(alpha=.2);fig.savefig(figures/f'{domain}-daily.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,arm in zip(axes,('baseline','easy','hard')):
        selection=json.loads((training/(arm+'-selection.json')).read_text())
        for trial in selection['trials']:
            lr=trial['lr'];p=training/(arm+f'-lr-{lr:g}-progress.json');progress=json.loads(p.read_text())
            ax.plot([0]+[v['active_seconds'] for v in progress['history']],[progress['initial']['entropic']]+[v['entropic'] for v in progress['history']],marker='o',label=f'lr={lr:g}')
        ax.set(title=arm,xlabel='Active training seconds',ylabel='Own-distribution development entropic risk');ax.legend();ax.grid(alpha=.2)
    fig.savefig(figures/'tuning-curves.png',dpi=160);plt.close(fig)
    gates={}
    for d in ('easy','hard'):
        gain=g[d]['specialist_vs_baseline'];gates[d]=dict(useful_gain_pass=gain['ci95'][0]>.001,**gain)
    ordering={k:m['hard']['hard'][k]-m['easy']['easy'][k] for k in ('terminal_std','centered_entropic','centered_cvar95','daily_error_std')}
    write(root/'learned_gate.json',dict(target_gains=gates,hard_minus_easy_specialist_residual=ordering,accepted=all(x['useful_gain_pass'] for x in gates.values()) and all(x>0 for x in ordering.values()),scope='Single-seed development/confirmation evidence; no adaptive teacher comparison'))
    lines=['# Learned credit baseline and specialists','',
    'Three recurrent hedgers were trained on the unchanged credit markets: exact P0 mixture, easy only and hard only. All arms used identical initialization, GRU(382,32), target/retention action map, entropic gamma10, batch160, clipping1 and the same three learning rates and 600-active-second budgets. Each trial started exactly at one-unit expiry replenishment; the initial checkpoint was eligible for selection. This warm start and action parameterization are shared improvements to the baseline, not privileged specialist information.','',
    'All selected policies were frozen using separate development episodes before this fresh paired evaluation of8192 episodes per component. Base is the exact .05 easy/.90 ordinary/.05 hard empirical mixture. One training initialization; paired bootstrap intervals measure evaluation uncertainty only. No adaptive teacher ran.','',
    '| Evaluation domain | Hedge | Entropic | CVaR95 | MSE | Daily SD | Mean cost |',
    '|---|---|---:|---:|---:|---:|---:|']
    for d in ('easy','ordinary','hard','base'):
        for name in names:
            a=m[d][name];lines.append(f"| {d} | {name} | {a['entropic']:.8f} | {a['cvar95']:.8f} | {a['mse']:.8f} | {a['daily_error_std']:.8f} | {a['mean_cost']:.8f} |")
    baseline_result=json.loads((root/'in_distribution_baseline.json').read_text())
    lines += ['',f"On its P0 training distribution, the neural baseline has entropic risk {baseline_result['learned_entropic']:.8f} versus {baseline_result['static_entropic']:.8f} for the development-tuned static hedge (target {baseline_result['static_target']:.2f}). Gain {baseline_result['gain']:.8f}, paired95% interval [{baseline_result['ci95'][0]:.8f}, {baseline_result['ci95'][1]:.8f}]."]
    lines+=['','## Specialization and static comparators','', '| Target | Specialist gain over neural baseline | Paired95% interval | Lower bound above0.001? |','|---|---:|---|---|']
    for d in ('easy','hard'):
        a=gates[d];lines.append(f"| {d} | {a['gain']:.8f} | [{a['ci95'][0]:.8f}, {a['ci95'][1]:.8f}] | {a['useful_gain_pass']} |")
    lines+=['','Static protection levels were selected independently on development data from the frozen menu, not from final outcomes. The same full-calendar vintage ledger and costs apply.','']
    for arm,v in s.items():lines.append(f"- {arm}: selected static target {v['target']:.2f}.")
    lines+=['','| Target | Learned hedge | Entropic gain over its development-tuned static comparator | Paired95% interval |','|---|---|---:|---|']
    for d,rs in g.items():
        for arm in ('baseline',d):
            a=rs[arm+'_vs_tuned_static'];lines.append(f"| {d} | {arm} | {a['gain']:.8f} | [{a['ci95'][0]:.8f}, {a['ci95'][1]:.8f}] |")
    lines+=['','Hard-minus-easy specialist residual diagnostics: `'+json.dumps(ordering)+'`. A larger loss level alone does not establish difficulty.','',
    '## Training and selection','', '| Arm | Selected learning rate | Selected update | Selected active seconds | Trial terminal steps |','|---|---:|---:|---:|---|']
    for arm in ('baseline','easy','hard'):
        a=json.loads((training/(arm+'-selection.json')).read_text());v=a['selected'];lines.append(f"| {arm} | {v['lr']:g} | {v['best_step']} | {v['best_active_seconds']:.2f} | "+', '.join(f"{x['lr']:g}: {x['steps']}" for x in a['trials'])+' |')
    lines+=['','Training uses160 paths per step. Baseline uses fixed8/144/8 stratification and the weighted exponential moment; this estimates mixture entropic risk rather than averaging component risks. The finite-minibatch log-mean-exp objective has sampling bias and is not an exact population gradient. Initial compilation includes one retained learner update, excluded from active timing equally for all trials. Compilation, evaluation and I/O are additional paid costs; equal active time may yield differing update counts. All tuning trials and resumable model/optimizer/PRNG checkpoints are retained.','',
    '## Numerical scope','',
    'The preceding campaign validated the unchanged production grids and pricing convention against refined grids and boundary widening. All41 GPU tests passed before training, including causal accounting, gradients, exact initial static-rule equivalence and checkpoint replay. Selected-policy full-calendar raw records passed accounting/finite/bound checks at1e-10. This is a daily finite-state model with custom daily-coupon CDS, not a continuous-time error certification or ISDA market model. Policies see quotes and inventory, not simulator labels or bond marks. Quotes may reveal state information.','',
    'Risk metrics are positive discounted loss per bond face; CVaR95 uses fractional tail mass. Daily population SD includes absorbing zeros after default. Opening-trade turnover excludes final liquidation notional; costs include liquidation. Initial bond value, mean and centered risks, default/survival outcomes and entropic ESS are retained in metrics.json. Raw selected-policy records include every daily loss, discounted cashflow/cost, purchase, sale and total notional, plus scenario paths. They do not contain full per-vintage cubes; replay uses retained policies, sources and paths. No across-seed or global-optimality claim is supported.','',
    '![All tuning curves](figures/tuning-curves.png)','', '![Easy risk](figures/easy-risk.png)','', '![Hard risk](figures/hard-risk.png)']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    # Standalone LaTeX table and scientific scope for export.
    tex=[r'\documentclass{article}',r'\usepackage[margin=1in]{geometry}',r'\usepackage{longtable,graphicx}',r'\begin{document}',r'\title{Learned credit hedging baseline and specialists}\maketitle',
    'Single shared training seed; entropic gamma10; three learning rates with matched600-active-second trial budgets. Independent development selection precedes8192 paired fresh episodes per component. Baseline P0 is5 percent easy,90 percent ordinary,5 percent hard. Conditional episode uncertainty only.',r'\begin{longtable}{llrrr}',r'Domain & Hedge & Entropic & CVaR95 & MSE \\',r'\hline']
    for d in ('easy','ordinary','hard','base'):
        for n in names:
            a=m[d][n];tex.append(f"{d} & {n.replace('_',r'\_')} & {a['entropic']:.8f} & {a['cvar95']:.8f} & {a['mse']:.8f}"+r' \\')
    tex+=[r'\end{longtable}','No adaptive teachers were trained. Full accounting checks and all41 GPU tests passed. Source/configuration, selection and raw records are retained. The accompanying Markdown report records intervals, tuning histories, costs and numerical limits.',r'\includegraphics[width=\linewidth]{figures/tuning-curves.png}',r'\includegraphics[width=\linewidth]{figures/easy-risk.png}',r'\includegraphics[width=\linewidth]{figures/hard-risk.png}',r'\end{document}']
    (root/'report.tex').write_text('\n'.join(tex)+'\n')
