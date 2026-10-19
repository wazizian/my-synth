"""Development-screen figures and LaTeX source from retained worker artifacts."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(root):
    root = Path(root)
    metrics = json.loads((root/'all_metrics.json').read_text())
    gate = json.loads((root/'gate.json').read_text())
    figures = root/'figures'
    figures.mkdir(exist_ok=True)
    for domain, values in metrics.items():
        fig,axes = plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
        for ax,metric,title in zip(axes,('entropic','cvar95','daily_error_std'),('Entropic loss','Terminal CVaR95','Daily error SD')):
            ax.bar(range(len(values)),[v[metric] for v in values.values()])
            ax.set_xticks(range(len(values)),list(values),rotation=60,ha='right',fontsize=7)
            ax.set_title(title);ax.set_ylabel('Bond-face units')
        fig.suptitle(domain+' — development scenarios, fixed policies')
        fig.savefig(figures/f'{domain}-risk.png',dpi=140);plt.close(fig)
        fig,axes = plt.subplots(2,1,figsize=(10,7),sharex=True,constrained_layout=True)
        for name,m in values.items():
            axes[0].plot(m['daily_mean'],label=name,linewidth=.7)
            axes[1].plot(m['daily_std'],label=name,linewidth=.7)
        axes[0].set_ylabel('Daily mean loss');axes[1].set_ylabel('Daily error SD')
        axes[1].set_xlabel('Trading day; absorbing zeros after default included')
        axes[0].legend(fontsize=7,ncol=2);fig.suptitle(domain)
        fig.savefig(figures/f'{domain}-daily.png',dpi=140);plt.close(fig)
    for domain in ('easy','ordinary','hard'):
        curves=np.load(root/domain/'initial_curves.npz')
        fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        axes[0].plot(np.arange(len(curves['survival']))/252,curves['survival'])
        axes[0].set_ylabel('Survival probability');axes[0].set_xlabel('Years')
        axes[1].plot(np.arange(len(curves['bond']))/252,curves['bond'])
        axes[1].set_ylabel('Bond value per face');axes[1].set_xlabel('Years')
        fig.suptitle(domain);fig.savefig(figures/f'{domain}-initial-curves.png',dpi=140);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
        inventory=np.load(root/domain/'replenish_1.0-inventory-mean.npz')['mean_inventory']
        im=ax.imshow(inventory.T,origin='lower',aspect='auto')
        ax.set_xlabel('Trading day');ax.set_ylabel('Vintage age in days')
        fig.colorbar(im,ax=ax,label='Mean surviving notional')
        ax.set_title(domain+' — fixed target 1.0, replenishment')
        fig.savefig(figures/f'{domain}-inventory.png',dpi=140);plt.close(fig)
    outcome=('The frozen economic pre-screen found a common policy within the useful-gain tolerance on both targets. Stop before learned-specialist and adaptive training.'
             if gate['stop_uninformative'] else
             'The economic pre-screen did not establish a common near-best policy on both targets. The learned-specialist gate remains required; no adaptive or final-test result is established.')
    rows=[]
    for domain,values in metrics.items():
        for name,m in values.items():
            rows.append((domain,name,m['entropic'],m['cvar95'],m['daily_error_std'],m['mean_cost']))
    header='| Domain | Policy | Entropic | CVaR95 | Daily SD | Cost |\n|---|---|---:|---:|---:|---:|\n'
    table=''.join('| %s | %s | %.6f | %.6f | %.6f | %.6f |\n'%r for r in rows)
    limits='''The model uses daily default timing, positive transformed OU factors, and custom daily-coupon par CDS. Executable quotes come from an informed dealer; the policy receives quotes and inventory, not factor labels. Quotes may reveal factor information. Ordinary is a component; base is the declared 5% easy / 90% ordinary / 5% hard mixture. Results use positive discounted financed-bond loss per unit face. Daily dispersion includes absorbing zeros after default. Paired intervals resample whole episodes and describe evaluation noise for fixed diagnostic policies; they are not training-seed uncertainty. No learned policy has been trained in this pre-screen. Numerical refinement covers declared probe states, not a uniform continuous-time error bound. Full daily trade/cash/position records and paths are retained; full vintage traces are retained for eight representative paths and mean vintage surfaces for all paths. Source and deterministic rules permit reconstruction of the remaining vintage histories.'''
    (root/'report.md').write_text('# Credit-curve economic pre-screen\n\n'+outcome+'\n\n'+header+table+'\n'+limits+'\n')
    def esc(s):return s.replace('_',r'\_').replace('%',r'\%')
    texrows='\n'.join('%s & %s & %.6f & %.6f & %.6f & %.6f \\\\'%(esc(r[0]),esc(r[1]),*r[2:]) for r in rows)
    tex=r'''\documentclass{article}
\usepackage[margin=1in]{geometry}
\usepackage{longtable,graphicx}
\begin{document}
\title{Credit-curve economic pre-screen}\maketitle
'''+esc(outcome)+r'''
\begin{longtable}{llrrrr}
Domain & Policy & Entropic & CVaR95 & Daily SD & Cost \\
\hline
'''+texrows+r'''
\end{longtable}
'''+esc(limits)+r'''
\end{document}
'''
    (root/'report.tex').write_text(tex)
    await_export(root.parent)


def await_export(output_root):
    """Keep the paid worker available until the supervisor verifies exports.

    This guard changes neither numerical calculations nor gate decisions.
    The original launcher retains its independent maximum-lifetime watchdog.
    """
    import hashlib
    import os
    import time
    output_root=Path(output_root)
    records=[]
    for path in sorted(output_root.rglob('*')):
        if not path.is_file() or path.name=='gpu.csv' or '.tmp' in path.name:
            continue
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):
                digest.update(block)
        records.append(dict(path=str(path.relative_to(output_root)),bytes=path.stat().st_size,sha256=digest.hexdigest()))
    manifest=output_root/'export_ready.json'
    temporary=output_root/'export_ready.tmp.json'
    temporary.write_text(json.dumps(dict(files=records,scope='Immutable outputs before final launcher completion; live GPU telemetry excluded'),indent=2))
    temporary.replace(manifest)
    print('EXPORT_READY: waiting for supervisor hash verification and export_ack.json',flush=True)
    deadline=time.monotonic()+3600
    while not (output_root/'export_ack.json').exists():
        if time.monotonic()>deadline:
            raise RuntimeError('Supervisor export acknowledgement timed out')
        time.sleep(5)
