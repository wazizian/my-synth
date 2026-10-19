def make_report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((OUT/'gate_summary.json').read_text())
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    lines=['# Entropic benchmark development gate','',
           'Single shared training seed 20261017. Paired bootstrap intervals are conditional on trained policies; they do not measure training-seed uncertainty.', '',
           'Accepted candidate: '+str(summary['accepted']), '',
           'Two predeclared trinomial candidates, common causal Bayesian liability pricing, stock-only trading, gamma=10, CVaR alpha=0.95. Positive pre-premium normalized losses.', '',
           'Training uses a finite-minibatch log-mean-exp estimator (batch 256), which is not exact population entropic risk. Selection and screening use separate random namespaces. Exponential concentration diagnostics are in metrics.json.', '',
           'Policy features contain only causal returns, moneyness, time, liability mark, previous position and trading indicator. Initial trading costs enter interval zero; liquidation enters the last interval. Exact mathematical pricing is evaluated numerically; GPU validation records observed accounting errors.', '']
    texrows=[]
    for candidate in summary['results']:
        name=candidate['candidate']; root=OUT/name
        lines.extend(['## '+name,''])
        for phase in ['screen','confirmation']:
            p=root/phase/'gate.json'
            if p.exists():
                result=json.loads(p.read_text());lines.extend([phase+': '+('PASS' if result['passed'] else 'FAIL'), '',json.dumps(result['checks']), '',json.dumps(result['paired_intervals']), ''])
        for domain in PROBS:
            for metric in ['entropic','cvar95','daily_error_std']:
                fig,ax=plt.subplots()
                for arm in ['baseline','easy_reference','hard_reference']:
                    points=[json.loads((root/arm/f'selection-{t}'/'metrics.json').read_text())[domain][metric] for t in [0,60,120,180]]
                    ax.plot([0,60,120,180],points,marker='o',label=arm)
                ax.set(xlabel='Active training seconds (scheduled checkpoint)',ylabel=metric,title=f'{name}: {domain} selection');ax.legend();fig.tight_layout();fig.savefig(figdir/f'{name}-{domain}-{metric}.png');plt.close(fig)
            for kind in ['daily','event']:
                fig,axes=plt.subplots(2,1,figsize=(8,7))
                for arm in ['baseline','easy_reference','hard_reference']:
                    raw=np.load(root/'screen'/arm/(domain+'.npz'));daily=raw['daily'];sw=raw['switch']
                    if kind=='daily':
                        x=np.arange(H);means=daily.mean(0);sd=daily.std(0,ddof=1)
                    else:
                        x=np.arange(-6,8);means=[];sd=[]
                        for delta in x:
                            indices=sw+delta;valid=(sw>=0)&(indices>=0)&(indices<H)
                            v=daily[np.arange(len(sw))[valid],indices[valid]]
                            means.append(v.mean() if len(v) else np.nan);sd.append(v.std(ddof=1) if len(v)>1 else np.nan)
                    axes[0].plot(x,means,label=arm);axes[1].plot(x,sd,label=arm)
                axes[0].set(ylabel='Mean daily error',title=f'{name}: {domain} {kind}');axes[1].set(ylabel='Cross-episode SD',xlabel='Trading day' if kind=='daily' else 'Day relative to hidden switch');axes[0].legend();fig.tight_layout();fig.savefig(figdir/f'{name}-{domain}-{kind}.png');plt.close(fig)
        for arm in ['baseline','easy_reference','hard_reference']:
            events=[json.loads(x) for x in (root/arm/'training.jsonl').read_text().splitlines()]
            fig,ax=plt.subplots();ax.plot([x['active_seconds'] for x in events],[np.mean(x['loss']) for x in events]);ax.set(xlabel='Active training seconds',ylabel='Mean block minibatch entropic loss',title=f'{name}: {arm}');fig.tight_layout();fig.savefig(figdir/f'{name}-{arm}-training.png');plt.close(fig)
            scores=json.loads((root/'screen'/arm/'metrics.json').read_text())
            for d,m in scores.items():
                texrows.append(f"{name} & {arm.replace('_',' ')} & {d} & {m['entropic']:.6f} & {m['cvar95']:.6f} & {m['daily_error_std']:.6f} \\\\")
    if summary['accepted'] is None:
        lines.extend(['## Outcome','','No candidate passed the predeclared gate. This is an uninformative benchmark for the proposed adaptive comparison. Adaptive training and final-test evaluation were not run. The failure is retained without changing the candidate menu or thresholds.'])
    else:
        lines.extend(['## Remaining work','','Benchmark confirmation passed. Freeze accepted configuration and datasets, implement/audit entropic teachers, and run all eight arms before claiming campaign completion.'])
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    tex = [r"\documentclass{article}", r"\usepackage[margin=1cm]{geometry}", r"\begin{document}", r"\section*{Single-seed entropic benchmark gate}", "Accepted candidate: "+str(summary['accepted'])+". Uncertainty is conditional on trained policies.", r"\begin{tabular}{llllll}", "Candidate & Arm & Domain & Entropic & CVaR95 & Daily SD " + chr(92)*2]
    tex += texrows + [r"\end{tabular}", r"\end{document}"]
    (OUT/'report.tex').write_text(chr(10).join(tex))
    # Hash every scientific artifact only after files have been closed.
    manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and not p.name.endswith('.tmp') and p.name not in ['gpu.csv','artifact_hashes.json']}
    save_json(OUT/'artifact_hashes.json',manifest)
