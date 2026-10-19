def make_report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((OUT/'gate_summary.json').read_text())
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    lines=['# Entropic benchmark development gate','',
           'Single shared training seed 20261017. Paired bootstrap intervals are conditional on trained policies; they do not measure training-seed uncertainty.', '',
           'Accepted candidate: '+str(summary['accepted']), '',
           'Two predeclared BS/SLV families, domain-specific causal variance/regime filtering, stock-only trading, gamma=10, CVaR alpha=0.95. Positive pre-premium normalized losses.', '',
           'Training uses a finite-minibatch log-mean-exp estimator (batch 256), which is not exact population entropic risk. Selection and screening use separate random namespaces. Exponential concentration diagnostics are in metrics.json.', '',
           'Policy features contain only causal returns, moneyness, time, previous position and trading indicator. Liability marks and latent states are excluded. Initial trading costs enter interval zero; liquidation enters the last interval. Pricing uses an approximate variance-grid filter and backward integration checked against refined grids, independent filtering and forward Monte Carlo. GPU validation records actual approximation and accounting errors.', '']
    texrows=[]
    for candidate in summary['results']:
        name=candidate['family']; root=OUT/name
        lines.extend(['## '+name,''])
        for phase in ['screen','confirmation']:
            p=root/phase/'gate.json'
            if p.exists():
                result=json.loads(p.read_text());lines.extend([phase+': '+('PASS' if result['passed'] else 'FAIL'), '',json.dumps(result['checks']), '',json.dumps(result['intervals']), ''])
        lines.extend(['Overall screening outcome: '+str(candidate['screen_passed']),''])
        for phase in ['screen','confirmation']:
            review=root/phase/'residual_review.json'
            if review.exists():lines.extend(['Supplementary '+phase+' marked-residual review: '+review.read_text(),''])
        for domain in ['base','easy','hard']:
            for metric in ['entropic','cvar95','daily_error_std','terminal_sd','centered_entropic','centered_cvar95']:
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
    elif not (OUT/'final_summary.json').exists():
        lines.extend(['## Remaining work','','Benchmark confirmation passed. Freeze accepted configuration and datasets, implement/audit entropic teachers, and run all eight arms before claiming campaign completion.'])
    if (OUT/'final_summary.json').exists():
        final=json.loads((OUT/'final_summary.json').read_text())
        lines.extend(['## Final eight-arm evaluation','','All selections frozen before final episodes. Equal600-active-second budgets, one shared seed. Full scores and paired conditional entropic intervals in final_summary.json.','',json.dumps(final['paired_entropic_intervals'],indent=2),''])
        for domain in ['base','easy','hard']:
            for metric in ['entropic','cvar95','daily_error_std','terminal_sd','centered_entropic','centered_cvar95']:
                fig,ax=plt.subplots(figsize=(10,5))
                for arm in final['scores']:
                    values=[json.loads((OUT/'final_training'/arm/f'selection-{t}'/'metrics.json').read_text())[domain][metric] for t in [0,200,400,600]]
                    ax.plot([0,200,400,600],values,marker='o',label=arm)
                ax.set(xlabel='Active seconds',ylabel=metric,title='Final training selection: '+domain);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(figdir/f'final-{domain}-{metric}.png');plt.close(fig)
        for arm in final['scores']:
            events=[json.loads(line) for line in (OUT/'final_training'/arm/'training.jsonl').read_text().splitlines()]
            if arm in ['baseline','easy_reference','hard_reference']:
                candidate=summary['accepted']; prior=[json.loads(line) for line in (OUT/candidate/arm/'training.jsonl').read_text().splitlines()];events=prior+events
            fig,ax=plt.subplots();ax.plot([e['active_seconds'] for e in events],[np.mean(e['loss']) for e in events]);ax.set(xlabel='Active seconds',ylabel='Block minibatch entropic loss',title=arm);fig.tight_layout();fig.savefig(figdir/f'final-{arm}-training.png');plt.close(fig)
        for domain in ['base','easy','hard']:
            for kind in ['daily','event']:
                fig,axes=plt.subplots(2,1,figsize=(10,7))
                for arm in final['scores']:
                    raw=np.load(OUT/'final_test'/arm/(domain+'.npz'));daily=raw['daily'];switch=raw['switch']
                    if kind=='daily':xx=np.arange(H);means=daily.mean(0);deviation=daily.std(0,ddof=1)
                    else:
                        xx=np.arange(-6,10);means=[];deviation=[]
                        for delta in xx:
                            index=switch+delta;valid=(switch>=0)&(index>=0)&(index<H);values=daily[np.arange(len(switch))[valid],index[valid]]
                            means.append(values.mean() if len(values) else np.nan);deviation.append(values.std(ddof=1) if len(values)>1 else np.nan)
                    axes[0].plot(xx,means,label=arm);axes[1].plot(xx,deviation,label=arm)
                axes[0].set(ylabel='Mean error',title=domain+' '+kind);axes[1].set(ylabel='Cross-episode SD',xlabel='Trading day' if kind=='daily' else 'Day relative to hidden switch');axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(figdir/f'final-{domain}-{kind}.png');plt.close(fig)
            fig,axes=plt.subplots(2,1,figsize=(10,7))
            for arm in final['scores']:
                raw=np.load(OUT/'final_test'/arm/(domain+'.npz'));axes[0].plot(raw['positions'].mean(0),label=arm);axes[1].plot(raw['costs'].mean(0),label=arm)
            axes[0].set(ylabel='Mean stock position',title=domain+' position/cost diagnostics');axes[1].set(xlabel='Trading day',ylabel='Mean transaction cost');axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(figdir/f'final-{domain}-positions-costs.png');plt.close(fig)
        lines.extend(['Empty event-aligned plots in a no-switch domain indicate no qualifying events, not zero risk. Mean loss, initial liability, costs, turnover, and centered terminal risks are retained separately in each metrics.json.'])
        for arm,scores in final['scores'].items():
            for domain,m in scores.items():
                texrows.append(f"Final & {arm.replace('_',' ')} & {domain} & {m['entropic']:.6f} & {m['cvar95']:.6f} & {m['daily_error_std']:.6f} "+chr(92)*2)
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    tex = [r"\documentclass{article}", r"\usepackage[margin=1cm]{geometry}", r"\begin{document}", r"\section*{Single-seed entropic benchmark gate}", "Accepted candidate: "+str(summary['accepted'])+". Uncertainty is conditional on trained policies.", r"\begin{tabular}{llllll}", "Candidate & Arm & Domain & Entropic & CVaR95 & Daily SD " + chr(92)*2]
    tex += texrows + [r"\end{tabular}", r"\end{document}"]
    (OUT/'report.tex').write_text(chr(10).join(tex))
    # Hash every scientific artifact only after files have been closed.
    manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and not p.name.endswith('.tmp') and p.name not in ['gpu.csv','artifact_hashes.json']}
    save(OUT/'artifact_hashes.json',manifest)
