"""Build unified credit/options report from immutable exported experiment evidence."""
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl-hedging-report')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'hetzner-results-20261019'
SUP = HERE / 'supplement'
CREDIT = DATA / 'campaigns/comparison-20261018/credit-final/evaluation/launcher/runs/20261019T021407Z-6f5fb002/outputs/evaluation'
OPTION = DATA / 'campaigns/volatility-shocks-20261018'
COMP = DATA / 'campaigns/comparison-20261018'
ARMS = ['baseline','easy_reference','hard_reference','dpg_hard','soar_hard','selfplay','dpg_mixed','soar_mixed']
LABEL = dict(zip(ARMS,['Baseline','Easy specialist','Hard specialist','DPG hard','SOAR hard','Self-play','DPG mixed','SOAR mixed']))
OLABEL = {'baseline':'Baseline','easy':'Easy specialist','hard':'Hard specialist','static':'Tuned constant','bs':'BS delta (20%)'}
COLORS = dict(zip(ARMS,['#333333','#16866b','#d25b33','#3979ae','#6b65a5','#aa60a0','#008fa5','#b79124']))
DOM = ['easy','ordinary','hard']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'pdf.fonttype':42,'savefig.bbox':'tight'})
USED = set()


def read(path):
    """Read a JSON evidence file and record its path for provenance."""
    USED.add(Path(path).resolve())
    return json.loads(Path(path).read_text())


def risk(x):
    """Compute stable empirical entropic risk at gamma 10 from terminal loss samples."""
    x=np.asarray(x); z=10*x; m=z.max()
    return float((m+np.log(np.mean(np.exp(z-m))))/10)


def metrics(x):
    """Return entropic risk, fractional-tail CVaR95, population SD and mean for one loss vector."""
    x=np.asarray(x); mass=.05*len(x); k=int(mass); tail=np.sort(x)[::-1]
    cvar=(tail[:k].sum()+(mass-k)*tail[k])/mass
    return dict(entropic=risk(x),cvar95=float(cvar),sd=float(x.std()),mean=float(x.mean()))


def daily_pool(summary):
    """Reconstruct pooled sample SD from credit's per-date means and sample SDs and episode count."""
    mu=np.asarray(summary['mean']); sd=np.asarray(summary['sd']); count=len(mu)*summary['episodes']
    return float(np.sqrt((np.mean(sd**2)*(summary['episodes']-1)/summary['episodes']+np.var(mu))*count/(count-1)))


def save(fig,name):
    """Write one vector PDF and preview PNG and close the figure."""
    fig.savefig(HERE/'figures'/f'{name}.pdf')
    fig.savefig(HERE/'figures'/f'{name}.png',dpi=160)
    plt.close(fig)


def tex_escape(s):
    """Escape plain text identifiers used inside LaTeX tables."""
    return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')


def table(headers, rows, spec=None, size='small'):
    """Render a booktabs LaTeX table with preformatted cells."""
    spec=spec or ('l'+'r'*(len(headers)-1))
    return '\\begin{'+size+'}\n\\begin{tabular}{'+spec+'}\n\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\n'+'\n'.join(' & '.join(re.sub(r'(?<!\\)%', lambda m: r'\%', str(cell)) for cell in row)+r' \\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n\\end{'+size+'}\\par\\medskip\n'


def figure(name,caption,width='.98',height=None):
    """Render a fixed-position report figure with a descriptive caption."""
    opts='width='+width+r'\linewidth'
    if height:opts+=',height='+height+r'\textheight,keepaspectratio'
    return '\\begin{figure}[H]\\centering\n\\includegraphics['+opts+']{figures/'+name+'.pdf}\n\\caption{'+caption+'}\n\\end{figure}\n'


def page(title):
    """Start a new report page with a section heading."""
    return '\n\\clearpage\n\\section{'+title+'}\n'


def main():
    """Validate exported scores, harmonize metrics, make figures/tables and write report.tex."""
    c=read(CREDIT/'results.json');o=read(OPTION/'results.json');diag=read(OPTION/'diagnostic.json')
    gates={bp:read(OPTION/f'gate-{bp}bp-results.json') for bp in [1,3,10]}
    budgets=read(CREDIT/'training-budgets.json')['trials']; frozen=read(CREDIT/'frozen-evaluation-manifest.json')
    rows=[]; validations=[]
    for domain in DOM:
        p=CREDIT/f'terminal-{domain}.npz';USED.add(p.resolve());a=np.load(p)
        for arm in ARMS:
            m=c['domains'][domain]['terminal'][arm]; re=metrics(a[arm])
            for key,original in [('entropic','entropic'),('cvar95','cvar95'),('sd','terminal_std'),('mean','mean')]:
                assert min(abs(re[key]-m[original]), abs(np.std(a[arm],ddof=1)-m[original]) if key=='sd' else float('inf'))<1e-11,(domain,arm,key)
            d=c['domains'][domain]['daily_marked_errors'][arm]
            rows.append(dict(experiment='credit',stage='sealed final',cost_bp=10,domain=domain,method=arm,
                             entropic_bp=m['entropic']*1e4,cvar95_bp=m['cvar95']*1e4,terminal_sd_bp=re['sd']*1e4,
                             daily_pooled_sd_bp=daily_pool(d)*1e4,mean_loss_bp=m['mean']*1e4,mean_cost_bp=m['mean_cost']*1e4,
                             terminal_n=8192,daily_n=8192))
    for bp in [1,3,10]:
        for domain in ['easy','base','hard']:
            a_path=next(DATA.rglob(f'cost-{bp}bp-{domain}-terminal.npz'));USED.add(a_path.resolve());a=np.load(a_path)
            for arm in OLABEL:
                m=o['costs'][f'cost-{bp}bp'][domain]['terminal'][arm]; re=metrics(a[arm])
                for key in ['entropic','mean']:assert abs(re[key]-m[key])<1e-11
                assert abs(np.sort(a[arm])[int(.95*len(a[arm])):].mean()-m['cvar95'])<1e-11
                # Option source uses sample ddof=1 for terminal dispersion.
                assert min(abs(re['sd']-m['sd']),abs(np.std(a[arm],ddof=1)-m['sd']))<1e-11
                d=o['costs'][f'cost-{bp}bp'][domain]['daily_marked_errors'][arm]
                rows.append(dict(experiment='option',stage='820 diagnostic',cost_bp=bp,domain=domain,method=arm,
                                 entropic_bp=m['entropic']*1e4,cvar95_bp=re['cvar95']*1e4,terminal_sd_bp=re['sd']*1e4,
                                 daily_pooled_sd_bp=d['sd']*1e4,mean_loss_bp=m['mean']*1e4,mean_cost_bp=m['mean_cost']*1e4,
                                 terminal_n=8192,daily_n=256))
    with (HERE/'tables/metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lookup={(r['experiment'],r['cost_bp'],r['domain'],r['method']):r for r in rows}
    # Harmonized credit risk overview: all methods, all target domains, same metric order.
    fig,axs=plt.subplots(3,3,figsize=(12,9.7))
    for j,d in enumerate(DOM):
        for i,(key,title) in enumerate([('entropic_bp','Entropic risk'),('cvar95_bp','CVaR95'),('daily_pooled_sd_bp','Daily pooled SD')]):
            ax=axs[i,j]; vals=[lookup['credit',10,d,a][key] for a in ARMS]
            ax.scatter(vals,np.arange(8),c=[COLORS[a] for a in ARMS],s=35)
            ax.axvline(vals[0],color='#666',ls=':',lw=1);ax.set_yticks(np.arange(8),[LABEL[a] if j==0 else '' for a in ARMS]);ax.invert_yaxis()
            ax.set_xlabel('bp of bond face');ax.set_title(f'{d.capitalize()} · {title}',fontsize=10)
    fig.tight_layout();save(fig,'credit_metrics')
    # Paired intervals for gains, both baseline and own-domain specialist references.
    for reference,key in [('baseline','baseline_minus'),('specialist','specialist_minus')]:
        methods=ARMS[1:] if reference=='baseline' else ARMS[3:]
        fig,axs=plt.subplots(1,2,figsize=(10,4.1),sharey=True)
        for ax,d in zip(axs,['easy','hard']):
            for i,a in enumerate(methods):
                g=c['domains'][d][key][a];point=1e4*g['gain'];lo,hi=np.array(g['ci95'])*1e4
                ax.plot([lo,hi],[i,i],color=COLORS[a],lw=2);ax.scatter(point,i,color=COLORS[a],s=30)
            ax.axvline(0,color='black',lw=1);ax.set_yticks(range(len(methods)),[LABEL[a] for a in methods]);ax.invert_yaxis();ax.set_title(d.capitalize());ax.set_xlabel('Entropic improvement (bp); right is better')
        fig.tight_layout();save(fig,'credit_gain_'+reference)
    # Credit all-LR cross-domain development curves, comparable balanced risk in bp.
    fig,axs=plt.subplots(4,2,figsize=(11,10),sharex=True)
    lrcolors={.0001:'#2166ac',.0003:'#1b9e77',.001:'#d95f02'}
    selection=[]
    for ax,arm in zip(axs.flat,ARMS):
        for lr,color in lrcolors.items():
            folder=next((SUP/'comparison-20261018/credit-final'/arm).glob(f'launcher/runs/*/outputs/training/lr-{lr:g}'))
            points=[]
            for f in folder.glob('selection-*.json'):
                sec=float(f.stem.split('-')[-1]);v=read(f);points.append((sec,.5*(v['easy']+v['hard'])*1e4))
            points.sort();ax.plot(*np.array(points).T,marker='o',ms=3,color=color,label=f'{lr:g}')
            sel=read(folder/'selection.json');selection.append(sel)
        ax.set_title(LABEL[arm],fontsize=10);ax.set_ylabel('Balanced dev. risk (bp)');ax.set_xlabel('Active training seconds')
    axs[0,0].legend(title='Learning rate',fontsize=8);fig.tight_layout();save(fig,'credit_learning')
    # Credit calendar daily SD and protection profiles, using selected methods only for readability.
    chosen=['baseline','easy_reference','hard_reference','dpg_mixed']
    fig,axs=plt.subplots(2,3,figsize=(11,6),sharex=True)
    for j,d in enumerate(DOM):
        for arm in chosen:
            dd=c['domains'][d]['daily_marked_errors'][arm];sd=np.asarray(dd['sd'])*1e4
            # Plot non-overlapping 21-day RMS bins; do not smooth across defaults within paths.
            vals=[np.sqrt(np.mean(sd[k:k+21]**2)) for k in range(0,len(sd),21)]
            t=(np.arange(len(vals))*21+10.5)/252
            axs[0,j].plot(t,vals,label=LABEL[arm],color=COLORS[arm])
            holdings=c['domains'][d]['holdings_by_vintage'][arm]['mean_notional_by_vintage']
            axs[1,j].plot(np.arange(len(holdings)),holdings,label=LABEL[arm],color=COLORS[arm])
        axs[0,j].set_title(d.capitalize());axs[0,j].set_xlabel('Years');axs[1,j].set_xlabel('CDS vintage age (days)')
    axs[0,0].set_ylabel('21-day RMS of daily SD (bp)');axs[1,0].set_ylabel('Mean notional by vintage')
    axs[0,0].legend(fontsize=7);fig.tight_layout();save(fig,'credit_daily_holdings')
    # Option event study: only actual exported time points (future random arrivals included).
    fig,axs=plt.subplots(1,3,figsize=(11,3.3),sharey=True)
    for ax,d in zip(axs,['easy','base','hard']):
        points=diag['event_summary'][d];days=sorted(map(int,points))
        ax.plot(days,[100*points[str(t)]['rv'] for t in days],'-o',color='#c65a12',label='5-day RV')
        ax.plot(days,[100*points[str(t)]['iv'] for t in days],'-s',color='#2468b4',label='21-day ATM IV')
        ax.set_title(d.capitalize());ax.set_xlabel('Trading days after conditioned event');ax.set_ylim(15,46)
    axs[0].set_ylabel('Annualized volatility (%)');axs[0].legend();fig.tight_layout();save(fig,'option_event')
    # Independent screen and confirmation gate intervals; threshold at 5bp.
    fig,axs=plt.subplots(1,2,figsize=(10,4),sharex=True)
    for ax,d in zip(axs,['easy','hard']):
        for j,bp in enumerate([1,3,10]):
            for shift,stream,color in [(-.12,'screen','#7297b9'),(.12,'confirmation','#b45735')]:
                g=gates[bp]['streams'][stream][d];lo,hi=np.array(g['ci95'])*1e4
                ax.plot([lo,hi],[j+shift]*2,color=color,lw=2);ax.scatter(g['gain']*1e4,j+shift,color=color,label=stream if j==0 else None,s=25)
        ax.axvline(0,color='black',lw=.8);ax.axvline(5,color='#aa2222',ls='--',lw=1,label='Required lower bound: 5 bp')
        ax.set_yticks(range(3),['1 bp cost','3 bp cost','10 bp cost']);ax.invert_yaxis();ax.set_title(d.capitalize());ax.set_xlabel('Baseline minus own specialist risk (bp)')
    axs[1].legend(fontsize=8,loc='lower right');fig.tight_layout();save(fig,'option_gates')
    # Option normalized metric views: percent changes vs baseline, retaining absolute tables.
    fig,axs=plt.subplots(3,3,figsize=(11,8.8))
    for j,bp in enumerate([1,3,10]):
        for i,(key,title) in enumerate([('entropic_bp','Entropic'),('cvar95_bp','CVaR95'),('daily_pooled_sd_bp','Daily pooled SD')]):
            ax=axs[i,j]
            for d,color in [('easy','#16866b'),('base','#777777'),('hard','#d25b33')]:
                base=lookup['option',bp,d,'baseline'][key]
                values=[100*(lookup['option',bp,d,a][key]-base)/abs(base) for a in ['easy','hard','bs']]
                ax.plot(range(3),values,'o-',color=color,label=d)
            ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(3),['Easy spec.','Hard spec.','BS delta'],rotation=20)
            ax.set_title(f'{bp} bp cost · {title}',fontsize=10);ax.set_ylabel('Change / |baseline| (%)')
    axs[0,0].legend(fontsize=8);fig.tight_layout();save(fig,'option_metrics')
    # Option own-domain dev curves, subtract the fixed domain initial premium before plotting.
    fig,axs=plt.subplots(3,3,figsize=(11,8.6),sharex=True)
    for i,arm in enumerate(['baseline','easy','hard']):
        domain={'baseline':'base','easy':'easy','hard':'hard'}[arm];premium=o['initial_premiums'][domain]
        for j,bp in enumerate([1,3,10]):
            ax=axs[i,j]
            for lr,color in lrcolors.items():
                folder=next((SUP/'volatility-shocks-20261018/launchers'/arm).glob(f'runs/*/outputs/training/cost-{bp}bp/lr-{lr:g}'))
                points=[]
                if (folder/'checkpoint.json').exists():
                    checkpoint=read(folder/'checkpoint.json')
                else:
                    checkpoint={'history':[]}
                    for p in sorted(folder.glob('dev-*.npz')):
                        USED.add(p.resolve());loss=np.load(p)['loss']
                        checkpoint['history'].append({'active_seconds':float(p.stem.split('-')[-1]),'entropic':risk(loss)})
                for h in checkpoint['history']:points.append((h['active_seconds'],1e4*(h['entropic']-premium)))
                points.sort();ax.plot(*np.array(points).T,marker='o',ms=3,color=color,label=f'{lr:g}')
                # Loss-array risk must agree with checkpoint risk, including the premium convention.
                for h in checkpoint['history']:
                    sec=min([60,180,360,600,1200,1800],key=lambda t:abs(t-h['active_seconds']))
                    p=folder/f'dev-{sec}.npz';USED.add(p.resolve());a=np.load(p)
                    assert abs(risk(a['loss'])-h['entropic'])<1e-10
            ax.set_title(f'{OLABEL[arm]} · {bp} bp cost',fontsize=10);ax.set_xlabel('Active training seconds');ax.set_ylabel('Own-domain dev. risk (bp)')
    axs[0,0].legend(title='Learning rate',fontsize=8);fig.tight_layout();save(fig,'option_learning')
    # Static comparison and loss decomposition, all costs.
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    for ax,d in zip(axs,['easy','hard']):
        for a,color in [('static','#999999'),('baseline','#333333'),(d,'#16866b' if d=='easy' else '#d25b33'),('bs','#3979ae')]:
            ax.plot([1,3,10],[lookup['option',bp,d,a]['entropic_bp'] for bp in [1,3,10]],'o-',label=OLABEL[a],color=color)
        ax.set_title(d.capitalize());ax.set_xlabel('Transaction cost (bp)');ax.set_ylabel('Empirical entropic risk (bp)')
    axs[0].legend(fontsize=8);fig.tight_layout();save(fig,'option_static')
    # Main text, with explicit page breaks to keep tables readable.
    t=r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=19mm]{geometry}
\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{microtype}
\usepackage{amsmath,amssymb,booktabs,array,longtable,graphicx,float,xcolor,hyperref,fancyhdr}\setlength{\emergencystretch}{2em}
\definecolor{navy}{HTML}{203A52}\hypersetup{colorlinks=true,linkcolor=navy,urlcolor=navy,pdftitle={Credit and option hedging: unified experimental report}}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\renewcommand{\arraystretch}{1.15}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small Credit and option hedging}\fancyhead[R]{\small 19 October 2026}\fancyfoot[C]{\thepage}\setlength{\headheight}{14pt}
\newcommand{\bp}{\mathrm{bp}}\newcommand{\risk}{\widehat R_{10}}\newcommand{\cvar}{\widehat{\mathrm{CVaR}}_{.95}}
\begin{document}
{\Huge\bfseries Credit and option hedging}\par
{\Large Unified experimental report}\par
{\large Results available on 19 October 2026}\par\vspace{5mm}
\textbf{Main finding.} These experiments have learned useful hedges, but have not established the desired clear advantage of specialists on \emph{both} easy and hard targets. Consequently they do not yet provide a clean benchmark for adaptive training.

\begin{center}
'''
    t+=table(['Experiment','Evaluation evidence','Conclusion'],[
        ['Credit','Sealed final; 8,192 paths/domain','Adaptive superiority inconclusive'],
        ['Options','Screen/confirmation + diagnostic','Hard gain; easy gap absent']],spec='lll')
    t+=r'''\end{center}
\textbf{Credit.} All eight arms completed three learning-rate trials. DPG mixed improves hard-domain entropic risk relative to baseline by 0.853 bp, with paired 95\% interval [0.057, 1.628] bp. Its improvement relative to the hard specialist is 0.328 bp [\(-0.174\), 0.861], which is inconclusive. All adaptive-versus-own-specialist intervals include zero.

\textbf{Options.} The hidden-IV/price-history-only design produces the intended volatility separation. Hard specialists improve entropic scores, but easy specialists do not reliably improve over baseline at 1, 3 or 10 bp transaction cost. No cost passes the joint gate. No adaptive option arm was run and the sealed final stream was not opened.

\textbf{How to read this report.} A positive entropic improvement always means reference risk minus method risk, so positive is better. All monetary risks and costs are reported in basis points: one bp is \(10^{-4}\) of bond face for credit or initial spot for options. Lower absolute risk, CVaR and dispersion are better. Negative option scores can arise because the domain-specific Q premium exceeds physical expected hedging costs; negative loss means profit. Absolute values across the two markets are not directly comparable: contracts, horizons and normalization differ.

\textbf{Evidence boundaries.} There is one shared initialization, not multiple training seeds. Confidence intervals reflect paired episode uncertainty conditional on fitted policies. Credit results are exploratory because its discovery gates failed. Option entropic scores are empirical: the frozen model has an infinite population exponential-loss moment at positive liquidation costs (Section~\ref{sec:limitations}). No new training was performed for this report.

\vfill
\textit{Includes consistent metric tables, paired comparisons, learning curves for every learning rate, IV/RV diagnostics, costs, training budgets, selected checkpoints and reproducibility records.}
'''
    t+=page('Common definitions and reporting conventions')
    t+=r'''Let \(L_i\) denote terminal hedging loss on episode \(i\), including trading costs. Positive loss is adverse. The common metrics are
\begin{align*}
\risk(L)&=\frac1{10}\log\!\left(\frac1N\sum_{i=1}^N e^{10L_i}\right),\\
\cvar(L)&=\frac{\sum_{j=1}^{k}L_{[j]}+(m-k)L_{[k+1]}}{m},
\quad m=.05N,\quad k=\lfloor m\rfloor,\\
\widehat\sigma_T&=\sqrt{\frac1N\sum_i(L_i-\overline L)^2},
\end{align*}
where \(L_{[1]}\geq\cdots\geq L_{[N]}\). Thus CVaR averages the worst 5\% with fractional boundary mass; it is not the 95th percentile. Option source CVaR averages the worst 410 paths; here it is recomputed using fractional mass 409.6 to match credit exactly. Terminal SD uses population normalization \(1/N\) consistently in this report; source SDs using \(1/(N-1)\) were recomputed from retained terminal arrays.

\textbf{Daily metric harmonization.} Let \(e_{t,i}\) be the marked daily hedge loss, \(T\) the fixed calendar and \(\bar e\) its pooled mean. The main tables use
\[
\widehat\sigma_{\rm day,pool}=\sqrt{\frac{\sum_{t,i}(e_{t,i}-\bar e)^2}{NT-1}}.
\]
Credit originally reports \(\sqrt{T^{-1}\sum_t\operatorname{Var}_i(e_{t,i})}\); options originally report the pooled SD above. They are different statistics. For credit the common pooled SD is reconstructed exactly from exported daily means \(\mu_t\) and sample SDs \(s_t\):
\[
\widehat\sigma_{\rm day,pool}^2=\frac{NT}{NT-1}
\left[\frac{N-1}{N}\frac1T\sum_t s_t^2+\operatorname{Var}_t(\mu_t)\right].
\]
This changes presentation, not any fitted model or gate decision. Daily dispersion includes transaction costs. It measures marked fluctuations; it is not terminal risk, nor an average of individual paths' time-series SDs.

\begin{center}
'''
    t+=table(['Quantity','Credit','Options'],[
        ['Terminal metrics','8,192 episodes/domain','8,192 episodes/domain/cost'],
        ['Daily marked metrics','8,192 episodes/domain','256 episodes/domain'],
        ['Calendar','1,260 trading days','126 trading days'],
        ['Daily pricing','Finite-state tables','4,096 conditional Q paths/quote'],
        ['Reporting stage','Sealed final','820 diagnostic; gates separately']],spec='lll')
    t+=r'''\end{center}
\textbf{Mean versus dispersion.} Entropic risk combines mean loss and tail dispersion; CVaR is also sensitive to mean. Lower SD alone need not mean lower economic risk. We report mean loss and cost alongside all three requested risk metrics rather than switching metrics to favor a method.

\textbf{Uncertainty.} All quoted intervals are 1,000 paired whole-episode bootstrap comparisons from the source experiment. They are pointwise, with no multiplicity adjustment. No confidence interval for an absolute score or a different metric is implied. We do not bootstrap individual days as independent observations.
'''
    t+=page('Contracts, information and training protocol')
    t+=r'''\subsection*{Credit: financed five-year ZCB, rolling six-month CDS}
The portfolio is a long defaultable zero-coupon bond of face one, financed at its initial model value. Daily CDS issuance creates 126 active vintage slots under 252 trading days/year. The agent buys the newest six-month CDS, retains old vintages, or sells some/all of them. It holds long-only protection with a total-notional cap of two. Each vintage retains its original fixed coupon. Default pays bond recovery 40\% and CDS loss-given-default 60\%; rates are fixed at 2\%. Daily coupons include the default day. Final liquidation and proportional transaction costs enter the ledger. These are custom daily CDS conventions, not ISDA contracts.

The latent intensity is \(\lambda_t=\log(1+e^{X_t+Y_t})\), with independent mean-reverting OU factors. Easy and ordinary have a single active factor; hard adds a faster factor. A finite grid defines both simulation and pricing; end-of-day default probability is \(1-e^{-\lambda_t/252}\). The executable new-CDS quote and old-vintage marks are observed by the recurrent policy. Factors and future states are not inputs. Quotes may reveal factor information: hidden factors do not establish partial observability by themselves.

The baseline trains on \emph{ordinary only}, not the earlier 5/90/5 empirical mixture. The final comparison uses the ordinary-credit candidate at 10 bp cost, not the separate high-default candidate. Domain names are historical hypotheses: the reported hard domain has lower residual risk than easy under these fitted policies.

\subsection*{Options: one short six-month ATM call}
The call has initial spot/strike one, 126 daily intervals, zero rates and stock/cash hedging. Holdings lie in \([0,1]\). Costs of 1, 3 and 10 bp apply to traded stock notional, including opening and final liquidation. The model receives only causal normalized stock-price history, time to expiry and previous holding. No IV, RV, option marks, factors, arrivals or domain labels are inputs. Policy memory starts at issuance; there is no warmup.

\subsection*{Shared learning and fair scope}
Both use recurrent GRU learners of hidden size 32 within their respective markets. Architectures and observations are identical \emph{within} each experiment; the two contracts have different output dimensions. Gamma is 10. Each arm tries Adam learning rates \(10^{-4},3\times10^{-4},10^{-3}\), with 1,800 active seconds per trial. Checkpoint/development decisions precede the reported evaluation stream. Initialization seed is 20261017, training streams 220261017 and development streams 320261017. These distinct stream IDs are not additional training-seed replicates.

Credit minibatches have 160 paths; teacher blocks contain eight updates. Option minibatches have 256 paths. Credit baseline selects on ordinary, specialists on their domain, and adaptive arms on fixed scaled easy/hard development risk. Option models select on their own training domain. Equal active-time budgets include teacher overhead; they do not imply equal updates or training paths.
'''
    t+=page('Credit: full terminal and daily results')
    t+=r'''\textbf{Sealed final evaluation.} All values below are bp of bond face. The column order is fixed throughout the report. Daily SD is the harmonized pooled measure; it includes absorbing zero increments after default. Terminal and daily metrics use 8,192 episodes/domain.
'''
    for d in DOM:
        t+='\n\\subsection*{'+d.capitalize()+'}\n'
        rr=[]
        for a in ARMS:
            r=lookup['credit',10,d,a];rr.append([LABEL[a]]+[f"{r[k]:.3f}" for k in ['entropic_bp','cvar95_bp','terminal_sd_bp','daily_pooled_sd_bp','mean_cost_bp']])
        t+=table(['Method','Entropic','CVaR95','Terminal SD','Daily SD','Cost'],rr,size='footnotesize')
    t+=r'''\textbf{Reading the rankings.} DPG mixed has the smallest hard entropic point estimate, but SOAR hard has the smallest hard CVaR95. The baseline has the smallest easy CVaR95 even though the hard specialist has a lower easy entropic point estimate. These are different criteria; the report does not collapse them into a single winner.
'''
    t+=page('Credit: comparison across metrics')
    t+=figure('credit_metrics','Sealed final credit results. Each row uses a fixed metric across the three domains. Dotted vertical lines mark the baseline. Axes differ by domain to make small method differences visible; read the labeled numbers rather than comparing visual widths.',height='.79')
    t+=r'''The easy domain carries more residual dispersion than the domain labelled hard. This is an outcome of this benchmark, not evidence that the intended easy/hard design succeeded.'''
    t+=page('Credit: paired gains and what is established')
    t+=figure('credit_gain_baseline','Entropic gains relative to baseline; lines are pointwise paired 95\% bootstrap intervals. Positive values favor the named method. The easy specialist is judged in easy and hard specialist in hard, not by a favorable cross-domain result.',height='.34')
    t+=figure('credit_gain_specialist','Adaptive gains relative to the own-domain specialist: easy specialist on the left, hard specialist on the right. Every interval crosses zero. Thus there is no clear adaptive advantage over these comparators.',height='.34')
    t+=r'''DPG mixed versus baseline on hard is positive on this conditional interval, but its interval relative to the hard specialist crosses zero. The original specialist separation gate failed, so this is an exploratory comparison rather than a confirmed successful benchmark. The multiple pointwise intervals do not provide a familywise significance claim.'''
    t+=page('Credit: development curves during training')
    t+=r'''All 24 learning-rate trials are displayed. Each point is \(\tfrac12(R_{\rm easy}+R_{\rm hard})\), measured on development episodes, in bp. This common plotting score allows a consistent view of progress; it is \emph{not} the original scaled checkpoint-selection criterion for every arm. Development curves reuse the selection set and are not test curves.
'''
    t+=figure('credit_learning','All arms and learning rates. Curves show checkpoints at 0, 60, 180, 360, 600, 1,200 and 1,800 active seconds. No smoothing or best-so-far replacement is applied.',height='.75')
    t+=page('Credit: daily error, inventory and defaults')
    t+=figure('credit_daily_holdings','Top: 21-day bins of RMS cross-episode daily SD (the original credit diagnostic), not pooled daily SD. Bottom: exported mean protection notional by vintage age. Four policies are shown to avoid eight nearly overlapping lines; full metrics for all eight appear in the tables.',height='.53')
    rr=[]
    for d in DOM:
        v=c['domains'][d];b=v['terminal']['baseline'];h=v['terminal']['hard_reference'];rr.append([d.capitalize(),f"{v['initial_price_mean']:.6f}",str(v['default_count']),f"{100*v['default_count']/8192:.2f}\\%",f"{b['entropic_ess']:.0f}",f"{h['entropic_ess']:.0f}"])
    t+=table(['Domain','Bond price','Defaults','Default rate','ESS base','ESS hard spec.'],rr)
    t+=r'''Entropic ESS is \(1/\sum_i w_i^2\), where \(w_i\propto e^{10L_i}\). It diagnoses concentration of the empirical score; it is not a training-seed sample size. Costs are charged per unit CDS notional traded, so 10 bp transaction cost does not mean a 10 bp total cost over five years. The baseline pays about 72 bp of discounted cumulative costs.

Credit daily ledgers agree with terminal losses to the exported numerical tolerance. Calendar zeros after default lower the fixed-calendar daily dispersion; this convention is common to all credit policies. Comparing this scalar across credit and options still requires care because the horizons, marks and absorbing structure differ.
'''
    t+=page('Options: dynamics and the IV--RV mechanism')
    t+=r'''Between arrivals, the two log-variance factors satisfy
\[
 dX_j=-\kappa_j^M X_j\,dt+0.35\,dB_j^M,
 \qquad v_t=\min\{0.04e^{X_{1,t}+X_{2,t}},1\},\quad M\in\{P,Q\}.
\]
Stock returns use beginning-of-day variance. Factors undergo exact daily OU transitions followed by a simultaneous \((\log2,\log2)\) jump with Bernoulli probability \(1/63\) per interval in base/hard, under both measures. Easy has no arrivals. Diffusions and shock support agree across P/Q within a domain; drifts differ. The stock Brownian motion is independent of the volatility factors. The explicit cap bounds conditional variance, but does not bound stock prices or terminal losses.
'''
    t+='\n\\begin{center}\n'+table(['Domain','P half-lives (days)','Q half-lives (days)','Arrivals'],[['Easy','3, 3','3, 3','None'],['Base','3, 3','40, 40','Probability 1/63'],['Hard','40, 3','1, 40','Probability 1/63']],spec='llll')+'\\end{center}\n'
    t+=figure('option_event','Stochastic-model conditional event diagnostic: no earlier arrival, one arrival at day 35, followed by ordinary random arrivals. Points are the exported checkpoints at 0, 5, 15, 30 and 60 days; lines only connect them. RV is backward-looking; IV is from a rolling 21-day ATM call and includes future Q arrivals.',height='.32')
    t+=r'''Unlike the initial imposed-event demo, this stochastic model prices future shocks. Base IV settles around 25\%, rather than returning to the 20\% easy level, because future pricing-measure shocks remain possible. Hard RV stays around 31--34\% at days 30--60 while IV is around 23\%. This is the desired separation, not a demonstration that a learned specialist must outperform.

For terminal replication, intermediate IV marks telescope out of final wealth. A fixed initial premium shifts entropic loss without changing hedge gradients. Consequently Q/P mismatch alone cannot establish a specialist advantage; the different physical persistence and causal information must create an economically meaningful change in the hedge.
'''
    t+=page('Options: specialist gates, separate from the diagnostic')
    t+=r'''A passing cost requires the lower paired 95\% gain bound to exceed 5 bp of initial spot \emph{on both targets}, and hard-specialist terminal SD above easy-specialist SD. Screen and confirmation sets are development evidence. The frozen final stream 520261017 stayed unopened. The later 820261017 diagnostic is a separate evidence set and did not select a passing cost.
'''
    t+=figure('option_gates','Original screen and confirmation gains. The dashed threshold applies to the lower confidence bound, not just the point estimate. Easy fails at every cost; hard-only improvement does not pass the joint gate.',height='.38')
    rr=[]
    for bp in [1,3,10]:
        g=gates[bp]['streams']['confirmation']
        for d in ['easy','hard']:
            v=g[d];lo,hi=np.array(v['ci95'])*1e4
            rr.append([str(bp),d.capitalize(),f"{v['gain']*1e4:.3f}",f'[{lo:.3f}, {hi:.3f}]','Yes' if g['checks'][d+'_useful_gain'] else 'No'])
    t+=table(['Cost (bp)','Target','Gain (bp)','Paired 95\% interval','Lower bound $>5$?'],rr)
    t+=r'''All three costs fail the joint condition. Adaptive option results are absent because the adaptive phase was skipped, not because those methods were trained and performed poorly. The harder target does have materially more residual risk than easy in this option setup, unlike the credit naming issue.'''
    for bp in [1,3,10]:
        t+=page(f'Options: full diagnostic results at {bp} bp cost')
        t+=r'''\textbf{Independent 820 diagnostic, not the sealed final evaluation.} Risks and costs are bp of initial spot. Terminal metrics use 8,192 episodes/domain; daily SD uses 256 independent episodes and Monte Carlo option marks. The fixed ordering matches the credit tables.
'''
        for d in ['easy','base','hard']:
            t+='\n\\subsection*{'+d.capitalize()+'}\n';rr=[]
            for a in OLABEL:
                r=lookup['option',bp,d,a];rr.append([OLABEL[a]]+[f"{r[k]:.3f}" for k in ['entropic_bp','cvar95_bp','terminal_sd_bp','daily_pooled_sd_bp','mean_cost_bp']])
            t+=table(['Method','Entropic','CVaR95','Terminal SD','Daily SD','Cost'],rr)
        t+=r'''The tuned constant holding is selected separately on own-domain development episodes. BS delta uses a fixed 20\% volatility and causal stock/time inputs; it receives no realized-volatility oracle. Reported scores are empirical finite-sample scores, subject to the population-tail limitation in Section~\ref{sec:limitations}.'''
    t+=page('Options: consistent comparisons across metrics')
    t+=figure('option_metrics','Signed change divided by absolute baseline risk within each domain/cost; below zero is better. The tuned constant is omitted here to keep the scale readable, but is retained in every absolute-value table. These are point-estimate comparisons, not simultaneous confidence bands.',height='.76')
    t+=r'''An improvement in entropic risk need not produce an improvement in CVaR or daily SD. The table values, not a chosen favorable subplot, define the full result.'''
    t+=page('Options: development curves during training')
    t+=r'''Each panel is the model's own-domain development entropic risk. The fixed initial Q premium is subtracted from the source training/development objective so the vertical scale matches terminal-loss reporting. This is an exact translation, not a refit. Panels use different vertical scales and do not measure the same training distribution.
'''
    t+=figure('option_learning','All 27 cost/arm/learning-rate trials. Points are exported development evaluations during training; they are not minibatch losses, final-test measurements or independent seed replicates. Curves begin at 60 seconds because retained histories begin there. Where a resumed trial lacks its checkpoint history, nominal checkpoint times and raw development loss arrays are used.',height='.72')
    t+=page('Options: learning a hedge versus gaining from specialization')
    t+=figure('option_static','Independent diagnostic: learned policies substantially improve empirical entropic risk relative to development-tuned constant holdings. A weak specialization gap does not imply failure to learn a useful hedge.',height='.37')
    t+=r'''The easy market is close to a stable-volatility problem. A baseline trained on fast shocks can transfer an effective stock hedge to it, leaving little room for an easy specialist. The hard market has persistent variance and a larger benefit from a specialist, but one successful target does not satisfy the planned two-target comparison.

\textbf{Monte Carlo daily-mark noise.} Dealer marks are evaluation-only; quote sampling noise enters the daily diagnostic. The largest exported RMS difference between two independent marks (among dates 0, 21, 63 and 105) is:
'''
    rr=[]
    for d in ['easy','base','hard']:
        v=max(o['daily_quote_error'][d],key=lambda x:x['rms_independent_price_difference']);rr.append([d.capitalize(),str(v['day']),f"{1e4*v['rms_independent_price_difference']:.3f}"])
    t+='\n'+table(['Domain','Date of maximum','Independent quote difference (bp)'],rr)
    t+=r'''This is an RMS difference of prices, not an error bar for daily hedge SD. It shows that small daily-metric differences in base/hard may be sensitive to quote precision. Terminal option P\&L uses the realized payoff and a fixed initial premium, so it does not inherit repeated daily quote noise in the same way. Shared marks across methods reduce some comparison noise but do not establish the precision of their absolute daily SDs.'''
    t+=page('Interpretation, limitations and next steps')+r'\label{sec:limitations}'+'\n'
    t+=r'''\subsection*{What the evidence supports}
Credit: no adaptive method clearly improves on the own-domain specialists. Some baseline-relative point estimates improve, but benchmark discovery itself failed and the final own-specialist gaps remain small or inconclusive. The easy/hard labels do not describe an accepted credit difficulty ordering.

Options: the desired IV/RV event pattern is present and hard-specialist gains are meaningful on conditional episode intervals. Easy-specialist gains are too small to satisfy the joint gate, so the adaptive comparison was correctly not entered.

\subsection*{Population entropic risk is not certified for options}
For positive proportional liquidation cost \(c\), condition on the history before the final stock return and let the causal holding be \(h\in[0,1]\). On the large-stock-price tail, short-call loss has coefficient
\[
 1-h+ch\geq c>0
\]
on terminal stock price. The remaining return is lognormal and has no finite positive exponential moment. Therefore the frozen option model's untruncated \emph{population} entropic risk is infinite, despite finite empirical log-mean-exp scores. A variance cap does not remove this stock tail. Conditional bootstrap intervals describe the recorded samples, not a population guarantee. CVaR, cost and SD remain different, finite-moment criteria. No outcome-driven truncation or objective change was applied here.

\subsection*{Other boundaries}
One initialization cannot measure training-seed variability. LR/checkpoint tuning uses development data; intervals are conditional on those selected policies. Credit pricing is coherent with its finite-state law, but an earlier refinement-pass flag does not establish convergence of every production grid to a continuous model. Hidden credit factors can be revealed by quotes. Option daily SD uses only 256 paths and noisy marks. Horizons, initial funding, costs and normalization differ across markets; even harmonized units do not make a cross-market performance contest valid.

\subsection*{A useful next experiment}
First settle a well-defined finite objective for the intended option task (for example CVaR, or an explicitly justified bounded-tail/contract model if entropic population risk is essential). Then use independent development episodes to establish a real, useful specialist gap on both targets before freezing a new adaptive comparison. Report stock-history inference limits and variation in optimal actions, not just IV/RV pictures. Preserve a pure-base and, if allocation is the question, a fixed common-support mixture comparator so adaptive scenario coverage is not mistaken for superior allocation.
'''
    # Complete comparable metric appendix with mean and original daily statistic.
    t+=page('Credit: mean loss and original daily diagnostic')
    rr=[]
    for d in DOM:
        for a in ARMS:
            r=lookup['credit',10,d,a];dd=c['domains'][d]['daily_marked_errors'][a]
            rr.append([d.capitalize(),LABEL[a],f"{r['mean_loss_bp']:.3f}",f"{dd['rms_cross_episode_sd']*1e4:.3f}",f"{r['daily_pooled_sd_bp']:.3f}"])
    t+=table(['Domain','Method','Mean loss (bp)','Daily RMS SD (bp)','Daily pooled SD (bp)'],rr,size='footnotesize')
    t+=r'''The two daily measures are very close in this credit sample, but they are not identical by definition. Both are retained to prevent an unnoticed change of metric. The machine-readable tables contain mean losses and costs for all option rows as well.'''
    t+=page('Options: mean losses and pricing offsets')
    t+=r'''All terminal option scores include a fixed, domain-specific initial Q premium. This premium is 564.004 bp in easy, 965.992 bp in base and 729.322 bp in hard. The base family physically normalizes much faster than its pricing measure anticipates. Large negative base losses therefore include a pricing-premium effect; they do not mean that base has zero residual hedging risk.

The following 10 bp-cost diagnostic table separates mean loss from entropic excess risk, defined as \(R_{10}(L)-\overline L=R_{10}(L-\overline L)\). This is a supplementary diagnostic, not a replacement for the common primary metrics or a new selection criterion. All values are bp of initial spot.
'''
    rr=[]
    for d in ['easy','base','hard']:
        for a in OLABEL:
            r=lookup['option',10,d,a]
            rr.append([d.capitalize(),OLABEL[a],f"{r['mean_loss_bp']:.3f}",f"{r['entropic_bp']-r['mean_loss_bp']:.3f}",f"{r['entropic_bp']:.3f}"])
    t+=table(['Domain','Method','Mean loss','Centered entropic','Entropic'],rr)
    t+=r'''Fixed initial premiums shift entropic risk, mean and CVaR by the same constant within a domain. They do not change within-domain method gains or terminal SD. Centering helps explain cross-domain levels, but centered risk ignores economic mean loss and costs already paid. Keep the main tables as the economic result and use this decomposition to interpret them.

Daily marked errors use changing option values, with MC-estimated initial marks on the daily subset. Terminal arrays use the fixed high-precision initial premium. Their sum/level conventions were audited separately; the 256-path daily metric is not another estimate from the same 8,192-path terminal sample.
'''
    t+=page('Training budgets and selection: credit')
    rr=[]
    for b in budgets:
        rr.append([LABEL[b['arm']],f"{b['lr']:g}",str(b['learner_updates']),f"{b['training_paths']/1e6:.3f}",str(b['teacher_feedback_paths']),f"{b['active_seconds']:.1f}",f"{b['compile_seconds']:.1f}"])
    t+=table(['Arm','LR','Updates','Paths (M)','Feedback paths','Active s','Compile s'],rr,size='footnotesize')
    t+=r'''There are 24 trials, roughly 12 hours of total active learner/teacher work, plus compilation, development, monitoring, export and recovery overhead. DPG performs fewer updates because its time budget includes differentiation through learner blocks. Equal active seconds are the budgeted comparison; the table makes the difference in data exposure explicit. Self-play retained its original run during recovery and charged known lost active work against the original budget.'''
    t+=page('Checkpoint choices and adaptive-method definitions')
    rr=[]
    for a in ARMS:
        cand=[v for v in selection if v['arm']==a];v=min(cand,key=lambda v:v['selection_score']);rr.append([LABEL[a],f"{v['lr']:g}",v['checkpoint'],tex_escape(v['criterion']).replace('fixed balanced scaled easy/hard development','Scaled easy/hard')])
    t+=table(['Credit arm','Selected LR','Selected checkpoint','Criterion'],rr,size='footnotesize')
    t+='\n\\subsection*{Option selections}\n';rr=[]
    for bp in [1,3,10]:
        for a in ['baseline','easy','hard']:
            v=gates[bp]['selected_sources'][a]['selected'];rr.append([str(bp),OLABEL[a],f"{v['lr']:g}",f"{v['best_time']:.1f}",str(v['steps'])])
    t+=table(['Cost (bp)','Option arm','Selected LR','Checkpoint active s','Trial updates'],rr,size='footnotesize')
    t+=r'''\subsection*{Adaptive methods in the credit experiment}
DPG differentiates held-out target risk through each complete learner block, including carried Adam moments, to assign sample-weight rewards. SOAR uses held-out target-risk improvement before/after the actual block and a moving reward baseline. Self-play uses historical parameter movement, Adam preconditioning and directional per-path entropic-gradient contributions; it does not receive the target feedback batches used by DPG/SOAR.

The generator samples the fixed easy/ordinary/hard menu with a uniform reference prior, 5\% exploration floor and KL coefficient 0.01. Hard-target teachers use hard feedback; mixed-target teachers use fixed scaled easy/hard risk with frozen SD scales 0.0203941 and 0.0152773. Adaptive models can see all families while baseline sees ordinary only, so coverage and allocation are confounded. All learners use the same credit architecture and differentiable Adam convention (\(\epsilon_{\rm root}=10^{-16}\)).
'''
    t+=page('Provenance, validation and reproducibility')
    t+=r'''\textbf{Evidence snapshot.} SHA-verified primary download: \path{hetzner-results-20261019}.

Credit terminal evaluation run: \path{20261019T021407Z-6f5fb002}.

Option diagnostic run: \path{20261019T042258Z-b20a116c}.

Original screen/confirmation evidence: the three \path{gate-*bp-results.json} files. Training/development histories and selection records are additional read-back SHA-verified files under this report's \path{supplement/} directory.

\textbf{Report-side checks.} Every terminal entropic score and mean was independently recomputed from exported episode arrays and checked against source JSON. Credit fractional CVaR and the original option rounded-tail CVaR were verified; option CVaR was then recomputed with fractional boundary mass for the main tables. Terminal SDs were harmonized to population normalization. Every option development-history score was checked against its retained 2,048-path loss array before subtracting the initial premium. Actual evaluation times are matched to the nearest nominal checkpoint time when locating raw loss arrays. Credit daily pooled SDs were reconstructed from exported daily sample moments with the correct degrees-of-freedom conversion, not approximated by their RMS statistic. No training, fitting, new selection or GPU rental was performed.

\textbf{Artifacts.} \texttt{report.pdf} is the compiled document; \texttt{report.tex} the full source. \texttt{figures/} contains vector PDFs and preview PNGs. \texttt{tables/metrics.csv} contains the 69 harmonized metric rows. \texttt{build\_report.py} regenerates tables, figures and LaTeX. \texttt{input-manifest.json} hashes each evidence input consumed by this builder; the original remote manifests remain unchanged.

\textbf{Rebuild locally.}
\begin{verbatim}
MPLCONFIGDIR=/tmp/mpl-hedging-report \
    python hedging-report/build_report.py
cd hedging-report
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
\end{verbatim}
The builder requires NumPy and Matplotlib; compiling requires a standard LaTeX installation. Evidence paths are relative to this project. Bundle the original data folder and report folder together for a fully portable rebuild.

\textbf{Execution record.} The final source reports all campaign GPU instances shut down. USD35.10 is the deduplicated provider total across the original campaigns and preceding work, not solely the most recent top-up or the cost of one comparison. A separate retained storage volume holds full self-play histories. Export recovery completed the same credit final evaluation; it did not rerun the scientific test. Financial and recovery details are retained in \texttt{FINAL\_COST\_AND\_SHUTDOWN.json} and \texttt{RECOVERY\_AND\_SCOPE.md} in the evidence snapshot.

\textbf{What is not claimed.} No confirmed two-target specialist benchmark, adaptive superiority, seed robustness, optimality, continuous-model credit convergence or finite option population entropic risk follows from these results.
\end{document}
'''
    (HERE/'report.tex').write_text(t)
    manifest={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(USED)}
    (HERE/'input-manifest.json').write_text(json.dumps(manifest,indent=2))
    (HERE/'validation.json').write_text(json.dumps({'status':'passed','harmonized_metric_rows':len(rows),'input_files':len(USED),'checks':['terminal entropic/mean and original CVaR reconstruction; fractional CVaR harmonization','population terminal SD normalization','option dev risk from raw losses','credit pooled daily SD from moments']},indent=2))
    print('Built report.tex and figures;',len(rows),'metric rows;',len(USED),'hashed inputs')


if __name__=='__main__':
    main()
