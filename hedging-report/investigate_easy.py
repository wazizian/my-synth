"""Compare baseline/easy specialist development trajectories and final credit loss components."""
from pathlib import Path
import json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-easy-investigation')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    """Read all six development trials, plot full/late curves, and export final decomposition."""
    root=Path(__file__).resolve().parent
    source=root/'supplement/comparison-20261018/credit-final'
    fig,axs=plt.subplots(2,3,figsize=(12,7),sharex='row')
    selections={'baseline':.001,'easy_reference':.0003}
    colors={'baseline':'#2468b4','easy_reference':'#c65a12'}
    styles={.0001:':',.0003:'--',.001:'-'}
    for arm in selections:
        for lr in styles:
            folder=next((source/arm).glob(f'launcher/runs/*/outputs/training/lr-{lr:g}'))
            points=[]
            for sec in [0,60,180,360,600,1200,1800]:
                value=json.loads((folder/f'selection-{sec}.json').read_text())
                points.append([sec]+[value[d]*1e4 for d in ['easy','ordinary','hard']])
            points=np.array(points)
            for j,d in enumerate(['Easy evaluation','Ordinary evaluation','Hard evaluation']):
                for i in range(2):
                    ax=axs[i,j];selected=lr==selections[arm]
                    ax.plot(points[:,0],points[:,j+1],styles[lr],color=colors[arm],lw=2.4 if selected else 1.1,alpha=1 if selected else .4,marker='o',ms=3)
                    ax.set_title(d if i==0 else d+' — late training',fontsize=10)
                    ax.set_ylabel('Development entropic risk (bp)');ax.set_xlabel('Active training seconds');ax.grid(alpha=.2)
                    if i==1:ax.set_xlim(600,1820);ax.set_ylim({'easy':(97,104),'ordinary':(77,86),'hard':(84,94)}[['easy','ordinary','hard'][j]])
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color=colors[a],lw=2,label='Baseline (ordinary training)' if a=='baseline' else 'Easy specialist') for a in colors]
    handles += [Line2D([],[],color='gray',ls=styles[lr],label=f'LR {lr:g}') for lr in styles]
    fig.legend(handles=handles,loc='lower center',ncol=5,frameon=False)
    fig.suptitle('Credit: baseline versus easy specialist — selected trials bold',fontsize=14)
    fig.tight_layout(rect=(0,.06,1,.95))
    fig.savefig(root/'figures/easy_investigation.pdf',bbox_inches='tight')
    fig.savefig(root/'figures/easy_investigation.png',bbox_inches='tight',dpi=170)
    results=json.loads((root.parent/'hetzner-results-20261019/credit-evaluation/results.json').read_text())
    comparison={}
    for domain in ['easy','ordinary','hard']:
        b=results['domains'][domain]['terminal']['baseline'];e=results['domains'][domain]['terminal']['easy_reference']
        comparison[domain]={k:dict(baseline=b[k]*1e4,easy_specialist=e[k]*1e4,deterioration=(e[k]-b[k])*1e4) for k in ['entropic','mean','centered_entropic','terminal_std','mean_cost','cvar95']}
    (root/'tables/easy_investigation.json').write_text(json.dumps(comparison,indent=2))
    print(json.dumps(comparison,indent=2))


if __name__=='__main__':
    main()
