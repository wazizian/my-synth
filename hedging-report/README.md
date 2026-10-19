# Unified credit and option report

Open `report.pdf` (22 pages). `report.tex` is the complete LaTeX source. Vector plots are in `figures/`; `tables/metrics.csv` contains 69 consistently defined metric rows. `validation.json` records successful raw-data checks; `input-manifest.json` hashes all 383 consumed inputs.

The report separates sealed credit results, option specialist development gates, and the independent option diagnostic. CVaR boundary mass, terminal-SD normalization, and daily-SD definitions are harmonized explicitly. All monetary scores use basis points; market units differ and absolute levels are not directly comparable.

Rebuild from the parent folder:

```bash
MPLCONFIGDIR=/tmp/mpl-hedging-report python hedging-report/build_report.py
cd hedging-report
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
```

The portable ZIP contains this directory plus the original selected `hetzner-results-20261019` evidence snapshot. No training or new market simulation is needed to rebuild. Requires NumPy, Matplotlib and LaTeX. `supplement-sha256.json` records 587 additional read-back-verified remote files, including all-LR development evidence.
