# Restored distinct credit benchmark

`--market distinct --problem credit` restores the historical **two-factor grid**
model and learned target/retention policy. It supersedes the one-factor
continuous analogue as the default.

The original production settings are versioned in
`credit_hedging/configs/distinct_regimes.json`, copied from the previous
ordinary-long campaign's `numerical_summary.json` production configurations:

| Regime | Slow kappa | Slow volatility | Slow nodes | Fast kappa | Fast volatility | Fast nodes |
|---|---:|---:|---:|---:|---:|---:|
| Easy | .25 | .3535533905932738 | 1201 | 6 | 0 | 1 |
| Base/ordinary | 1 | .7071067811865476 | 601 | 6 | 0 | 1 |
| Hard | .25 | .25 | 1201 | 6 | 1.224744871391589 | 801 |

Slow grid [-6.5,-.5], mean/initial -3.5; fast grid [-2,2], mean/initial 0.
Singleton fast factors stay at their mean. Hazard is softplus(slow+fast).
Independent Gaussian OU cell probabilities define the finite-state law.
The same matrices price and simulate the market. Daily default is drawn using
current hazard; factor transitions are applied only on survival. Absorption
freezes the grid state. No interpolated neural-Q marks replace these prices.

Contracts retain 1,260-day ZCB and 126-day CDS, discount rate .02, recovery .4,
1 bp trading cost and total protection cap 2. The vintage ledger retains fixed
coupons, allows only sales of old protection and purchase of today's new CDS,
accounts for default/end-point cashflows, and masks empty-vintage marks.

The original float64 GRU32 uses spread/spread change/time/total protection plus
vintage inventory, coupons and held marks. Its zero head initializes target
protection 1, full retention, and replenishment to target. All arms share that
architecture and initialization.

Baseline draws fresh ordinary paths; specialists draw fresh own-domain paths.
DPG/SOAR keep the fixed hard feedback pool with resampled subsets. Their neural
proposal learns mean/volatility corrections on both hard-grid factors, with zero
corrections reproducing the hard reference. Exact stopped-path scores include
factor cell probabilities and Cox default/survival. Synthetic paths use the
fixed hard-market prices; base/easy target paths use their own tables.

The grid proposal/default sampler is discrete. Pathwise meta-learning is
rejected for this restored credit benchmark; the old differentiable one-factor
experiment is available only with explicit `--market neural-demo` and is
separately documented in [CREDIT_CONTINUOUS_EXPERIMENT.md](CREDIT_CONTINUOUS_EXPERIMENT.md).
It is not claimed as a two-factor implementation.

Tables are explicit dynamic inputs to the top-level compiled learner/teacher
workflows. The hard bond table alone is about 10 GB in float64; temporary
pricing buffers and training require additional memory. Production resource
usage has not been measured for the new runner. Smaller CLI grid overrides
are smoke approximations. New tests compare tables, sampling and policy
initialization to archived code; they have not been run.
