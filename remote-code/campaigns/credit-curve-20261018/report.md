# Credit-curve campaign assessment

The frozen economic pre-screen stopped this bounded candidate before learned-specialist or adaptive training. Keeping one unit of protection and replenishing it only at expiry was the best of the six tested rules on both easy and hard scenarios. The candidate did not exhibit the intended target-dependent preferred strategy. This is an uninformative benchmark for the proposed adaptive comparison, not proof that this rule is optimal or that adaptive training can never help.

The common rule had empirical regret zero on both targets and paired 95% bootstrap regret intervals [0, 0] in both domains. It remained best within this six-rule menu in all 1,000 resamples. These are menu-relative development intervals, not a population zero-error guarantee. The frozen stopping threshold was an upper interval endpoint below 0.001 per unit bond face on both targets.

The run evaluated 8,192 independent episodes per component over all 1,260 bond intervals with 126 CDS vintage slots. All six rules used the same paths within each component. Base P0 is the exact empirical mixture 0.05 easy + 0.90 ordinary + 0.05 hard; ordinary alone is not the baseline distribution. The weighted base summary reuses these component episodes, rather than adding independent observations.

## Common replenishment rule

All monetary quantities are per unit bond face; loss is the financed long bond's initial price minus discounted net bond/CDS cashflows. Entropic risk uses gamma 10. CVaR uses the upper 5% tail with fractional boundary mass.

| Domain | Entropic loss | CVaR95 | Daily error SD | Mean discounted cost | Initial bond price |
|---|---:|---:|---:|---:|---:|
| easy | 0.009209 | 0.074680 | 0.001259 | 0.000893 | 0.827687 |
| ordinary | 0.008178 | 0.059780 | 0.001231 | 0.000891 | 0.825324 |
| hard | 0.007869 | 0.056445 | 0.001212 | 0.000890 | 0.826012 |
| base | 0.008215 | 0.060481 | 0.001231 | 0.000891 | 0.825477 |

Daily replacement incurred roughly 0.22 face units of discounted costs, versus about 0.00089 for expiry replenishment. It could have a slightly lower daily dispersion while much worse terminal economic loss. The daily statistic therefore cannot substitute for terminal risk and costs. Turnover in the worker metrics counts opening purchases/sales, excluding final liquidation notional; the cost metric includes final liquidation costs.

Under the common best diagnostic rule, the candidate called hard had lower terminal SD, centered entropic risk, centered CVaR and daily dispersion than easy. The names were hypotheses and did not establish an accepted residual-difficulty ordering. Learned specialists were not trained, so no learned-specialist ordering or specialization gap is claimed.

For this rule, the easy/ordinary/hard samples contained respectively 1201, 1231 and 1248 defaults out of 8,192 each. Raw default-day and survival-path outcomes, mean loss, centered risks, costs, positions and entropic effective sample sizes are retained separately.

## Numerical validation and information

The model is a daily finite-state approximation to independent slow/fast OU factors transformed through softplus intensity. Pricing and simulation use the same transition law, deterministic 2% rates, recovery 0.4, end-of-day default, and custom daily-coupon par CDS. Old coupons do not reset; default-day premiums, expiry, sales and final liquidation share the audited ledger. These are not standard ISDA CDS conventions.

An informed dealer publishes executable new and held-vintage quotes. The policy observes these quotes, time and its inventory, not simulator factor labels or bond marks. Quotes may reveal factor information; no observational-unidentifiability claim is made. The observable sensitivity rule uses the same ordinary-model quote-implied estimate in every domain.

All 39 GPU tests passed, including independent dated-contract cash accounting, the two-factor mixed-coupon martingale check, production/reference pricing equivalence, finite-difference gradients, causality and checkpoint replay. The independent exported-record audit passed all 18 component/policy cells, each with 8,192 episodes, at the unchanged 1e-10 accounting tolerance.

The initial coarse grids failed the frozen 0.00005 pricing tolerance and were refined before economic outcomes. Production grids are easy 1,201 by 1, ordinary 601 by 1, and hard 1,201 by 801, checked against doubled reference resolution. Passing maximum bond-price differences were approximately 0.00001925, 0.00002523 and 0.00002123. Boundary widening also passed. These checks cover the declared interior probe lattice over all relevant maturities; they are not uniform latent-state bounds or a certification of continuous-time accuracy. Transition-moment discrepancies and all rejected resolutions remain in the numerical records.

The daily dispersion uses population cross-episode variance (ddof 0), then RMS over the fixed calendar, including absorbing zero increments after default. All uncertainty here concerns evaluation episodes for fixed diagnostic rules. There is no training-seed uncertainty, learned-policy superiority or optimality claim.

## Execution, preservation and scope

The original uploaded source snapshot and 76 closure-manifest files from the prior SLV and trinomial campaigns were verified unchanged. Prior model checkpoints had already been deleted at the user's explicit request; this report does not claim they remain available. A subsequent explicit cleanup removed 20 preflight checkpoints (115,280 bytes) across this campaign's five worker runs after export verification. Their hashes and deletion authorization are recorded in `/home/jump/campaign-control/checkpoint-cleanup-20261018-followup.json`. Historical export manifests describe the pre-deletion state; the final integrity audit accounts for these exact authorized removals.

One test helper was updated for the worker's newer JAX API without changing its assertion. The first economic attempt completed easy and ordinary, then exhausted GPU memory constructing the hard table. A single preallocated maturity buffer removed the duplicate allocation while preserving the validated grid and float64 recurrence. The full hard grid then passed validation. The successful recovery replay used the same seed and streams; it is not an additional seed replicate. A checked sensitivity-rule batch differed only at round-off level (maximum terminal/daily difference 4.44e-16).

The failed run's 996-file manifest was fully verified before shutdown. Its additional repair validation summary, complete stdout and submitted source were separately preserved; its temporary remote JUnit/cache tree was not in that frozen failure manifest and is not claimed retained. The complete retry retains its own full JUnit evidence.

After the retry's calculations, audit and report completed, local orchestration sessions ended with code 143 while the worker waited for export acknowledgement. Export verification and shutdown were recovered without rerunning the scientific computation. After acknowledgement, a final stdout print raised BrokenPipeError after the completion marker had been written; the marker matches the gate, and the transport-error trace is retained. See the recovery records for exact scope and integrity evidence.

Full daily cash/trade/position records and scenario paths are retained. Full vintage traces cover eight representative episodes per component/rule, with mean vintage surfaces for all episodes; deterministic rules and retained paths permit reconstruction of other vintage histories. No scientific neural model/optimizer checkpoint or training-loss curve exists because the learned branch was not entered. Checkpoint-resume validation pertains to the infrastructure tests.

No confirmation or final-test data was opened. The plan's later learned-specialist, eight-arm, tuning and final-test stages were not entered under the predeclared economic stopping rule. No candidate expansion, threshold adjustment or favorable-outcome selection occurred.

Provider-reported campaign cost: $1.344, including failed attempts and recovery. Provider rounding and later settlement may change this total. All campaign instances are confirmed shut down.

## Artifacts

- Full worker report and all-policy tables: `/home/jump/vast-tools/runs/20261018T135331Z-dd6fa036/outputs/economic/report.md`.
- Canonical raw outputs, source hashes, test results and numerical checks: `/home/jump/vast-tools/runs/20261018T135331Z-dd6fa036/outputs`.
- Gate and independent raw-record audit: `economic/gate.json` and `economic/independent_audit.json` within those outputs.
- Final metadata/integrity audit, preservation review, recovery records and billing: this campaign directory.
- LaTeX assessment: `report.tex`; figures: `figures/`. The LaTeX source has been reviewed; no compiled PDF is claimed.

![Easy-domain risk comparison](figures/easy-risk.png)

![Hard-domain risk comparison](figures/hard-risk.png)

![Hard-domain daily means and dispersion](figures/hard-daily.png)
