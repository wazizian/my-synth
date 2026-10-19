"""Assemble the campaign assessment from already computed worker evidence."""
from pathlib import Path
import json
import shutil

root=Path(__file__).resolve().parent
run=Path('/home/jump/vast-tools/runs/20261018T135331Z-dd6fa036')
out=run/'outputs'
metrics=json.loads((out/'economic/all_metrics.json').read_text())
gate=json.loads((out/'economic/gate.json').read_text())
numerical=json.loads((out/'numerical/numerical_summary.json').read_text())
billing=json.loads((root/'provider_billing.latest.json').read_text())
assert gate['stop_uninformative'] and gate['common_near_best_policies']==['replenish_1.0']
status=('All campaign instances are confirmed shut down.' if not billing['active_campaign_instances']
        else 'Export verification and provider shutdown are still pending.')
rows=[]
for domain in ('easy','ordinary','hard','base'):
    m=metrics[domain]['replenish_1.0']
    rows.append(f"| {domain} | {m['entropic']:.6f} | {m['cvar95']:.6f} | {m['daily_error_std']:.6f} | {m['mean_cost']:.6f} | {m['initial_bond_price']:.6f} |")
table='\n'.join(rows)
report=f'''# Credit-curve campaign assessment

The frozen economic pre-screen stopped this bounded candidate before learned-specialist or adaptive training. Keeping one unit of protection and replenishing it only at expiry was the best of the six tested rules on both easy and hard scenarios. The candidate did not exhibit the intended target-dependent preferred strategy. This is an uninformative benchmark for the proposed adaptive comparison, not proof that this rule is optimal or that adaptive training can never help.

The common rule had empirical regret zero on both targets and paired 95% bootstrap regret intervals [0, 0] in both domains. It remained best within this six-rule menu in all 1,000 resamples. These are menu-relative development intervals, not a population zero-error guarantee. The frozen stopping threshold was an upper interval endpoint below 0.001 per unit bond face on both targets.

The run evaluated 8,192 independent episodes per component over all 1,260 bond intervals with 126 CDS vintage slots. All six rules used the same paths within each component. Base P0 is the exact empirical mixture 0.05 easy + 0.90 ordinary + 0.05 hard; ordinary alone is not the baseline distribution. The weighted base summary reuses these component episodes, rather than adding independent observations.

## Common replenishment rule

All monetary quantities are per unit bond face; loss is the financed long bond's initial price minus discounted net bond/CDS cashflows. Entropic risk uses gamma 10. CVaR uses the upper 5% tail with fractional boundary mass.

| Domain | Entropic loss | CVaR95 | Daily error SD | Mean discounted cost | Initial bond price |
|---|---:|---:|---:|---:|---:|
{table}

Daily replacement incurred roughly 0.22 face units of discounted costs, versus about 0.00089 for expiry replenishment. It could have a slightly lower daily dispersion while much worse terminal economic loss. The daily statistic therefore cannot substitute for terminal risk and costs. Turnover in the worker metrics counts opening purchases/sales, excluding final liquidation notional; the cost metric includes final liquidation costs.

Under the common best diagnostic rule, the candidate called hard had lower terminal SD, centered entropic risk, centered CVaR and daily dispersion than easy. The names were hypotheses and did not establish an accepted residual-difficulty ordering. Learned specialists were not trained, so no learned-specialist ordering or specialization gap is claimed.

For this rule, the easy/ordinary/hard samples contained respectively {metrics['easy']['replenish_1.0']['default_count']}, {metrics['ordinary']['replenish_1.0']['default_count']} and {metrics['hard']['replenish_1.0']['default_count']} defaults out of 8,192 each. Raw default-day and survival-path outcomes, mean loss, centered risks, costs, positions and entropic effective sample sizes are retained separately.

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

Provider-reported campaign cost: ${billing['total_reported_usd']:.3f}, including failed attempts and recovery. Provider rounding and later settlement may change this total. {status}

## Artifacts

- Full worker report and all-policy tables: `{out/'economic/report.md'}`.
- Canonical raw outputs, source hashes, test results and numerical checks: `{out}`.
- Gate and independent raw-record audit: `economic/gate.json` and `economic/independent_audit.json` within those outputs.
- Final metadata/integrity audit, preservation review, recovery records and billing: this campaign directory.
- LaTeX assessment: `report.tex`; figures: `figures/`. The LaTeX source has been reviewed; no compiled PDF is claimed.

![Easy-domain risk comparison](figures/easy-risk.png)

![Hard-domain risk comparison](figures/hard-risk.png)

![Hard-domain daily means and dispersion](figures/hard-daily.png)
'''
(root/'report.md').write_text(report)
shutil.copytree(out/'economic/figures',root/'figures',dirs_exist_ok=True)
worker_tex=(out/'economic/report.tex').read_text()
worker_tex=worker_tex.replace('\\begin{longtable}', '\\scriptsize\n\\begin{longtable}').replace('\\end{longtable}', '\\end{longtable}\n\\normalsize')
supplement=r'''
\section{Validation and execution scope}
All 39 GPU tests and all 18 exported-record audit cells passed. Production grids
were 1201 by 1, 601 by 1, and 1201 by 801 for easy, ordinary and hard. The frozen
pricing tolerance was 0.00005 per unit face; doubled-grid and widened-boundary
checks passed on the declared probe lattice. This is not a uniform or
continuous-time error certificate. Accounting retained its 1e-10 tolerance.

The memory repair replaced duplicate maturity tensors with one preallocated
float64 buffer. The failed run was preserved, and the recovery replay used the
same seed and streams; it supplies no additional independent seed evidence.
Local orchestration ended after the final computation/report, requiring recovery
of exports and shutdown. The accompanying Markdown assessment and recovery
records state the retained-artifact scope in full.

The common replenishment rule had regret intervals [0,0] relative to the six-rule
menu in both targets. This triggered the predeclared economic stop, not a claim
of global optimality. Hard had lower residual diagnostics under this rule than
easy. No learned-specialist, adaptive, confirmation or final-test stage was opened.

Opening-trade turnover excludes final liquidation notional; reported costs
include final liquidation. Daily variance uses ddof zero and absorbing zeros
after default. Full daily trade/cash records and paths are retained; full vintage
traces cover eight representative episodes, with mean surfaces over all episodes.
There are no scientific neural checkpoints because training was not entered.
After export verification, the user requested checkpoint cleanup; twenty small
preflight checkpoints were removed with their hashes retained in the cleanup
audit. Historical export manifests describe the state before that deletion.

'''+f"Provider-reported cost: \\${billing['total_reported_usd']:.3f}. "+status+r'''

\clearpage
\includegraphics[width=\linewidth]{figures/easy-risk.png}
\includegraphics[width=\linewidth]{figures/hard-risk.png}
\clearpage
\includegraphics[width=\linewidth]{figures/hard-daily.png}
'''
(root/'report.tex').write_text(worker_tex.replace('\\end{document}',supplement+'\n\\end{document}'))
print('Campaign Markdown and LaTeX assessments written; provider status:',status)
