# BS/SLV campaign final assessment

Neither of the two frozen benchmark families passed the full development gate. The plan requires stopping before adaptive training and final-test evaluation. This is an uninformative benchmark for the proposed adaptive comparison.

One shared training seed: 20261017. Evaluation intervals condition on those trained policies; they do not measure training-seed uncertainty.

Provider-reported total cost: $0.838; campaign elapsed time: 89.4 minutes. All campaign instances are confirmed shut down. Provider charges are rounded and may settle later.

## Development results

| Family | Target | Baseline minus specialist entropic risk | 95% interval | Required lower bound |
|---|---|---:|---|---:|
| A | easy | 0.00001605 | [-0.00002448, 0.00005525] | 0.00020 |
| A | hard | 0.00003079 | [-0.00001601, 0.00007696] | 0.00050 |
| B | easy | 0.00003590 | [0.00000003, 0.00007177] | 0.00020 |
| B | hard | 0.00002600 | [-0.00002789, 0.00007407] | 0.00050 |

Primary gates and the supplementary marked-residual review are retained separately. The latter checks X minus V0 so differing initial option values cannot establish difficulty by themselves. Its conservative numerical margins were declared before scientific training.

### Family A

Overall screen passed: False. Fresh confirmation passed: not opened.

screen: primary failures: easy_specialization, hard_specialization; material marked-residual contradiction: False.

Initial-value SD: easy 0.00000000, hard 0.00388609. Marked terminal SD: easy 0.00843257, hard 0.01028074.

### Family B

Overall screen passed: False. Fresh confirmation passed: not opened.

screen: primary failures: easy_specialization, hard_specialization; material marked-residual contradiction: False.

Initial-value SD: easy 0.00006860, hard 0.00500575. Marked terminal SD: easy 0.00872530, hard 0.01099380.

## Numerical evidence and limits

All final-run numerical, independent pricing/filter, and objective-gradient checks passed before the first diagnostic arm trained. Independent reconstruction of exported accounting and metrics passed at the fixed1e-10 accounting tolerance. See independent_audit.json for exact coverage and maxima.

The initial coarse price grid failed its frozen tolerance. Production resolution was increased to192 variance nodes,1601 log-spot nodes and64 Gauss-Hermite nodes; reference resolution doubles each dimension. Market parameters and usefulness thresholds were not changed.

A NumPy reference-checker shape error was corrected. A matched-weight/path probe then established float32 finite-difference instability: float64 finite differences converged to autodiff at about5e-12. Every scientific learner and optimizer was promoted to float64 before training, retaining the original initialization values and identical architecture/settings. This shared numerical feasibility change and the original design are preserved in precision_change.json and design.initial.json. Failed attempts remain available.

The simulator uses a declared daily log-Euler stock step and exact CIR variance transition with coefficients fixed during each interval. Claims concern this discrete simulator. Two-substep Monte Carlo diagnostics are reported with sampling error; they do not certify continuous-time SLV discretization error at the useful-gain scale. Price/filter refinement checks cover the declared validation paths and independent reference states, not all possible latent states.

Family A changes training model-family coverage as well as allocation: its baseline is pure base SLV, while the adaptive generator has positive off-base support. It is not a pure matched-support reweighting experiment. Family B has shared baseline/adaptive support. Neither synthetic leverage function is claimed to be market-calibrated.

Raw pre-premium risk, mean loss, initial liability value, centered terminal risk, daily dispersion, costs, turnover and positions are retained separately. Specialists are learned comparators, not proven optimal hedges. Empty event-aligned plots in domains without switches indicate no qualifying events, not zero risk.

Compilation, evaluation/dataset generation, checkpoint I/O and active training are recorded per arm. Provider cost includes setup, validation, failed attempts and exports. The original MSE and rejected trinomial campaigns remain closed; prior authorized pilot cleanup had already deleted some datasets/checkpoints.

## Evidence

- Canonical GPU outputs: `/home/jump/vast-tools/runs/20261018T114635Z-f4f72780/outputs`.
- Original worker report and LaTeX: `/home/jump/vast-tools/runs/20261018T114635Z-f4f72780/outputs/report.md`, `/home/jump/vast-tools/runs/20261018T114635Z-f4f72780/outputs/report.tex`.
- Raw datasets, per-checkpoint episode arrays, all checkpoints and figures: `/home/jump/vast-tools/runs/20261018T114635Z-f4f72780/outputs`.
- Metadata/hash/protocol audit: `/home/jump/deep-hedging/campaigns/slv-20261018/artifact_audit.json`.
- Independent numeric audit: `/home/jump/vast-tools/runs/20261018T114635Z-f4f72780/outputs/independent_audit.json`.
- Provider accounting and shutdown: `/home/jump/deep-hedging/campaigns/slv-20261018/provider_billing.json`.
