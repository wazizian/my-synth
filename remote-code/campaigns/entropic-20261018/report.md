# Entropic benchmark gate: no accepted benchmark

Neither of the two predeclared candidates passed the required baseline-to-specialist improvement on both target domains. The bounded development gate is complete with a negative outcome. Following PLAN.md §3, the campaign stopped before adaptive training. No benchmark was frozen as accepted, no confirmation set was opened, and no final-test set was opened. This is not a completed eight-arm adaptive comparison or evidence of adaptive superiority.

## Results

A positive improvement is baseline entropic risk minus specialist entropic risk on paired independent screening episodes. Acceptance required the lower endpoint of the paired 95% episode-bootstrap interval to exceed the fixed minimum. Each domain used 8,192 screening episodes and 400 resamples. These intervals are conditional on policies trained with the single shared seed 20261017; they do not measure variation across training seeds.

| Candidate | Target | Improvement | Conditional 95% interval | Required minimum | Result |
|---|---|---:|---:|---:|---|
| trinomial04 | easy | 0.0001480 | [0.0000663, 0.0002289] | 0.0002 | Fail |
| trinomial04 | hard | 0.0006647 | [0.0004228, 0.0009079] | 0.0005 | Fail |
| trinomial06 | easy | 0.0001861 | [0.0001011, 0.0002769] | 0.0002 | Fail |
| trinomial06 | hard | 0.0008108 | [0.0005190, 0.0010917] | 0.0005 | Pass |

Both candidates failed on easy. The first also failed the hard confidence-bound requirement. The second cleared the hard requirement alone. Thresholds and the two-candidate menu were recorded before running; neither was changed in response to outcomes.

## Difficulty interpretation

Both raw screens showed higher hard-domain pre-premium entropic risk, CVaR and daily-error SD. Those comparisons are insufficient to establish the intended residual difficulty ordering. Independent supplementary diagnostics found lower terminal-loss SD, mean-centered entropic risk and mean-centered CVaR for the hard specialist than for the easy specialist in BOTH candidates. For example, first-candidate mean-centered CVaR was 0.030787 on easy versus 0.021871 on hard. The higher raw hard loss partly reflects a more expensive liability. These diagnostics reinforce rejection; they did not change the predeclared screening decision or trigger extra candidates. Daily-error dispersion alone does not resolve the conflicting terminal dispersion evidence.

See [independent numerical and residual audit](numeric_audit_final.json). The finite-menu family is a martingale trinomial model with low-intensity stable easy paths and hidden persistent increases in move intensity on hard paths. It is a discrete research model, not a calibrated Heston market. No accepted benchmark or general claim about market difficulty follows.

## Protocol and retained evidence

- Two candidates; baseline, easy_reference and hard_reference for each. Identical GRU(6,32), scalar bounded stock position, initial weights, Adam learning rate 0.001, batch 256, 8 updates/block, and gradient clipping 1. One training seed; independent namespaces for training, selection and screening.
- Stock-only hedging; 32 observation-context dates and 20 trading intervals; proportional cost 0.001. Hidden switches occur at trading intervals 6–12, leaving 8–14 intervals after the change. Baseline has strictly positive mass on all nine scenarios. Policy inputs exclude hidden labels and future switch times.
- Exact common-P0 Bayesian trinomial liability marks use only realized observations. Target populations are conditional scenario subsets of that common model. Gamma=10, pre-premium normalized terminal loss; CVaR alpha=0.95 with fractional tail mass. Initial costs enter interval zero, liquidation costs the last interval.
- Each diagnostic arm received 180 active simulation/update seconds and checkpoint opportunities at 0, 60, 120 and 180 seconds. Actual block overruns were below one second. Domain-specific selection used base/easy/hard selection risk respectively. All initialization hashes and paired dataset hashes agree where required.
- Retained: 24 learner/optimizer/PRNG checkpoints plus two validation-resume snapshots, 90 raw evaluation array sets, selection and screening metrics on all three domains, training logs, 36 figures, environment locks, source and configuration hashes. No teacher histories exist because teacher training was never entered.
- Finite-minibatch log-mean-exp is an estimator, not exact population entropic risk. Exponential weight concentration diagnostics accompany every evaluation. CPU orchestration performed no numerical experiment; simulation, training, metrics, figures and independent numerical reconstruction ran on the GPU worker host.

## Validation and limitations

Pretraining GPU checks passed for martingale construction, exact pricing recursion, cash translation, fractional CVaR, causal actions, finite gradients and exact checkpoint resume. Independent NumPy reconstruction on the GPU worker later covered all 90 evaluation datasets (1,350 checks). Independent Bayesian marks agree within 4.45e-16; metric implementations agree within 2.10e-16 for entropic risk and 5.56e-17 for CVaR.

The full numerical audit is **not an all-pass result**: seven dataset-level accounting checks exceeded the 1e-7 tolerance. Maximum loss reconstruction error was 1.240e-7 and telescoping error 1.302e-7. All failures are retained in the audit; the tolerance was not loosened after this audit. These discrepancies are much smaller than the prespecified 0.0002/0.0005 useful-gain scales, but numerical certification for a future accepted experiment remains incomplete. The preliminary 256-path check had missed the larger discrepancies. An earlier worker attempt failed its stricter 1e-10 check before any diagnostic training and was exported and destroyed.

Compilation, evaluation and checkpoint I/O were excluded from active training. Per-arm post-warmup wall time is recorded, but their individual compile/evaluation/I/O durations were not separately instrumented; this is a telemetry limitation, not evidence those costs were zero. There were no adaptive teacher feedback costs. The initial and later failed numerical checks are preserved, not relabeled as successful.

Plots with easy-domain event alignment have no switch events by construction; their empty traces do not represent zero event risk. Selection curves are development curves, not final-test curves. No confirmation was warranted after either failed screen.

## Artifacts

- [Predeclared design](design.json), [campaign ledger](ledger.json), [submitted worker source](gate_packaged.py).
- [Worker report](/home/jump/vast-tools/runs/20261018T090952Z-5439b23c/outputs/report.md), [worker LaTeX tables](/home/jump/vast-tools/runs/20261018T090952Z-5439b23c/outputs/report.tex), [all figures](/home/jump/vast-tools/runs/20261018T090952Z-5439b23c/outputs/figures).
- [Authoritative raw output directory](/home/jump/vast-tools/runs/20261018T090952Z-5439b23c/outputs), [worker artifact hashes](/home/jump/vast-tools/runs/20261018T090952Z-5439b23c/outputs/artifact_hashes.json), [launcher inventory](/home/jump/vast-tools/runs/20261018T090952Z-5439b23c/artifacts.json).
- [Artifact/protocol audit](artifact_audit.json): 1,165 checks passed, final export complete, zero transfer failures. This metadata/hash pass is distinct from the numerical audit failures above.
- [Numerical audit](numeric_audit_final.json), [provider billing and shutdown evidence](provider_billing.json), [requirement audit](completion_audit.json).
- [LaTeX final assessment](report.tex).

## Closure

Both paid instances, 56528797 and 56529076, are confirmed absent from active provider state. All known instances have billing records. Provider-reported total is **$0.272**, rounded and subject to later settlement, within the separate $5 diagnostic and $25 overall ceilings. No further rentals were launched. The prior pilot's existing files were preserved; its earlier user-authorized cleanup had already deleted datasets/checkpoints, so they could not be preserved retroactively.

The required next step is a new benchmark design, with a new bounded development plan, if further research is requested. Automatically expanding the candidate list or lowering the thresholds would violate this campaign's stopping rule. Entropic teacher implementation, accepted-benchmark freeze, eight-arm adaptive training and final-test evaluation were not pursued after the mandatory negative gate.
