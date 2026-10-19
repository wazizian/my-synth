# Recovery and interpretation supplement

Final reconciliation 2026-10-19 10:46 UTC: training, final credit evaluation, verified exports and provider shutdown are complete. See FINAL_REPORT.md, credit-pipeline-reconciliation.json and FINAL_COST_AND_SHUTDOWN.json. The paragraphs below retain the interpretation and recovery accounting recorded during execution; statements about pending gates describe that historical point. All three option gates ultimately failed.

## Benchmark selection

All six credit discovery gates failed. The GPU freeze selected ordinary credit at 10bp transaction cost by the prespecified ranking of confirmation evidence. This comparison is exploratory. Its easy/hard names describe imposed parameter families; they do not establish an ordering of residual risk. In particular, hard credit residual risk was often lower than easy residual risk in discovery.

Option comparisons require the declared joint gate. The 1bp and 3bp gates failed. At 3bp the hard specialist improved entropic risk by approximately 9.04bp, with a paired confidence interval of [5.107, 13.534]bp; the easy specialist was approximately flat. Thus the hard result does not turn the joint gate into a pass. The 10bp gate remains pending. No threshold is changed in response to these results.

## Credit recovery accounting

Selfplay instance 56640830 resumed the original run `20261019T002055Z-d51d3e98` on its retained disk. The original setup-controller failure receipt is preserved; its masked inner exception is not established. No replacement training run, new initialization, or new seed was introduced.

The last complete checkpoint preceded 7.55641655 seconds of known executed learner work, containing 24 updates that were subsequently lost. Those seconds count against the original active-training budget. A separate 71.70605612-second interval between the last worker log and the first confirmed provider stop is an upper bound on uncertain failure overhead, not measured learner time. An intermediate conservative checkpoint repair is retained for provenance but was superseded before resume. The final repair charges only the known 7.55641655 seconds. Model, optimizer, generator, RNG, progress and history payloads otherwise matched their preserved checkpoint hashes.

The same instance resumed under a durable supervisor with the original remaining wall cap and original deadline. Exact checkpoint replay error was zero. Lost updates remain identifiable in preserved logs; accepted fitted-model update counts exclude lost/replayed work. Paid-instance cost accounting must include the full provider-billed interval, including setup, recovery, compilation, export and other overhead. The final training-budget table alone is not a complete paid-cost ledger.

Evidence: `credit-final/selfplay/launcher/runs/20261019T002055Z-d51d3e98/recovery/repair-known-active-receipt.json`, `manual-handoff-receipt.json`, `remote-resume-progress.json`. The original failure and intermediate repair receipts remain alongside these files.

## Scope of numerical claims

Credit pricing and hedge ledgers are conditional on the audited coherent finite-state model. The discovery numerical summary retains grid-refinement evidence, but its parent `passed` flag can refer to a finer grid than the selected production configuration. It must not be read as certification of continuous-model convergence on the production grid. No final comparison should make that stronger claim.

All learners share one initialization seed; paired episode bootstrap intervals are conditional on the fitted models. They do not measure variation over independent training seeds or prove global optimality. The baseline trains on the ordinary family alone. Specialists and adaptive learners receive different declared training-family coverage; comparisons must retain that distinction.

Option policy inputs are causal stock price history, time and the policy's own holding. Latent volatility factors are available only to the independent pricing and simulation machinery. Future stochastic shock arrivals are included in option valuation. The constant-volatility Black–Scholes control uses current price and time, not realized volatility.

Under the option model's positive costs and unbounded lognormal stock, population exponential moments need not be finite. Empirical entropic comparisons are finite-sample descriptive results, with the detailed limitation retained in `OPTION_SCIENTIFIC_LIMITATIONS.md`. They do not establish a finite population entropic risk.

## Pending financial and shutdown reconciliation

Merge the credit and option instance ledgers by provider instance ID, avoiding duplicate control/audit workers. Report provider charges with their query time and possible billing lag separately from listed-rate estimates. Include failed attempts and recovery overhead; do not attribute all wallet depletion solely to accepted training. Record retained private artifact volume 56629013 separately, including its approximately USD0.32 per 30 days and restore manifest. Preserve that volume until required history archives have another verified copy.

Completion requires verified final artifacts and API-confirmed absence of every campaign GPU instance. This paragraph is a requirement, not evidence that shutdown has occurred.
