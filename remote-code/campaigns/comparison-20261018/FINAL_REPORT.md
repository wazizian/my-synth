# Specialist versus adaptive hedging comparison

Completed overnight on GPU workers; reconciled 2026-10-19 10:43 UTC. All campaign GPU instances are absent from a fresh provider query. Approximately USD7.14 remains in the wallet. No numerical workload or tests ran on the Hetzner control plane.

## Result

The credit experiment does not show a clear advantage of DPG, SOAR or self-play over trained own-domain specialists. DPG-mixed improves hard-domain empirical entropic risk versus the ordinary-only baseline by 0.853bp (95% paired interval [0.057, 1.628]bp), but its advantage over the hard specialist is inconclusive: 0.328bp [-0.174, 0.861]bp. All adaptive-minus-specialist comparisons on easy and hard include zero. These are conditional single-initialization intervals, not training-seed confidence intervals.

The credit benchmark is explicitly exploratory: all six original discovery gates failed. Longer training and higher costs did not establish the intended strong easy/hard specialist separation. No gate was relaxed to make this appear successful.

The price-history-only option benchmark also failed the joint specialist gate at all three costs (1, 3 and 10bp). Its adaptive phase was therefore skipped as specified, and its final 520 test stream was not opened. Hard specialists helped, but easy specialists did not reliably improve over the learned baseline. The independent 820 diagnostic confirms that learned hedging strongly improves on tuned constant holdings; the failure is the desired specialist separation, not failure to learn a hedge at all.

## Credit protocol and comparisons

All eight arms completed three independent learning-rate trials, each with 1800 seconds of active work and shared initialization: ordinary-only baseline, easy specialist, hard specialist, DPG-hard, SOAR-hard, self-play, DPG-mixed and SOAR-mixed. Every learning rate remains in the budget table and learning curves. Checkpoint/LR selection used development evidence before the sealed paired final stream was opened. Final evaluation used 8192 episodes per easy/ordinary/hard domain and 1000 whole-episode bootstrap resamples. Policies were evaluated on identical episode draws.

Entropic risks below are in basis points of face value; lower is better. Rounded display values are derived from the exported report, whose full precision and intervals remain authoritative.

| Arm | Easy | Ordinary | Hard |
|---|---:|---:|---:|
| Ordinary-only baseline | 95.1823 | 85.1505 | 84.9097 |
| Easy specialist | 95.2656 | 86.2167 | 85.2770 |
| Hard specialist | 94.3001 | 85.5272 | 84.3851 |
| DPG-hard | 94.5865 | 85.1599 | 84.1873 |
| SOAR-hard | 95.1538 | 84.9976 | 84.6825 |
| Self-play | 95.2163 | 86.2270 | 84.8971 |
| DPG-mixed | 94.5705 | 85.7813 | 84.0568 |
| SOAR-mixed | 94.7488 | 85.8684 | 84.5684 |

The hard specialist's improvement over baseline is 0.525bp [-0.253, 1.251]bp; the easy specialist's gain is −0.083bp [-1.047, 1.091]bp. Thus neither own-domain specialist has a clear final entropic advantage. No adaptive arm clearly improves ordinary-domain entropic risk over baseline. Different risk measures can rank methods differently: for example, hard-domain DPG-mixed has lower entropic risk but slightly higher CVaR95 than baseline. The full report retains CVaR95, terminal SD, daily RMS SD, costs, defaults, turnover and holdings rather than selecting only a favorable metric.

[Full credit report](credit-final/evaluation/launcher/runs/20261019T021407Z-6f5fb002/outputs/evaluation/report.md), [LaTeX source](credit-final/evaluation/launcher/runs/20261019T021407Z-6f5fb002/outputs/evaluation/report.tex), [all trial curves](credit-final/evaluation/launcher/runs/20261019T021407Z-6f5fb002/outputs/evaluation/all-trial-learning-curves.pdf), [specialist gaps](credit-final/evaluation/launcher/runs/20261019T021407Z-6f5fb002/outputs/evaluation/specialist-gaps.pdf), [daily errors](credit-final/evaluation/launcher/runs/20261019T021407Z-6f5fb002/outputs/evaluation/daily-errors.pdf).

## Option outcome and information restriction

The stochastic-shock option model includes future shock arrivals in pricing. The neural policy receives causal stock price history, time and its own holding; it receives no RV, IV, dealer quotes, latent factors, shock labels or domain labels. The reproduced imposed-shock demo and stochastic pricing audit are retained separately from the learned results.

At 3bp cost, the hard specialist improves confirmation entropic risk by 9.042bp [5.107, 13.534]bp, while the easy gain is 0.043bp [−0.501, 0.599]bp. At 10bp, hard improves by 9.071bp [4.601, 14.242]bp, while easy improves by only 0.156bp [−0.600, 0.870]bp. These fail the prespecified joint requirement; hard-only improvement is not a joint pass.

The separate 820 diagnostic at 10bp gives empirical entropic risk easy: static 0.0123332, learned baseline 0.0025206, specialist 0.0025613; hard: static 0.0284584, learned baseline 0.0072202, specialist 0.0063894. This diagnostic did not choose the gate or frozen benchmark. The constant-volatility Black–Scholes control likewise uses no realized-volatility oracle.

## Reliability, scope and costs

The main credit launcher encountered a final export timeout. The independent recovery service completed the same evaluation's export, verified all 234 output files, then removed its worker; it did not repeat final evaluation. The original failure log is preserved. All eight training runs and their selections completed before that evaluation. Self-play resumed once on its original retained disk; known lost work of 7.55641655 active seconds was charged against its original budget. Its three complete history archives were uploaded and independently SHA-verified before source cleanup. See [recovery and interpretation details](RECOVERY_AND_SCOPE.md).

Credit numerical claims are conditional on the audited finite-state law; parent refinement-pass flags do not certify continuous-model convergence at the production grid. Option empirical entropic scores have a further population-tail limitation described in [OPTION_SCIENTIFIC_LIMITATIONS.md](OPTION_SCIENTIFIC_LIMITATIONS.md). No seed robustness, global optimality, or population-integrability claim follows from this experiment.

Fresh deduplicated provider records cover 55 instances and USD35.10 of reported charges across the original credit/option campaigns, including work preceding this nine-hour objective. This is not the cost of the latest USD20 top-up alone. Billing may lag. The provider reports no remaining instances, and current compute burn is zero. A private 120GB artifact volume is retained for complete self-play histories at approximately USD0.32 per 30 days (USD0.96 per 90 days), with restore manifests and a 2027-01-16 retention review. It is a single-host volume, not replicated storage. See [financial and shutdown receipt](FINAL_COST_AND_SHUTDOWN.json) and [storage contract](storage-contract.json).

The evidence does not justify spending more on the same benchmark merely to obtain a positive result. A next experiment should first establish on independent development episodes that the intended family difference changes the optimal causal hedge enough to produce a reproducible learned-specialist gap, then freeze that benchmark before repeating adaptive comparisons. More training alone has now been tried without establishing that gap.
