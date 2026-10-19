# Option specialist and adaptive comparison, 3 bp

Scope: fixture only.
All eight checkpoints and learning-rate choices were frozen before terminal test seed 520261017 was opened. Each domain contains 8192 paired whole episodes; bootstrap intervals use 1000 paired whole-episode resamples.
There is one shared training initialization. Intervals describe episode uncertainty conditional on the fitted and selected policies; they do not estimate training-seed uncertainty or prove optimality.
The baseline trains on the ordinary/base family only. Adaptive teachers may sample easy/base/hard, so the comparison changes model-family coverage as well as allocation. Specialists are learned comparators.
Common learner Adam eps_root: 1e-16. When 1e-16 is supplied, all final arms use this shared differentiable-Adam migration; preliminary specialist screens remain separate evidence.
Reported entropic risks and CVaR95 are finite-sample empirical quantities. CVaR uses fractional boundary mass. Entropic ESS is retained in results.json.
No continuous-model convergence or high-default pricing accuracy is inferred from these training/evaluation checks. Numerical claims require the separate supplied pricing audit.
Cost, paid-runtime and provider-shutdown evidence will be attached by campaign control after exports and API verification.
Policy inputs contain causal price history, time and its own holdings only. Daily marked errors use 256 independent episodes and dealer Q prices including future arrivals; Q price replicate errors are reported separately. Final daily stream 920261017 is distinct from terminal 520261017 and preliminary diagnostics. Finite-sample entropic performance does not establish population exponential-moment integrability.

## Actual training budgets

All 24 trials, including unselected learning rates. Active time includes sampling/learning/teacher/history work. Compile, development and monitor times are separate. Checkpoint/export and total paid rental runtime require the provider/control supplement.

| Arm | LR | Updates | Training paths | Feedback paths | Active s | Compile s | Dev s | Monitor s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.0001 | 8 | 2048 | 0 | 10.000 | 2.000 | 3.000 | 4.000 |
| baseline | 0.0003 | 16 | 4096 | 0 | 20.000 | 4.000 | 6.000 | 8.000 |
| baseline | 0.001 | 24 | 6144 | 0 | 30.000 | 6.000 | 9.000 | 12.000 |
| easy_reference | 0.0001 | 8 | 2048 | 0 | 10.000 | 2.000 | 3.000 | 4.000 |
| easy_reference | 0.0003 | 16 | 4096 | 0 | 20.000 | 4.000 | 6.000 | 8.000 |
| easy_reference | 0.001 | 24 | 6144 | 0 | 30.000 | 6.000 | 9.000 | 12.000 |
| hard_reference | 0.0001 | 8 | 2048 | 0 | 10.000 | 2.000 | 3.000 | 4.000 |
| hard_reference | 0.0003 | 16 | 4096 | 0 | 20.000 | 4.000 | 6.000 | 8.000 |
| hard_reference | 0.001 | 24 | 6144 | 0 | 30.000 | 6.000 | 9.000 | 12.000 |
| dpg_hard | 0.0001 | 8 | 2048 | 256 | 10.000 | 2.000 | 3.000 | 4.000 |
| dpg_hard | 0.0003 | 16 | 4096 | 512 | 20.000 | 4.000 | 6.000 | 8.000 |
| dpg_hard | 0.001 | 24 | 6144 | 768 | 30.000 | 6.000 | 9.000 | 12.000 |
| soar_hard | 0.0001 | 8 | 2048 | 256 | 10.000 | 2.000 | 3.000 | 4.000 |
| soar_hard | 0.0003 | 16 | 4096 | 512 | 20.000 | 4.000 | 6.000 | 8.000 |
| soar_hard | 0.001 | 24 | 6144 | 768 | 30.000 | 6.000 | 9.000 | 12.000 |
| selfplay | 0.0001 | 8 | 2048 | 0 | 10.000 | 2.000 | 3.000 | 4.000 |
| selfplay | 0.0003 | 16 | 4096 | 0 | 20.000 | 4.000 | 6.000 | 8.000 |
| selfplay | 0.001 | 24 | 6144 | 0 | 30.000 | 6.000 | 9.000 | 12.000 |
| dpg_mixed | 0.0001 | 8 | 2048 | 512 | 10.000 | 2.000 | 3.000 | 4.000 |
| dpg_mixed | 0.0003 | 16 | 4096 | 1024 | 20.000 | 4.000 | 6.000 | 8.000 |
| dpg_mixed | 0.001 | 24 | 6144 | 1536 | 30.000 | 6.000 | 9.000 | 12.000 |
| soar_mixed | 0.0001 | 8 | 2048 | 512 | 10.000 | 2.000 | 3.000 | 4.000 |
| soar_mixed | 0.0003 | 16 | 4096 | 1024 | 20.000 | 4.000 | 6.000 | 8.000 |
| soar_mixed | 0.001 | 24 | 6144 | 1536 | 30.000 | 6.000 | 9.000 | 12.000 |

## easy

| Arm | Entropic | CVaR95 | Terminal SD | Daily RMS SD | Mean cost |
|---|---:|---:|---:|---:|---:|
| baseline | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| easy_reference | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| hard_reference | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| dpg_hard | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| soar_hard | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| selfplay | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| dpg_mixed | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| soar_mixed | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |

Baseline minus method entropic gains (positive means improvement):


Base retention is reported in the base domain above; compare each adaptive policy against baseline without selecting on this final result.

