# Distinct regimes, feedback and specialists

The default is now `--market distinct`. Market equations, stochastic arrivals,
Q pricing, per-regime credit grids and hedgers are restored from the historical
training experiment, not from the imposed day-35 diagnostic. See
[README.md](README.md) and [CREDIT.md](CREDIT.md).

- Baseline: fresh base/ordinary training paths every iteration.
- Easy/hard specialists: fresh own-regime training paths every iteration.
- DPG/SOAR: static 1,024-path hard feedback population, 128 uniformly resampled
  without replacement per iteration, shared across within-round comparisons.
  Synthetic learner-training data is separate and comes from the teacher.
- Evaluation: fresh 2,048 paths from each easy/base/hard law at every scheduled
  evaluation (every 16 rounds, last requested round, and completed budget-ending
  round). No evaluation path enters learner training or teacher rewards.

This deliberately retains the user's updated sampling protocol rather than
reproducing the old fixed development-selection campaign. Matching rounds share
benchmark draws across methods. Training uses checkpointed RNG; feedback pool
seed is seed+101 (hard), seed+103 (easy), while evaluation namespaces are
seed+102 (hard), seed+104 (easy), seed+106 (base), folded with the round index.
The static pool is reconstructed deterministically on resume and protected by
source/config fingerprints. `evaluation_easy_*`, `evaluation_base_*`,
`evaluation_hard_*` hold domain-specific results; `validation_*` aliases the
requested feedback target, default hard (easy for the easy specialist).

Specialists reject external feedback-pool overrides; changing their labels
cannot silently substitute a different regime. An adaptive arm can use a
supplied feedback population, whose correct provenance remains the caller's
responsibility. `--regime-pool-size` aliases `--hard-pool-size`.

Baseline and specialist methods can run without the neural-teacher opt-in.
Other neural methods remain paused until explicitly enabled. No rental/job has
been launched. Numerical equivalence, integration and resume tests are added
but unrun; only Ruff and py_compile were executed locally.

The earlier day-35 easy/hard diagnostic and single-factor credit analogue are
preserved exclusively under `--market neural-demo`. They are not the current
benchmark and must not be presented as the historical distinct-regime laws.
