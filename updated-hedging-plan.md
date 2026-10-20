# Current plan: hard targets and easy/hard specialists

Updated 2026-10-20. The target regime is separate from the adaptive generator.

1. Restore the options persistent-shock hard physical law from the volatility
   diagnostic: factor half-lives (40,3) days, noise .35, log(2) shock to both
   factors at day 35. Continuous credit uses a new one-factor persistent-hazard
   analogue, not the old two-factor credit benchmark. Verify economic hardness
   before making claims about specialist gaps.
2. Generate 1,024 hard training paths once. Each DPG/SOAR iteration samples 128
   without replacement and shares that batch across its downstream comparisons.
   Their synthetic learner data still comes from the teacher. Do not replace
   the hard objective by risk on the initial neural prior.
3. Evaluate on 2,048 freshly simulated paths from each of easy and hard every
   16 rounds and at the last completed round. Use independent RNG from the static training pool and
   teacher, common evaluation draws across methods at matching rounds.
4. Include `hard-specialist` and `easy-specialist`: train the same GRU directly
   on subsets of their respective fixed pools, with the same initialization,
   optimizer, inner update count and learner batch size. Easy options restore
   the no-shock law with (3,3)-day factor half-lives; credit easy is a new
   no-shock one-factor analogue with a 3-day half-life. One initialization seed; no multiseed study.
5. Record entropic risk, CVaR95, terminal error SD, credit daily marked-error SD,
   and entropic ESS/concentration diagnostics. Finite-sample tail bias remains.
6. Preserve the explicit pause on neural teacher computation. Both non-neural
   specialists are available without the neural opt-in. Neural teacher methods
   still require NEURAL_SDE_ENABLED=1. Price-only fitted-SDE work is pending;
   the preserved full-state fitted baseline is not the requested final method.
7. Only Ruff and py_compile locally/control-plane. GPU validation and numerical
   training remain unrun. Do not rent GPUs without an authorized budget. Use
   detached jobs, checkpoints, export and provider cleanup for authorized runs.

See [HARD_REGIME.md](neural_sde_hedging/HARD_REGIME.md) for implementation,
scientific caveats and the specialist command. The four-GPU launcher still has
its original four arms; run the specialists as separate jobs.
