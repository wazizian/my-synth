# Current plan: restore the distinct-regime experiment

Updated 2026-10-20. The default economic adapter is `--market distinct`.

1. Options: original random Bernoulli arrivals (1/63 daily in base/hard, zero in
   easy), simultaneous log(2) factor jumps at interval end, variance cap 1.
   Restore original P/Q half-lives, Q pricing including future jumps, and IV.
   No deterministic day-35 training event.
2. Credit: original calibrated two-factor finite-grid production settings,
   coherent grid/default simulation and per-regime ZCB/CDS tables, original
   five-year bond / rolling six-month CDS ledger. No one-factor substitution.
3. Restore original policies: GRU32, zero output heads, price-history/share-based
   [0,1] actions for options; float64 target-retention/replenishment for credit.
   Same architecture and initialization within each problem. One initialization
   seed; no multiseed. Learners use clip-1/Adam; learning rates/batches/budgets
   remain explicit CLI settings, not copied LR-selection outcomes.
4. Baseline trains on fresh base/ordinary episodes. Specialists train on fresh
   own-domain episodes. DPG/SOAR optimize feedback on a fixed 1,024-path hard
   training pool, resampling 128 each iteration. Synthetic learner data comes
   from the teacher; hard is not replaced by the initial neural prior.
5. Keep the requested fresh evaluation: 2,048 fresh easy/base/hard episodes per
   scheduled evaluation, separate RNG, common draws across arms at matching
   rounds. Preserve independent final scientific testing and selection rules.
6. Report entropic, fractional CVaR95, terminal SD and actual daily marked-error
   SD. Option daily Q marks run only at scheduled evaluations, on declared
   subsets/Q draws, with MC quote-error diagnostics. Keep ESS/concentration flags.
7. Neural teacher runs remain paused by default; baseline/specialists are
   available. Credit pathwise training is rejected for the discrete grid.
   Price-only fitted-SDE work remains pending; no full-state fitting baseline
   is presented as its replacement.
8. Static checks only locally/control-plane. Numerical archived-code equivalence,
   grid/default score, accounting, integration and resume tests need an authorized
   GPU. No rental/jobs launched. The restored hard-credit bond table is ~10 GB;
   calibrate resources before a campaign and use durable detached jobs, budgets,
   checkpoint export and verified provider cleanup.

The earlier continuous credit / dollar-bounded option experiments remain only
under explicit `--market neural-demo`. They are separate scientific settings.
