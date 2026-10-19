# Neural-SDE hyperparameter validation

Authorized 2026-10-19 against remaining Vast credit. This is a one-seed GPU
development study of the new options adapter, not a sealed easy/base/hard or
credit comparison. No computation runs locally or on Hetzner's one-vCPU host.

At launch: account credit approximately $15.98. Reserved campaign cap $5,
listed rate at most $0.65/hour, absolute cleanup timer 12,900 seconds. The
maximum listed compute envelope is $2.33, with a further $2 reserved for
storage/transfers. Earlier stopped allocations remain untouched.

The worker runs under an SSH-independent timeout. Hetzner's lingering user
systemd service supervises orchestration and periodic exports. Both the helper
and an independent systemd timer enforce provider deletion. Boot/setup/runtime
and transfers have separate deadlines. Source is embedded and SHA256-verified.

## Frozen study design

Same GRU32 architecture and initialization seed 20261017 everywhere. Neural-SDE
prior seed 20261018; training RNG starts at 20261019. Capped call payoff 1,
126 daily steps, gamma 10, transaction cost 3 bp, dollar holding cap 1,
zero premium offset, eight inner updates, gradient clipping 1, AdamW weight
decay 0.01. The target is the initial neural-SDE prior under P.

Three independent path sets:

- Teacher feedback: 512 episodes, seed 20261020. Exposed during training.
- Development: 2,048 episodes, seed 20261121. Used only for hyperparameter selection.
- Confirmation: 8,192 episodes, seed 20261122. All configurations are frozen before
  any confirmation risks are evaluated. This is pilot confirmation, not the
  project's sealed final benchmark data.

Screen each configuration for 64 complete teacher rounds:

1. Baseline: learner LR {0.0003, 0.001} × items per group {32, 64}, four groups,
   warmup 32. Select learner LR and batch using development entropic risk.
2. DPG: chosen learner/batch × teacher LR {0.0001, 0.0003, 0.001} ×
   warmup {8, 32}. The shorter warmup tests the fact that resetting optimizer
   state in each eight-update attribution prevents completing a 32-update warmup.
3. SOAR and self-play: chosen learner/batch × teacher LR {0.0001, 0.0003, 0.001},
   warmup 32. Select each method using development entropic risk.

Sixteen screening trials in total. Then run each method's selected configuration
and its original default (unless identical) from the same initialization for
256 rounds. No new initialization seeds. No adaptive retuning based on
confirmation outcomes. Every trial must complete its declared rounds; timeout
or nonfinite updates are failures, not lower-budget successful results.

## Evidence and interpretation

First run the GPU integration suite, including the fixed-prior baseline branch.
Record feedback learning curves, rewards, raw teacher gradient norms, feedback
entropic effective sample sizes, promotion counts and independent development
risks. Export learner parameters and configuration for every trial. Export a
rotating full-state recovery archive during training and selected confirmation
states, with bounded size; original full trial states remain on the worker
until provider deletion.

Report entropic risk, fractional CVaR95 and population terminal loss standard
deviation in underlying units; multiply by 10,000 for basis points in figures.
Paired 256-replicate path bootstraps compare confirmation entropic risk to the
initial learner and to original defaults. These intervals condition on the
single trained initialization and exclude training-seed uncertainty. Daily
marked hedge-error standard deviation is unavailable in this adapter.

A successful finite run establishes operational stability over this horizon.
Learning or an improvement over default settings requires supporting independent
confirmation results. Conclusions are restricted to this prior target and capped
contract; they do not establish convergence or a gain on the intended benchmark.

## DPG diagnostic planned before confirmation

Early screening confirmed that an eight-update reset learner remains close to
initialization. Before any confirmation outcomes were examined, add a frozen
generator readout: train identically initialized hedgers on prior/default/selected
DPG generators for matched 96- and 512-update budgets. Use LR .001, warmup 32,
batch 256, fresh training noise seed 20261223 and new evaluation seed 20261222
(8,192 prior-law paths). This diagnostic does not update teachers or reselect
hyperparameters. Its paired path intervals condition on the single initialization.
