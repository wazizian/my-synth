# Current plan: neural-SDE generators for deep hedging

Updated 2026-10-20. This replaces the previous market-design plan. No jobs or
rentals are authorized by this document update.

All active arms use the same neural-SDE path generator architecture and the same
GRU hedger architecture and initialization. Use one training initialization seed.
The control samples from the fixed initial neural SDE. Adaptive generators use
DPG with GRPO, SOAR with dataset RLOO, or self-play with batch-level GRPO and
expert iteration. Pathwise meta-learning is a separate experimental arm.

The implementation is neural_sde_hedging/. The current adapter hedges a capped
option using price-history observations. A neural-hazard credit adapter remains
future work; historical credit experiments are not the active implementation.

Teacher rewards use fresh target-training batches: default 1,024 paths per round,
shared across all within-round comparisons. The target law is a fixed neural SDE,
independent of the adapting teacher. External neural-SDE path datasets require
separate, disjoint training and validation files. Do not select feedback paths by
current losses without explicitly defining a new research objective.

Evaluate a fixed independent 2,048-path validation set every 16 rounds and at the
last round. Never use validation losses in teacher rewards. Keep the final test
set separate and untouched. Record entropic risk, CVaR95, terminal error standard
deviation and entropic weight ESS/concentration. Daily marked error standard
deviation remains unavailable until a coherent marking adapter is implemented.

Feedback entropic estimates are finite-sample and downward biased. Flag ESS below
10% of batch size, inspect maximum weight, and assess larger batches before
scientific runs. Current defaults are starting settings, not validated choices.

Replay currently stores neural-SDE generator snapshots, executes them with fresh
noise, and recomputes rewards for the current hedger. Retain that behavior pending
an explicit generator-replay versus trajectory-replay comparison and an audit of
the sampling/importance-weight estimator. No replay change is authorized here.

Validate new feedback sampling, periodic validation, resume RNG and pathwise
higher-order gradients on a GPU before a full campaign. Only Ruff and py_compile
may run on the local/control-plane CPU. GPU jobs require durable supervisors,
bounded runtime/cost/storage, checkpoint export and confirmed provider cleanup.
Existing stopped-run evidence remains unchanged in historical artifact folders.
