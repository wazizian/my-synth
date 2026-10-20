# Pathwise meta-learning

Run on a GPU worker using the existing resumable training entry point:

```bash
python -m neural_sde_hedging.training \
  --method pathwise-meta-learning --output /workspace/pathwise-meta-learning \
  --seconds 600 --rounds 16 --payoff-cap 1 --inner-steps 8
```

This is a new pathwise meta-learning variant, not a reproduction of the original
self-play paper. The current student generates the teacher's learning signal:
generate paths, train that student, measure its improvement on independent target
paths, then differentiate that improvement back to the generator.

For fixed Gaussian innovations z and bootstrap keys u, minimize

    J(phi) = rho_target(A(theta_t, optimizer_t, G_phi(z); u))
             - rho_target(theta_t).

The graph includes drift and diffusion, every simulated time step, observed price
histories, hedge decisions, costs/payoffs, inner loss gradients, and all inner
clipped-AdamW updates and moments. No score-function estimator, GRPO, detached
alignment rewards, hard replay selection, or expert iteration is used. Random
numbers are exogenous; differentiating their deterministic reparameterization
gives the pathwise estimator. Bootstrap indices do not depend on the teacher.
Absolute costs, clipping and capped payoffs are differentiable almost everywhere;
JAX uses its selected derivatives at kinks.

The student persists across outer rounds. Gradients are truncated at outer-round
boundaries, just as state is committed to checkpoints: this is differentiation
through every operation of the configured inner block, not through the complete
history of teacher updates. Increase --inner-steps for a longer unroll. Memory
and higher-order derivative cost can be substantial. Both optimizers clip
gradient norm to 1; this provides no guarantee of downstream improvement.

Target paths are independent of generated training paths and frozen during an
unroll. A fresh target-training batch is drawn each round and shared within the unroll.
The default downstream target is now a subset of a fixed hard-regime training
pool; evaluation draws fresh hard paths from independent RNG. See
[HARD_REGIME.md](HARD_REGIME.md). Fixed externally supplied training/validation
files require `--target-regime external`.

Atomic checkpointing, GPU-only enforcement, finite checks and time/disk budgets
are inherited from training.py. The four-GPU campaign launcher and hyperparameter
study have not been extended to include this fifth method.

Validation: Ruff and py_compile only on the control plane. GPU-only tests now
include a fixed-noise finite-difference check through SDE simulation and two Adam
updates, plus the method's training/checkpoint integration case. These new tests
have not been executed; no GPU was rented or computation launched for this change.

## Continuous credit

Add `--problem credit` to select continuous neural hazard proposals and the
five-year ZCB/rolling six-month CDS ledger. Default dates are analytically
marginalized conditional on each continuous factor path; entropic aggregation
retains the full loss distribution. Gradients include default probabilities and
all inner updates. A fixed neural pricing reference supplies interpolated quotes.
See [CREDIT.md](CREDIT.md) for the proposal/pricing distinction and unexecuted GPU
checks. The explicit legacy `--credit-simulation grid` adapter remains unsupported
for pathwise training.
