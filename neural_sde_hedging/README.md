# Neural-SDE teacher for deep hedging

This is the editable implementation. `../remote-code/` is a separately hashed,
snapshot of historical Hetzner code. Current training uses this package; old campaigns are evidence only.

## Dynamics and observations

The teacher learns a **continuous distribution of full paths** using a neural SDE. The joint state is log spot and stochastic
latent factors. Neural networks parameterize latent drift, joint diffusion, and
the P/Q market price of risk. All latent factors themselves evolve as SDEs.
Diffusion is nonsingular with explicit bounds; drift coefficients are bounded.
See [DYNAMICS.md](DYNAMICS.md) for the equations and discrete P/Q conventions.

The shared GRU hedger receives log prices, returns, time, and its own previous
holding. No latent state, volatility estimate, scenario label, or teacher identity
is supplied. Every arm starts from the same architecture and initialization.

The runnable financial adapter is an option hedge with a dollar holding cap.
It is a new implementation, not a retrofit of the old credit campaign. The full
old credit implementation is preserved in the source snapshot. The standalone
method primitives accept arbitrary differentiable learner/target callbacks,
including a future neural-hazard credit adapter.

## Method implementation map

Primary PDFs and a review of the *previous* code are in [papers](../papers/README.md).
The following mechanisms are implemented here:

| Method | Paper mechanisms | Files |
|---|---|---|
| DPG | Fixed inner initialization and optimizer reset; differentiation through all weighted inner updates, actual Adam moments and clipping; cross-group batching; candidate-group relative advantages and clipped GRPO surrogate | `methods.py`, `training.py` |
| SOAR | Multiple candidate datasets; four randomized training repeats per dataset, all from one baseline; shared fresh per-round feedback; repeat averaging; dataset leave-one-out teacher rewards; smoothed promotion threshold, best dataset and median repeat; retained promotion datasets | `methods.py`, `training.py` |
| Self-play | Halfway historical checkpoint; actual bias-corrected Adam preconditioner; absolute JVP alignment; fresh/mutation/replay; new Brownian executions of stored programs; stored proposal snapshots and recomputed behavior likelihoods; detached exp(±20)-clipped importance ratios; mutation exclusion from PG; whole-pool standardized rewards, prior penalty in stopped advantage, positive-reward expert iteration | `methods.py`, `replay.py`, `training.py` |
| Pathwise meta-learning | Direct gradients through reparameterized neural-SDE paths and persistent student optimizer updates to target risk; truncated at outer-round boundaries | `methods.py`, `training.py`, [details](PATHWISE_META_LEARNING.md) |
| Archive | Uniform earlier non-replay program sampling without replacement; parents sampled uniformly over occupied niches; eight positive-reward elites per niche; .97 reward decay per round | `replay.py` |
| Recovery | Atomic fsynced checkpoints; learner/teacher optimizers, random keys, branch promotion state, program bank and budget; per-round learner history; configuration/source/feedback hashes; duplicate-run lock; free-space reserve | `training.py` |

## Explicit adaptations

These papers train language models, not hedgers. This implementation does **not**
claim to reproduce their text-generation benchmarks or convergence theorems.

* Entropic risk replaces additive token loss. Weighted risk includes its weight
  denominator. Self-play weights raw loss JVPs by detached `B*softmax(gamma*loss)`;
  it does not differentiate these weights a second time.
* Whole **joint Euler path densities** replace sequence probabilities. Initial
  states are fixed and conditional, with no learned initial-state distribution.
  Every latent and spot transition and log determinant is included. Brownian
  innovation density alone is not used as a changing-teacher score.
* A reusable program is a frozen neural SDE. Its fresh execution is scored under
  its actual producing snapshot, the current teacher, and the frozen prior. The
  objective is over paths, not a claimed probability density over neural weights.
  Importance correction is componentwise; finite-bank selection and clipped
  ratios introduce the same kinds of off-policy approximation that must be
  assessed empirically. Mutation perturbs an archived SDE's parameters; this
  remains within the coefficient parameterization's bounds. Mutation rows are
  excluded from PG even though their conditional path density is available.
* Quality-diversity niches are 9 squared-return persistence bins × 4 realized
  volatility bins, replacing program depth/length. Admission rewards are
  recalculated on fresh paths and the current learner; old scores only rank
  mutation parents. Default pool fractions are 50% fresh, 25% mutation and 25%
  replay, with fresh fallback while banks are empty.
* SOAR's four repeats use bootstrap inner-training minibatches from one dataset;
  they share one initialization. This is algorithmic variance reduction, not a
  four-seed reported experiment. Accuracy-based promotion becomes risk reduction
  with an explicitly configured threshold. The runner uses the paper's most
  recent three round-average rewards for promotion smoothing. The pure
  promotion primitive also exposes EMA smoothing for separately named variants.
* The included DPG runner uses cross-group batching and one on-policy teacher
  update per round. The GRPO clipping machinery is present; at the first update
  its ratio is one, so clipping does not constrain that initial gradient.
  The primitive also supports separate group attribution.
* Learner and teacher gradients are clipped at norm one. This extra engineering
  choice does not bound Adam parameter displacement or guarantee improvement.
  Both use AdamW with explicitly recorded weight decay (default .01). The
  learner uses a linear warmup then constant learning rate (default 32 updates);
  the self-play preconditioner uses its actual current scheduled rate and
  second moments. These numerical values are financial-adapter settings, not
  claimed copies of the papers' language-model hyperparameters.

## Feedback and financial interpretation

All training arms generate paths through `NeuralSDE`. The fixed-prior control
uses the initial neural SDE; DPG, SOAR, self-play and pathwise meta-learning adapt
neural-SDE generators. Replay and mutation also execute neural SDEs. No categorical
market selector is part of the active training configuration.

By default, each round draws **1,024 fresh target-training paths** from the fixed
initial neural-SDE prior, independently of the adapting teacher. DPG, SOAR and
pathwise meta-learning share that exact batch across all within-round comparisons.
Self-play's alignment reward remains defined on its generated round pool.
Feedback randomness is part of the checkpointed RNG stream.

The fixed **2,048-path validation set** uses a separate random stream. It is
measured every **16 rounds**, at the last requested round, and when a completed
round exhausts the soft time budget. Validation never enters teacher rewards.
A separate, untouched final test set is still required for scientific evaluation.

For a declared target distribution, supply both `--feedback-paths training.npz`
and `--validation-paths validation.npz`. Each contains
`states[episodes,n_steps+1,1+n_latent]`. Training draws a uniform batch without
replacement each round; the training population must contain at least
`--feedback-size` paths. Supply genuinely disjoint files from the desired
neural-SDE target law: hashes reject identical files but cannot certify data
provenance or absence of partially overlapping rows. Validation uses its entire
file. No loss-based mining or target reweighting is applied.

Entropic estimates on finite batches are downward biased and may miss rare
losses. Every feedback/validation evaluation logs ESS, ESS/batch size, maximum
exponential weight, sample size and an `entropic_tail_underresolved` flag. The
flag threshold is configurable with `--entropic-ess-fraction` (default 0.1).
It is a concentration diagnostic, not proof that unseen tails are covered.
Increase `--feedback-size` when feedback is underresolved; check stability on
larger independent validation samples. No automatic batch-size adaptation or
bias correction is claimed. These starting sizes have not been GPU-validated.

The premium is fixed by `--premium` for every path and teacher. The simulator's
Q dynamics are available but no implied-volatility surface or Q-priced mark
is fed to this hedger. Its default zero premium is an accounting offset, not an
arbitrage-free price estimate. Changing that constant does not change entropic
hedger gradients. A study of IV/RV transfer still needs an independently frozen,
coherently priced evaluation design.

Training logs include feedback empirical entropic risk, fractional CVaR95,
population terminal standard deviation and mean loss in underlying currency
units. `metrics.py` can also consume actual daily marked hedging errors. The
new runner does not produce Q-marked daily errors yet; it never labels returns
or unmarked P&L as that metric. The old source and reports retain their existing
daily marked evaluation implementations.

The ordinary-call default remains available (`payoff_cap=None`). Its lognormal
tails can make population entropic risk infinite. Coefficient bounds, replay,
gradient clipping and finite empirical risk do not repair that. An explicitly
selected `--payoff-cap 1` experiment instead bounds the liability; with long-only
dollar holdings and costs below one, losses have an explicit finite upper bound.
This changes the contract and must be reported as such; it is not silently
substituted for the ordinary call.

## Verification and GPU execution

Ruff and `py_compile` pass. On 2026-10-19, all eleven opt-in GPU verification
cases passed on an RTX 4090. Two-round, 126-day smoke runs of DPG, SOAR and
self-play also passed, including separate-process loading of CLI checkpoints.
The tests exposed and verified a fix for replay-bank serialization. See
[smoke evidence](../gpu-smoke-20261019/README.md) for source hashes and logs.
See [GPU_VALIDATION.md](GPU_VALIDATION.md) for the required numerical acceptance
checks, including broader scientific checks not yet performed. No additional GPU rental is implied
by these commands. Do not run them on this workspace or Hetzner's one-vCPU host.

On an already authorized GPU worker, dependencies are the repository's JAX,
Equinox, Optax, Chex and NumPy dependencies, plus pytest for verification:

```bash
NEURAL_SDE_GPU_TESTS=1 python -m pytest -q tests/test_neural_sde_gpu.py
```

A bounded-contract integration example, **GPU worker only**, is:

```bash
python -m neural_sde_hedging.training \
  --method selfplay --output /worker/results/neural-selfplay \
  --seconds 600 --rounds 100 --payoff-cap 1
```

Use `--method dpg` or `--method soar` for the other implementations. Launch
through a durable supervisor, for example `systemd-run --user` with a unique
unit, `WorkingDirectory`, bounded `RuntimeMaxSec`, `KillMode=control-group`, and
persistent bounded logs. Set the hard deadline above the soft inner budget to
allow final export. Store provider instance ID, unit name, source hashes, budget,
output/history locations and recovery procedure before launch. The budget check
is between rounds; an external supervisor must enforce a hard deadline during
compilation or a long inner block. Keep `latest.pkl` and `learner-history/`
together. Resume with the exact same command/source; never reset history or
silently prune bank entries. Export and verify artifacts before worker shutdown.
# Four-H100 pilot

For a detached baseline/DPG/SOAR/self-play allocation with GPU calibration,
artifact exports, spending envelope and provider cleanup, see
[FOUR_GPU.md](FOUR_GPU.md) and the `neural_sde_hedging.four_gpu` launcher.

