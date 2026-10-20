# Neural-SDE teacher for deep hedging

**Neural teacher paused:** neural-SDE fitting, hyperparameter studies, readouts, and
campaign launches are disabled by default; neural teacher training also requires opt-in.
The non-neural `hard-specialist` and `easy-specialist` arms are available. Source is retained for future work.
Explicitly setting `NEURAL_SDE_ENABLED=1` re-enables these entry points; this does
not authorize a GPU rental. Cleanup/export/watchdog operations remain available.
The price-only fitting change is pending: the retained fitted-SDE implementation
still uses full simulator states. No new numerical validation has been run.


This is the editable implementation. `../remote-code/` is a separately hashed,
snapshot of historical Hetzner code. Current training uses this package; old campaigns are evidence only.

## Easy and hard benchmarks restored

The default downstream target is the fixed persistent-shock **hard regime**,
with 1,024 static training paths and 128 resampled feedback paths per iteration.
Evaluation simulates 2,048 fresh paths from each of easy and hard at each evaluation point. DPG and
SOAR optimize against the hard subset; `hard-specialist` trains directly on
that same static hard pool with the same learner architecture and initialization.
`easy-specialist` trains on a separate static easy pool using the original
no-shock options dynamics. It uses the same learner settings and automatically
selects easy as its training regime. Both specialists are evaluated on both regimes.
See [HARD_REGIME.md](HARD_REGIME.md) for exact laws, limitations and commands.
This supersedes older descriptions below of fresh neural-prior feedback and
fixed validation. The price-only fitted-SDE change remains pending.

## Dynamics and observations

The teacher learns a **continuous distribution of full paths** using a neural SDE. The joint state is log spot and stochastic
latent factors. Neural networks parameterize latent drift, joint diffusion, and
the P/Q market price of risk. All latent factors themselves evolve as SDEs.
Diffusion is nonsingular with explicit bounds; drift coefficients are bounded.
See [DYNAMICS.md](DYNAMICS.md) for the equations and discrete P/Q conventions.

The shared GRU hedger receives log prices, returns, time, and its own previous
holding. No latent state, volatility estimate, scenario label, or teacher identity
is supplied. Every arm for a given problem starts from the same architecture and initialization.

The runner supports both `--problem options` (an option hedge with a dollar
holding cap) and `--problem credit` (a five-year long ZCB hedged using daily
new six-month CDS and partial sales of existing vintages). The credit adapter
defaults to a continuous neural hazard SDE with trainable drift, diffusion and
risk-price networks, a fixed neural pricing reference, and exact conditional
default-date integration. All five methods support continuous credit. See
[CREDIT.md](CREDIT.md) for equations, conventions and restrictions. Each problem
uses one common GRU architecture/initialization across its training methods;
credit requires vintage inputs and outputs absent from the option hedger.

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
* Whole **Euler factor-path densities** replace sequence probabilities. Initial
  states are fixed and conditional, with no learned initial-state distribution.
  Every generated factor/spot transition and diffusion log determinant is included. Brownian
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

The adaptive arms generate synthetic learner data through the problem's neural
SDE (`NeuralSDE` for options, `NeuralCredit` for credit). The fixed-prior control
samples the initial neural model. The specialists instead train directly on
fixed paths from their respective easy or hard regimes. The target law is separate from the generator.

By default, generate **1,024 fixed hard training paths**, then sample **128**
without replacement per round. DPG, SOAR and pathwise meta-learning share that
subset across all within-round downstream comparisons. Self-play's alignment
reward remains defined on its generated round pool. Feedback subset randomness
is checkpointed; the fixed hard pool is regenerated deterministically on resume.

Each evaluation uses **2,048 freshly simulated paths per regime**, easy and
hard, from separate RNG namespaces. Evaluate every **16 rounds**, at the last requested round, and when a
completed round exhausts the soft time budget. Matching rounds use common draws
across methods. Evaluation never enters teacher rewards. A separate final test
protocol remains required for scientific comparisons.

An optional `--feedback-paths training.npz` replaces the static hard training pool.
For fixed external training/validation data, explicitly select
`--target-regime external` and supply both `--feedback-paths` and
`--validation-paths`. Files contain `states[B,T+1,D]`, with D=3 for options and
D=1 for continuous credit. Independent provenance is the caller's responsibility.
`--target-regime prior` explicitly selects the old neural-prior demonstration.

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
option adapter does not produce Q-marked daily errors yet; it never labels returns
or unmarked P&L as that metric. The credit adapter reports actual daily discounted
marked hedging errors from its bond/CDS ledger. The old source and reports retain their existing
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


Additional baseline: `fitted-sde` fits the neural SDE to full-state hard-regime
training paths using the Gaussian likelihood at every daily transition, then
freezes it and trains the same hedger on fresh simulations. It supports options
and continuous credit; independent validation is required. See [FITTED_SDE.md](FITTED_SDE.md).
Fitting is privileged full-state inference, not price-only latent inference.
The existing four-arm launcher is unchanged; GPU checks remain unrun.
