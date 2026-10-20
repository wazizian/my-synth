# Neural SDE teacher

These equations describe the synthetic neural proposal, not the fixed distinct benchmark markets. See [README.md](README.md) for default market laws.

`dynamics.py` replaces three categorical scenario weights with a trainable,
continuous path law. All latent drifts, all diffusion coefficients, and the
market price of risk are neural networks. A learner receives log spot history,
time, and its own previous hedge; it must never receive `paths.latent`, a regime
label, teacher parameters, or the likelihood. The simulator intentionally keeps
joint latent paths for teacher likelihoods and diagnostics.

## State, measures, and numerical law

Write the joint state as `Y = (log S, Z_1, ..., Z_m)`. The networks consume
`(t/T, tanh(Y/feature_scale))` and produce a lower triangular matrix `L`, a latent
Q drift, and a market price of risk `lambda`. The law is

\[
dY_t=b^P_\phi(t,Y_t)dt+L_\phi(t,Y_t)dW^P_t,
\quad b^P_\phi=b^Q_\phi+L_\phi\lambda_\phi,
\]
\[
b^Q_{\phi,0}=-\tfrac12\|L_{\phi,0,:}\|^2.
\]

The latent coordinates themselves are neural SDEs, driven by the same Brownian
vector. Spot volatility depends on their states, and triangular noise couples
spot and latent shocks. The default is two latent factors. Each diagonal
diffusion entry lies in `[0.05, 0.60]`, each strict lower entry in `[-0.25,0.25]`,
each latent Q drift in `[-2,2]`, and each market price of risk component in
`[-2,2]`. Bounds are configuration, not learned. Positive diagonal bounds make
every Euler covariance nonsingular; off diagonal bounds and fixed dimension
also give a uniform positive smallest singular value. No absolute bound on
latent states or stock prices is implied.

The implemented model is the **fixed-grid Euler joint law**, not its exact
continuous-time transition law. Every conditional joint increment is Gaussian
with mean `Y + dt*b` and covariance `dt*L@L.T`; its full normalized density is
used. In particular this includes latent increments and the diffusion
determinant, so the teacher can learn diffusion coefficients as well as drifts.
Under Q the Euler log-spot update gives exactly
`E[S_next | joint current state] = S_current`, at zero rates. The continuous
counterpart also has a true martingale stock because spot diffusion is bounded.
Shared diffusion and bounded market price of risk give a coherent continuous
P/Q change of measure. The discrete P/Q likelihood ratio is the corresponding
Euler Gaussian ratio; no continuous-time likelihood is substituted.

Changing diffusion across teachers changes their continuous path laws and can
make continuous path measures singular. Scores here are valid **only on this
finite grid**, for which every teacher has positive Gaussian transition density.
Keep the grid fixed when comparing likelihoods or replay ratios.

Initial log spot is fixed at `log(spot0)` and initial latent factors are zero.
`simulate(..., initial_state=...)` permits an explicitly supplied fixed initial
joint state. All scores are conditional on that state: no initial density is
included. Do not learn an initial-state sampler or change its distribution
between behavior and current teachers without adding its normalized likelihood.

## Interface and differentiation

```python
config = SDEConfig(n_steps=126, n_latent=2)
teacher = NeuralSDE(init_key, config)
noise = sample_noise(path_key, batch, config)  # [B,T,1+m]
paths = simulate(teacher, noise, config)       # differentiable pathwise rollout
scores = path_log_prob(teacher, paths, config) # [B], fixed sample coordinates
```

`SDEPaths.states` has shape `[B,T+1,1+m]`, `innovations` `[B,T,1+m]`,
`log_density` `[B]`, and `times` `[T+1]`. Its `log_spot` and `latent` properties
give the corresponding views. `log_density` records the sampling teacher's
conditional joint log density, evaluated at generated states. Store it detached
with a replay item. For a likelihood/score teacher update, call
`path_log_prob(current_teacher, stored_paths, config)`: this explicitly stops
all sampled states. **Do not differentiate `simulate(...).log_density` as a
score estimator**, because its sampled states retain pathwise dependencies.
`simulate` remains differentiable for a separately derived pathwise objective.
The fixed grid is regenerated from configuration during scoring, so an arbitrary
`paths.times` array cannot silently change the density being evaluated.

No JIT is applied inside the dynamics primitives. The training workflow compiles
complete updates, and replay's top-level `execute_program` wrapper compiles
program execution and scoring together, with static configuration using Equinox
filtering. Sampling and scoring both use `lax.scan`; there are no Python time
loops. Teachers and path batches are Equinox PyTrees.

## Reuse, mutation, and execution

An exact replay item stores joint states, sampling log density, initial state,
integration configuration, and behavior-teacher identity. Its current log ratio
is `path_log_prob(current, item) - stop_gradient(item.log_density)`. Very long
paths can produce poor effective sample size despite bounded coefficients;
importance clipping is a biased engineering choice that must be reported.

Reexecuting a historical generator with new Brownian innovations creates a
**new path**. Its behavior density must be computed using that historical
teacher on the new joint states, rather than copied from an earlier rollout.
Reusing the original innovations under an updated teacher is a common-random-
numbers comparison, not replay of the same state path. If initial states are
mutated, scoring is conditional on the new state; those samples are excluded
from a policy-gradient estimate for the original initial-state law unless its
proposal correction is derived. Learner/expert training can consume such
samples under its separately declared data distribution.

Brownian innovations can be stored and regenerated for exact execution within
the same software, precision, model, configuration, and initial state. Saved
states provide the authoritative trajectory for scoring; a seed alone is not a
portable guarantee across JAX versions or backends.

## Pricing and objective restrictions

The neural Q law is a pricing model only if the hedge experiment actually uses
it for initial premium and interim marks. A fixed benchmark Q model is an
alternative protocol, but then changing this teacher's Q network must not be
described as a change in observed implied volatility. Hidden-state conditional
Q prices are oracle diagnostics, not valid observations for a history-only
hedger. A realistic observed price would require a filtered latent-state belief.

Changing P and Q jointly can change premium as well as hedging difficulty. State
explicitly whether premium is fixed across teachers, teacher-specific, or
subtracted through a centered objective. Do not mix those targets silently.

Bounded diffusion does not bound stock prices, call payoff, or entropic risk.
For an uncapped short call with positive terminal stock liquidation cost,
the population exponential loss moment can remain infinite. Stable log-sum-exp,
gradient clipping, replay, and bounded neural coefficients do not repair it.
Use an explicitly bounded economic loss (for example a fully specified bounded
payoff together with bounded-cost exposure) or change the objective if a finite
population entropic target is required. Even a bounded payoff alone does not
bound terminal hedge liquidation costs for unbounded stock and nonzero holdings.

The optional `Contract(payoff_cap=C)` in `hedger.py` has a stronger, explicit
accounting argument: the policy is long only and caps dollar inventory at `D`
after each of `N` trades, with proportional cost `0 <= c < 1`. Each purchase
reduces cash by at most `(1+c)D`; every sale and final liquidation produces
nonnegative **net** proceeds `(1-c)*shares*S`. Thus
`loss <= C - premium + N*(1+c)*D`, although the liquidation cost itself is
unbounded. `Contract.loss_upper_bound(N)` exposes this bound. The default
`payoff_cap=None` preserves the ordinary call requested in the existing setup,
and deliberately returns an infinite bound. Selecting the capped alternative
is a change of economic experiment and must be reported as such.

This module supplies a continuous neural SDE teacher, not a calibrated market
model or a certificate of better curriculum learning. No numerical experiments
were run on the local one-vCPU control plane; validation is restricted to Ruff
and Python syntax compilation.
