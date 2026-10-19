# Credit-hedging infrastructure audit — 18 October 2026

The ledger passed independent cash-account and one-step martingale checks. Four
implementation issues were corrected. The finite-grid model and tiny training
example are **not yet a validated continuous-model experiment or performance
benchmark**. This was a local CPU audit, not a GPU campaign.

## Corrected findings

1. **Evaluation could fail open on nonfinite data.** The CLI checked the model,
   optimizer and training logs, but not all evaluation arrays. `NaN > tolerance`
   is false, so a NaN accounting residual could bypass the threshold and be
   serialized as nonstandard JSON. `validate_evaluation` now checks every result
   and metric array before publishing; pricing tables are checked too. JSON
   serialization rejects NaN/Infinity. Regression tests inject both a NaN metric
   and a NaN daily array hidden behind otherwise finite summaries.

2. **The policy could not represent exact no-trade or full-close actions at
   ordinary finite logits.** Its unmodified sigmoid produced positive purchases
   and retention strictly below one, so it continually bought/sold tiny amounts
   even when transaction costs made holding optimal. The new clipped/stretched
   sigmoid reaches both endpoints; initialization stays inside the trainable
   region. There are zero-gradient plateaus beyond those endpoints, a deliberate
   piecewise-differentiable trade-off. The legacy map remains available as
   `sigmoid_v1`; metadata names the new map. Replaying the existing full-horizon
   checkpoint in legacy mode reproduces its saved terminal losses to about
   7e-15. Historical smoke outputs were not overwritten.

3. **Invalid configuration and risk inputs were accepted.** NaN rates,
   volatilities or costs, infinite bounds, fractional day counts and NaN gamma
   could survive the original comparisons. Configuration validation now requires
   finite scalar parameters and integer dimensions; gamma must be finite and
   positive. Simulation rejects nonpositive/noninteger batch sizes. The guards
   cover model configuration, not arbitrary malformed externally supplied path
   or book arrays: low-level kernels still require their documented invariants.

4. **Small mean reversion could silently freeze the latent process.** The OU
   innovation variance used `1-exp(-kappa*dt)**2`. Cancellation in float32 can
   make this zero even with positive diffusion. It now uses
   `-expm1(-2*kappa*dt)`. The regression test checks the almost-Brownian limit at
   kappa=1e-12 in float32 against sigma-squared times dt, allowing the small grid
   quantization error.

## Financial/accounting checks

- A separate NumPy implementation uses a list of contracts keyed by issue and
  expiry dates, plus a funded bank balance. It does not use the vectorized aging
  or loss formulas. Scripted paths cover first-day default, intermediate default,
  default at bond maturity, survival, partial/full sales, no trading, capped
  purchases, natural expiry and liquidation. Every daily loss agrees with the
  change in discounted net wealth to 1e-12.
- An exhaustive one-day calculation weights every next latent state and default
  by its model probability. A mixed-coupon book with simultaneous sales and a
  new purchase has zero expected discounted daily loss with zero transaction
  costs, to 1e-13. This checks cashflow signs and the agreement between execution
  and pricing, beyond a telescoping identity alone.
- Constant-intensity bond, CDS protection and premium legs agree with independent
  closed-form geometric sums. New CDS have zero mid-market upfront value.
- Existing coupons survive aging and partial sales. Defaults settle all live
  protection once and terminate the book. Expiry incurs no liquidation fee;
  bond maturity liquidates still-live CDS with fees.
- A finite-difference output-bias derivative agrees with backpropagation through
  the entire recurrent rollout, including a default path.
- Causal actions, fractional-mass CVaR, entropic cash translation and exact
  optimizer/PRNG checkpoint resume have regression tests.

## Remaining numerical and experimental limitations

### Grid error is measurable

With the default parameters and initial state x=-3.5, varying only grid size:

| Grid points | Initial 5y bond price | Initial 6m par spread | One-day OU variance relative error |
|---:|---:|---:|---:|
| 151 | 0.8282974510 | 0.01831043670 | +18.3116% |
| 301 (default) | 0.8287867567 | 0.01825629997 | +4.5915% |
| 601 | 0.8289090504 | 0.01824274613 | +1.1479% |

The variance comparison is the transition-row variance against the exact OU
one-step variance, both at the initial state. The 301-to-601 bond-price difference
is approximately 1.22e-4 per unit face, or 1.22 basis points of face. It is **not**
an error bound against the continuous limit. Boundary and time-discretization
convergence remain unchecked. The local transition variance figures are not
relative hedging-loss errors.

Pricing and simulation use the same transition matrix, so this does not create
an internal pricing inconsistency. It does prevent claiming the default model
accurately reproduces a continuous positive-OU intensity without further work.
The default grid was not silently changed during this audit.

### Tail samples and training remain tiny

The full-horizon smoke evaluation has only 64 paths and 6 defaults. CVaR95 uses
3.2 observations of tail mass. Before hedging, the entropic exponential weights
have an effective sample size of about 6.6. Two updates with batch size four
cannot establish learning quality or convergence. Default counts per training
batch and evaluation tail-weight diagnostics are now retained.

The finite-minibatch log-mean-exp objective/gradient is not an unbiased estimator
of population entropic risk. Many small batches can contain no defaults. A
serious experiment needs adequately sized batches/evaluation sets or a validated
weighted rare-event sampling scheme. Such a scheme was not added in this audit.

### Contract and model scope

- Custom par CDS with daily premiums and end-of-day default; not ISDA standard
  coupons, upfronts, quarterly accrual or auction settlement.
- Softplus of a latent OU process, not an affine Gaussian hazard-rate model.
- One factor that can be inferred from its noiseless quoted par spread; not a
  hidden-factor information experiment.
- Long bond and long-only protection, deterministic rates, fixed recovery,
  identical physical/pricing laws and unconstrained funding at the given rate.
- Daily dispersion includes absorbing zeros after default. It must be read with
  terminal risk and default frequency, not used alone to rank market difficulty.
- The CLI saves only after its training chunk; it is not yet a periodic-checkpoint
  campaign runner. Low-level book/path functions assume valid input states.

## Verification

`python -m pytest -q`: **28 passed**, with one upstream Equinox deprecation warning.

The corrected full-horizon smoke used 1,260 trading intervals, 126 CDS slots,
two updates, batch four, 64 diagnostic paths and one seed. It completed with
finite model/optimizer/metric arrays. Maximum accounting residual across its
unhedged, initial and trained evaluations was below 3.1e-15. Outputs are in
`runs/audit-full-horizon/`; these numbers verify execution/accounting, not hedge
quality. No cloud resources were launched.
