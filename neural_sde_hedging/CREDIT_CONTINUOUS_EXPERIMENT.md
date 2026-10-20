# Continuous neural-SDE credit

Experimental one-factor continuous credit adapter, only under `--market neural-demo`. The default benchmark restores the historical two-factor grid; see [CREDIT.md](CREDIT.md).

Credit now defaults to continuous Gaussian Euler simulation with neural drift,
diffusion and risk-price networks. The hazard factor is not quantized, and default
is integrated analytically conditional on its path. All five training methods,
including **pathwise meta-learning**, support this default adapter.

On a GPU worker:

```bash
python -m neural_sde_hedging.training \
  --problem credit --method pathwise-meta-learning \
  --output /workspace/continuous-credit --rounds 16 --seconds 600 \
  --groups 4 --items 32 --inner-steps 8
```

This is an interface example, not a validated runtime/memory configuration. No
GPU has been rented or numerical test executed for this change.

## Contracts and observations

The existing contract is preserved: a long five-year unit-face ZCB (1,260 trading
days), daily issuance of a new six-month CDS (126 days), and up to 126 live vintage
slots. The agent can buy today's issue and partially sell any existing vintage.
Coupons remain fixed; old vintages cannot be increased. Recovery is 40%, rate 2%,
transaction cost 1 bp per unit traded and protection is capped at two.

The same GRU architecture and initialization are used across credit methods.
Inputs are quoted spread and its change, time, inventory, coupons and CDS marks.
The policy never receives the hazard factor or teacher identity. One-factor quotes
may nevertheless reveal substantial state information; no partial-observability
claim follows simply from hiding the factor.

The existing book_step ledger handles premium/default settlement, vintage aging,
sales and terminal liquidation. It now also accepts explicit continuous quotes;
its original table-indexed interface remains available.

## Generator and pricing reference

The trainable proposal evolves

    X[t+1] = X[t] + [b_phi(X[t]) + sigma_phi(X[t])*lambda_phi(X[t])]*dt
                         + sigma_phi(X[t])*sqrt(dt)*epsilon[t],
    epsilon[t] ~ Normal(0,1),
    hazard[t] = softplus(X[t]).

All three coefficient functions are neural networks. The drift component is
bounded by 2, volatility lies in [.05,.6], and the risk-price term is bounded by 2.
Unlike the previous restricted adapter, volatility and drift both learn. This
remains a one-factor, time-homogeneous model; options have additional hidden
factors. No discrete market selector is used.

Each campaign freezes an initial neural generator as its target/prior and uses
that model's Q drift/volatility as the common pricing reference. This reference
prices the ZCB and CDS across all curricula. Adaptive generators are **synthetic
training proposals**: their changing diffusion is not asserted to share a fixed
continuous-time risk-neutral measure with the reference. We do not reprice the
whole market separately for each teacher update or replay snapshot. This makes
quotes a common deterministic function of the factor history, so path likelihoods
and replay importance ratios compare distributions on the same sample space.

Q marks use a separate finite-state Gaussian-Euler pricing approximation on 601
grid points spanning [-10,3], followed by piecewise-linear interpolation at the
continuous state. Simulation itself is unbounded and continuous. Quotes clamp
at pricing-grid boundaries; quote_boundary_fraction reports this occurrence.
Pricing/interpolation/martingale error requires GPU grid-refinement checks. Prices
are not claimed exact for the continuous Euler law, particularly outside bounds.

## Default integration

For one simulated factor path, conditional day-end survival is

    s[t] = exp(-softplus(X[t])*dt).

We evaluate the causal strategy along the alive prefix. At each date, the same
action is settled once under default and once under survival. The survival book
continues; the default branch terminates. Thus we obtain T default-date losses
and one maturity-survival loss in O(batch * days * vintages), without expanding
T separate complete rollouts.

Default-date weights are prod(s[:t])*(1-s[t]); the survival weight is prod(s).
They sum to one. The factor can continue counterfactually after default because
the loss of each default branch uses only its causal prefix. Default is not made
soft, and recovery is not replaced by its expectation inside the entropic loss.

For each factor path we use the conditional entropic value

    c = logsumexp(log(event_probability) + gamma*event_loss)/gamma.

Then entropic aggregation over these c values equals aggregation over both
factor uncertainty and all default dates. This is exact conditional integration
for the discrete daily Cox-default convention. It removes default sampling noise
but retains Monte Carlo error over factor paths. Finite-path entropic bias and
rare-factor-tail issues remain; ESS diagnostics are over factor paths.

CVaR95 and terminal standard deviation use **probability-weighted actual event
losses**, never the standard deviation or CVaR of c. Daily marked error standard
deviation averages the default/survival daily-error moments, including zero
increments after default. Post-default cashflows are not invented.

## Training, gradients and replay

Baseline, DPG, SOAR and self-play use the same continuous adapter. Density scores
are full Gaussian Euler factor-path likelihoods, with sample coordinates stopped.
Defaults are marginalized inside the objective, so there is no sampled-default
likelihood to add. Scores include diffusion scale/determinants, not just the
parameter-independent primitive Gaussian density. Replay reexecutes stored neural
proposals with new Gaussian noise and recomputes their factor-path densities.

Pathwise meta-learning backpropagates through the complete configured inner
block: neural drift/diffusion, Gaussian reparameterization, survival/default
weights, interpolated quotes, causal CDS decisions, settlement, inner loss
and AdamW updates, then independent target risk. Gradients remain truncated at
outer-round boundaries. Costs, action limits and interpolation are differentiable
almost everywhere; their kinks are not claimed smooth. The pricing reference and
its tables stay fixed; gradients flow through the interpolated state coordinates.

Fresh feedback, fixed periodic validation, ESS checks, source/data fingerprints,
checkpointing and time/disk limits are shared with options. Continuous external
path files contain states[batch,1261,1], representing the hazard factor only.
Default dates are integrated by the adapter. Old packed grid-path files are
incompatible. Never use a final test set as teacher feedback.

The four-GPU launcher defaults to continuous credit under --problem credit;
it still schedules the original four methods. Run pathwise meta-learning as a
separate worker job. The old hyperparameter/readout scripts remain option-specific.

## Previous grid adapter and verification

--credit-simulation grid explicitly selects the earlier finite-state neural
risk-premium-only generator. It retains sampled defaults and rejects pathwise
meta-learning. Its packed files have shape [batch,days+1,3]. It is a separately
labelled reference, not the default credit implementation.

Ruff and py_compile pass. New **unexecuted GPU-only tests** compare every default
branch against the existing ledger, check probability weights, weighted CVaR/std,
daily moments, continuous likelihood and checkpoint round-trip, compare the
pathwise derivative through diffusion/default integration/Adam against finite
differences, and train/resume all five continuous-credit methods.

Before a scientific campaign: run those GPU tests and the credit ledger
regressions, a bounded full 1,260-day/126-vintage smoke, pricing grid refinement,
boundary/martingale checks and time/memory calibration. No empirical success,
correctness test pass or full-horizon resource fit is claimed yet.
