# Rolling-CDS deep hedging

This is a local JAX/Equinox research implementation of a **long, unit-face,
5-year defaultable zero-coupon bond**, hedged by buying a newly issued 6-month
CDS each day and selling any fraction of previously purchased CDS protection.
It includes simulation, consistent pricing, a vintage ledger, a recurrent
hedger, entropic training, risk metrics and accounting/resume tests.

The initial implementation assumes long-only protection: selling closes an
existing position and cannot create a naked short. Total protection is capped
at two units of bond face. Both the cap and transaction costs are configurable.

## Calendar and inventory

The default calendar has 252 trading days/year: the bond lasts 1,260 intervals
and each CDS lasts 126 intervals. Thus up to **126 outstanding daily vintages**
can coexist after a purchase, rather than approximately 150. The lengths are
configuration values, not hard-coded policy assumptions.

Every live vintage has its own notional, original coupon and remaining tenor.
At a daily decision, inventory slot zero is reserved for today's new issue;
slot `a` contains the contract issued `a` days ago, with `126-a` days remaining.
The oldest contract settles its last premium and then expires. The book shifts
by one slot at the end of the day.

Actions consist of:

- `buy`: nonnegative notional of today's new CDS.
- `retain[a]`: fraction of vintage `a` to keep; the rest is sold at its current
  contract mark, less transaction costs.

You cannot add to yesterday's vintage. You can retain it, partly sell it, or
fully sell it. Retained coupons never reset to the current spread. Even when
every day's new contract starts at par, older contracts generally have nonzero
values. A sale may generate a payment **or require one**.

## Credit model and pricing

The latent process has Vasicek/Ornstein-Uhlenbeck dynamics:

```
dx = kappa * (theta - x) dt + sigma dW
lambda = softplus(x)
```

A raw Gaussian Vasicek hazard can be negative. This implementation instead uses
a positive transformation of a Vasicek latent process. It is **not** the affine
Vasicek-intensity model and does not use its analytical bond formula.

The daily OU transition is discretized onto a finite grid using Gaussian bin
probabilities. Tail probability is assigned to endpoint bins. Both the
simulator and pricer use that SAME matrix. Prices are exact for this finite-grid,
daily-default model; convergence to the continuous process is not established.
Refine both grid spacing and bounds before scientific use. Defaults are sampled
with probability `1-exp(-lambda*dt)` using the start-of-day intensity, and occur
at the interval end. Recovery is a fixed fraction of face paid immediately at
that default date. Rates are deterministic and constant. Physical and pricing
laws are the same in this first version.

For transition matrix `P`, survival vector `q`, daily discount `d`, recovery `R`,
and time remaining `n`, the pricing recursion is:

```
B[0] = 1; D[0] = 0; A[0] = 0
B[n] = d * ((1-q)*R       + q * P @ B[n-1])
D[n] = d * ((1-q)*(1-R)   + q * P @ D[n-1])
A[n] = d * (dt            + q * P @ A[n-1])
```

`B` is the bond price, `D` the unit CDS protection-leg value, and `A` the unit
annual-coupon premium annuity. A newly issued contract has coupon
`c = D[126]/A[126]` and zero mid-market upfront payment. An existing contract
with fixed coupon `c_old` and `n` days left is worth `D[n]-c_old*A[n]` to its
protection buyer.

These are **custom daily-coupon par CDS**, not standard ISDA contracts. Daily
premium is paid on every exposed interval, including the day of default. This
end-of-day default convention replaces intraday accrual. No quarterly schedules,
standardized coupons/upfronts, auction delays, margin, counterparty default or
calendar conventions are implemented. See the [ISDA explanation of standard
coupon/upfront conventions](https://www.isda.org/2011/11/29/when-it-comes-to-sovereign-cds-collateral-is-king/).

## Cashflow order and loss

Each day:

1. Observe current par spread, past spread change, time remaining and the
   existing inventory; sell selected holdings and buy today's new protection.
2. Pay transaction costs per unit notional changed. New par protection has zero
   upfront mid value. Sale proceeds use each old contract's actual coupon.
3. At the next date, pay the daily coupon. If default occurs, receive bond
   recovery and CDS protection settlement, and terminate the entire book.
4. Otherwise expire mature CDS and age the surviving book. At bond maturity,
   receive the bond face value and liquidate all still-live CDS, with close costs.

The initial bond purchase is funded by the cash account. All cashflows are
discounted to inception. The positive-loss convention is:

```
loss = initial bond value - sum(discounted bond and CDS net cashflows)
```

Cash borrowing/lending is unconstrained at the stated rate. The ledger stores
daily loss as the negative discounted total portfolio gain, including cashflows
and mark changes. Because the terminal book is empty, `sum(daily_loss) == loss`.
This identity also holds on default paths. There is no independently chosen
initial option premium or liability mark in the training objective.

Default days are included in daily-error dispersion; post-default days have
zero error. This is an unconditional fixed-calendar statistic, not volatility
conditional on survival. Report the default frequency alongside tail metrics.

## Policy and objective

All policy state and parameters are PyTrees. The GRU receives current 6-month
par spread, its one-day change, remaining bond life, total protection, and each
vintage's notional, fixed coupon and current mark. It outputs a new purchase and
a separate retention fraction for every vintage. It receives neither tomorrow's
quote nor future defaults. A quoted spread can identify the single latent factor
in this simple model; this is **not yet a hidden-factor benchmark**.

The default action map is a clipped, slightly stretched sigmoid, allowing exact
zero purchases, full retention and full closure at finite logits. The ordinary
sigmoid used in the first smoke run could only approach these endpoints, causing
unavoidable small trades. New checkpoints record `policy_action_parameterization`.
For old checkpoint configurations lacking that field, explicitly construct the
restoration skeleton with `action_parameterization='sigmoid_v1'`; never silently
interpret old weights with the new action map.

Training minimizes empirical entropic risk via stable log-mean-exp. Evaluation
reports entropic risk, CVaR95 with fractional tail mass, terminal mean/SD, costs,
daily mean/SD curves, and RMS of the cross-episode daily SDs. Gamma is in inverse
bond-face-value units. Batch tail estimates can be noisy; the tiny example runs
are infrastructure checks and do not establish hedge performance.
Training logs include default counts per minibatch. Evaluation also reports
sample count, default fraction, tail mass and effective sample size of entropic
weights. Nonfinite pricing, training or evaluation data cannot be published as a
successful run by the CLI.

JIT is applied at top-level pricing, simulation, rollout and training boundaries.
Time evolution and training updates use `lax.scan`. Chex checks array shapes.

## Run

From the project root, use the existing Python environment, or install the
dependencies in `pyproject.toml` with `uv sync`. The tested environment initially
has JAX 0.7.1, Equinox 0.13.0, Chex 0.1.88 and Optax 0.2.4.

```bash
python -m pytest -q

# Fast short-horizon infrastructure check; NOT the 5-year problem.
python -m credit_hedging.train --short-smoke --updates 2 \
    --batch-size 4 --eval-size 32 --hidden-size 8 \
    --output credit_hedging/runs/short-check

# Actual 1,260-day bond / 126-day CDS dimensions; still only a tiny training check.
python -m credit_hedging.train --updates 2 --batch-size 4 \
    --eval-size 32 --hidden-size 8 \
    --output credit_hedging/runs/full-horizon-check
```

Both commands enable 64-bit precision and print their actual backend. Local CPU
execution is supported. No code contacts Hetzner or rents a GPU. Existing output
directories are never overwritten. Outputs include configurations, fixed
diagnostic evaluation paths, daily cashflows, training losses, scalar metrics
and an Equinox checkpoint containing model, optimizer and PRNG state.

Restore a checkpoint by constructing the same model/optimizer skeleton from its
configuration, calling `eqx.tree_deserialise_leaves`, and converting stored key
data with `jax.random.wrap_key_data`. Exact next-update equivalence is tested.
The entrypoint is intentionally a small local training example; it is not a
fault-tolerant campaign runner with periodic saves, CLI resume or final-test
selection.

## Next implementation stages

- Grid/boundary and time-step accuracy studies; larger evaluation sets with
  enough default events for stable tail estimates.
- Finite-budget convergence tests against unhedged and simple CDS-roll hedges.
- Standard-coupon CDS with explicit upfronts, realistic premium dates and accrual,
  if market-contract fidelity is needed.
- Two-factor/hidden-regime intensity dynamics and easy/base/hard distributions.
- Adaptive scenario teachers, full experiment orchestration and campaign reports.

The initial infrastructure does not claim that either specialist gap has been
demonstrated. The shared-seed and shared-architecture requirements remain.

See [AUDIT.md](AUDIT.md) for the fixes, independent accounting tests and measured
grid error. The 301-point default is a finite-grid research model: its local OU
transition variance is about 4.6% above the continuous OU variance at the initial
state. A grid/time convergence study is still required before interpreting small
between-method differences.
