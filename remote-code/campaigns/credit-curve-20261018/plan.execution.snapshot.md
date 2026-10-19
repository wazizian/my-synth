# Next experiment: credit-curve risk with a 5-year ZCB and rolling 6-month CDS

Updated 2026-10-18. User selected proposal 1: persistent credit deterioration versus temporary spread spikes. This replaces BS/SLV as the next experiment. The old MSE, trinomial and SLV campaigns remain closed; preserve their artifacts and negative findings.

This is a planning handoff, not a launch command. No new rental or training is initiated by publishing it. Use ONE shared training seed (20261017), identical hedger architecture and paired initialization across methods. No seed repeats.

## 1. Economic problem

Hold one unit face of a 5-year defaultable zero-coupon bond, financed through cash. Hedge its credit risk with same-issuer CDS protection and cash; initially use deterministic rates, fixed common recovery, and identical physical/pricing dynamics.

Every trading day:
- A newly issued 6-month CDS is available to buy.
- The agent may partly or fully sell any protection already held.
- It cannot increase an old vintage or open naked protection shorts.
- Every vintage retains its issue-date coupon, notional and expiry.
- Sales realize that vintage's current mark, which may be positive or negative.

On a 252-trading-day year this is 1,260 bond intervals and up to 126 simultaneous daily CDS vintages. Calendar and counts must be explicit. Preserve the same instruments, costs, notional bounds, observation history and action space in every domain and arm. Liquidate surviving CDS at bond maturity; terminate all exposures correctly on issuer default.

The core question is whether adaptive scenario allocation teaches the same hedger a better trade-off between short-term CDS exposure and long-term bond credit exposure. Raising hazard levels alone is not an accepted source of difficulty.

## 2. Selected market-design hypothesis

A 6-month CDS is particularly sensitive to near-term default risk. A 5-year bond also depends on longer-term risk. Similar current short spreads can therefore accompany different future exposures.

- Easy: one persistent credit factor with a stable relationship between short CDS spreads and long bond exposure.
- Base: ordinary, moderately mean-reverting credit dynamics, with a fixed training distribution defined in advance.
- Hard: a persistent credit-quality factor plus a faster transitory distress factor. Histories that end at similar current 6-month spreads can imply different long-term exposures and different desirability of retaining or replacing old protection.

A starting positive-intensity specification is:

    dX = kappa_X * (theta_X - X) dt + sigma_X dW_X
    dY = kappa_Y * (theta_Y - Y) dt + sigma_Y dW_Y
    lambda = softplus(X + Y), with kappa_X < kappa_Y.

Use a frozen zero/constant Y for a one-factor case. A raw Gaussian Vasicek hazard can be negative, so do not use it untransformed as an intensity. Softplus-OU is not an affine Vasicek hazard model; price the actual specified process.

The fast factor can generate temporary spread spikes; the slow factor captures persistent deterioration. Neither is exposed directly to the policy. Do not simultaneously add recovery uncertainty, rate risk, counterparty default, changing transaction costs or extra hidden regimes to this first experiment.

Define one bounded set of concrete easy/base/hard parameters before training. Initially control for marginal hazard/default-risk scale and report survival curves, initial bond prices and CDS spreads so a mere price-level change is visible. The proposed easy/hard ordering and specialist advantages are hypotheses, not guaranteed consequences of the parameter names.

Specify a finite scenario-generator menu and baseline probabilities. Prefer a fixed base-dominated mixture with small positive mass on target components so baseline and adaptive methods have common scenario support. Clearly distinguish this P0 distribution from its dominant ordinary-dynamics component. If a pure-base baseline is instead chosen before freezing, disclose that adaptive sampling changes model-family coverage as well as allocation. Never quietly switch baseline definitions after seeing results.

## 3. Information and coherent pricing

Use one declared market information set for every learner: observable current and historical CDS quotes, time/contract information, previous holdings, historical coupons, and executable marks of its existing inventory. No simulator factor labels, future defaults, future spreads or privileged specialist observations.

Old-vintage marks at different residual maturities can reveal credit-curve information. Do not claim factors are unidentifiable solely because only a new 6-month CDS can be purchased. Audit the full observation vector. Any prices treated as executable quotes must be generated consistently with the market information/pricing specification; do not insert unobservable true-state bond marks as policy inputs.

Compute bond/CDS marks and default cashflows under the same dynamics and recovery/default-timing convention. If the declared market only knows a posterior over latent factors, valuations must use that posterior; if executable quotes reveal additional state information, document that information instead of calling it hidden. An information-consistent valuation method is required before training.

Keep the first implementation's simplified custom daily-coupon par CDS unless realistic conventions are explicitly scoped. Their new coupon makes upfront mid value zero; an old coupon never resets. They are not standard ISDA fixed-coupon CDS. Daily premiums, default-day premium, recovery settlement, bid/ask cost, expiry and final liquidation must be consistent between pricing and execution.

## 4. Reuse the audited local infrastructure

Audited source is uploaded at /home/jump/deep-hedging/credit-hedging/ with source, tests, pyproject.toml, documentation and tested dependencies. SOURCE_MANIFEST.json records verified SHA256 hashes for all 15 transferred files. This is the one-factor foundation; the two-factor extension remains to implement. No dependencies or jobs were launched by the upload.

Implemented foundation: one-factor positive-OU finite-state model; coherent discrete pricing; long-bond funding and 126-vintage ledger; recurrent policy; entropic training; CVaR and daily-error metrics; independent accounting and checkpoint tests.

Audit constraints:
- 28 local tests passed, including an independent dated-contract cash account, expected self-financing gain, finite-difference gradients and checkpoint replay.
- Fixes include finite-data validation, stable OU variance via expm1, and exact no-trade/full-close actions. Preserve explicit policy versions when restoring old checkpoints.
- The default 301-node grid inflated initial one-day OU variance by about 4.6%; do not claim continuous-model convergence or carry that approximation into a scientific result without validation.
- The 64-path smoke had only six defaults. It establishes execution, not hedge quality or reliable tail risk.
- Implement and validate the two-factor simulator/pricer and its observable information before calling the selected design implemented.

## 5. Cheap economic pre-screen before neural campaigns

On development scenarios only, cross-evaluate simple feasible policies:
- Unhedged bond.
- Fixed target protection notional, replenished as contracts expire.
- Fixed target protection notional with daily replacement/rebalancing.
- A bond/CDS sensitivity-based rule, if feasible using the same observable information.

All strategies obey the same vintage ledger, quotes, trading costs and notional constraints. Sensitivity rules may use declared model/filter estimates but not realized hidden factors. These are analytical diagnostics, not extra adaptive arms.

Inspect default-day loss, survival-path loss, turnover, costs, holding ages and performance across domains. Look for a real strategy trade-off: different credit persistence should change which hedge decisions work. If the same strategy is nearly equally good everywhere, do not spend on a large teacher campaign merely because the model has more factors.

## 6. Single-seed learned-specialist gate

Train baseline, easy_reference and hard_reference using the same recurrent architecture, features, capacity, optimizer, batch/block sizes, initial weights, active-time caps and checkpoint opportunities. Only training distribution differs. Do not give specialists latent state or contract privileges.

Require on paired independent development episodes:
1. Easy specialist meaningfully beats baseline on easy entropic loss.
2. Hard specialist meaningfully beats baseline on hard entropic loss.
3. The hard specialist still faces greater residual hedging risk than the easy specialist, supported by terminal and daily diagnostics, not just initial bond-price differences.
4. Numerical/pricing errors are materially below the minimum useful gaps.

Predeclare absolute useful-gain thresholds in bond-face units based on economic scale, pricing tolerance and the bounded diagnostic design. Do not blindly transplant thresholds calibrated to equity-option MSE experiments. Freeze thresholds before learned-specialist outcomes; do not lower them afterward.

Report conditional paired episode intervals; they are not training-seed uncertainty. Preserve whole paths when resampling. A passing screen must pass a fresh independent development confirmation set. Final tests stay sealed. If the bounded design fails, preserve and report it rather than expanding the search automatically.

Compare learning curves at common active times and report updates/paths. Early learning-speed gaps may be useful, but predeclare any learning-curve criterion before screening; do not relabel a failed terminal-gap gate as success after observing it.

## 7. Objective and metrics

Use positive discounted loss per unit face for the financed long bond:

    loss = initial bond price - discounted net bond/CDS cashflows.

Train all learners with the same fixed entropic risk aversion. Hard teachers target hard entropic loss; mixed teachers target a predefined balanced easy/hard entropic objective with fixed positive scales. Audit DPG's nonlinear batch-objective derivatives rather than substituting an average of individual exponentials. Selfplay uses the same learner loss.

Required per-domain metrics:
- Terminal CVaR95 with fractional tail mass.
- Terminal entropic risk with stable log-mean-exp and one frozen gamma.
- RMS across dates of cross-episode daily hedging-error SD, plus the daily curve.

Retain mean loss, centered terminal risk, terminal SD, initial bond values, default frequency, default-day/survival diagnostics, costs, turnover, holdings by vintage, finite counts, tail mass and entropic effective sample size. Include absorbing zero increments after default in the declared fixed-calendar daily statistic and label it explicitly. Daily dispersion alone cannot determine easy/hard ordering.

Use adequately sized training/evaluation sets for default tails. If rare-event or stratified sampling is introduced, use verified likelihood weights in both objective and evaluation; changing default frequencies without weights changes the target problem. Small minibatch entropic estimates have finite-sample bias and are not exact population gradients.

## 8. Conditional eight-arm comparison

Only after the benchmark gate passes, run baseline, easy_reference, hard_reference, dpg_hard, soar_hard, selfplay, dpg_mixed and soar_mixed.

One shared seed and architecture throughout. Match active training budgets and report learner steps, simulated paths, teacher feedback, paid time and costs. Keep teacher feedback, monitoring, development selection, confirmation and final evaluation streams disjoint. Freeze checkpoint/configuration selections before final tests; preserve all stopped and unsuccessful trials.

Use base development selection for baseline, the corresponding domain for each specialist, and a common frozen balanced easy/hard criterion for adaptive arms. Report base retention separately. Specialists are learned comparators, not mathematical lower bounds.

## 9. Execution and completion

Create a fresh campaign namespace, source/config hashes, budget ledger and deadline before a later authorized execution. All prior campaign budgets are closed. Hetzner remains the lightweight control plane; numerical campaign work belongs on GPU workers, with one supervisor owning rentals, reservations, export and verified shutdown. This plan update launches nothing.

Sequence: source transfer and review; two-factor/information/pricing implementation; numerical and ledger validation; economic pre-screen; learned-specialist screen and confirmation; benchmark freeze; conditional eight-arm comparison; final evaluation/report; export and shutdown.

Deliver reproducible configurations, all-run curves and metrics, raw terminal/daily/default records, full model/optimizer/PRNG checkpoints, numerical convergence evidence, source provenance, cost/shutdown evidence and a LaTeX report. Stop honestly on a failed gate. No across-seed, optimality or continuous-model-accuracy claim without supporting evidence.
