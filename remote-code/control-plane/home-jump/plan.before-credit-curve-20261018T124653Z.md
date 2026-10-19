# Revised plan: adaptive deep hedging with informative easy and hard markets

Updated 2026-10-18: BS/SLV/hidden-regime SLV plus stable-versus-stochastic-volatility alternative. This supersedes the completed MSE pilot and rejected trinomial benchmark design.
Status: planning only; this document update does not launch jobs or authorize new rentals.
Use ONE training seed, shared across methods. No repeated-seed experiment or architecture ablation.

## 1. Research question and changes

Can an adaptive training distribution teach a fixed hedger to hedge shifted markets better than fixed base sampling?
Find genuinely easy-to-hedge and hard-to-hedge situations, with a meaningful baseline-to-specialist performance gap on BOTH domains.
The previous fixed-regime Heston reweighting produced almost no hard-specialist advantage; do not reuse it as an accepted benchmark by default.
Use entropic risk for learner training and target feedback. Report terminal CVaR, terminal entropic risk, and standard deviation of daily hedging error.
Keep the same hedger architecture and learner settings everywhere.

Preserve all currently retained pilot and rejected-gate artifacts. Prior authorized cleanup already removed some pilot datasets/checkpoints; do not claim those remain available. New experiments need a separate campaign ID and output namespace; never overwrite the old results or the closed entropic-20261018 gate.
The previous authoritative plan is backed up beside this file.

## 2. Two alternative benchmark designs to test

Test two bounded design families, not an unrestricted search. Family A is the user's BS/SLV proposal; Family B is the earlier stable-versus-stochastic-volatility proposal. Both must satisfy the same specialization and residual-difficulty gates. A family is a hypothesis, not an accepted benchmark.
Before numerical screening, freeze one concrete parameter configuration for each family, its pricing/observation specification, diagnostic runtime, thresholds and evaluation protocol. No post-outcome severity or threshold adjustment. Evaluate both families if the later authorized diagnostic budget allows; apply a predeclared choice rule (A first if both pass) rather than selecting whichever produces the largest gap. If both fail, stop and report.

Common scientific controls:
- Same payoff, maturity, trading dates, context length, stock-only hedging, cash account, position bounds, costs and normalization in all domains and arms.
- The same recurrent hedger observes causal stock history, time, contract information and previous holdings. No true variance, hidden factor, regime label, model-family label or future switch information is supplied.
- Do not feed simulator-state liability marks into the policy. Daily marks used for evaluation are computed separately with the declared information set.
- Use zero rates/discounted units and initially take physical and pricing dynamics equal, with martingale checks and no artificial drift-trading advantage.
- Start by approximately matching expected integrated variance across domains, as a control against mere volatility/payoff scale differences. Also report initial liability values; variance matching alone does not match option prices or establish difficulty.
- Costs are the same everywhere. Neither family may create specialization simply by giving methods different instruments, information or capacity.

### Family A: BS easy, one-factor SLV base, hidden-regime SLV hard

Easy: Black-Scholes with fixed, declared volatility sigma_BS:
    dS_t / S_t = sigma_BS dW_t.
Continuous frictionless BS replication is an analytical reference, not a claim of zero error with discrete trading and costs. Record a BS delta strategy as a diagnostic reference only; it is not a replacement for the learned easy_reference arm.

Base: one-factor stochastic-local volatility with fixed parameters:
    dS_t / S_t = L(t,S_t) sqrt(v_t) dW_t^S,
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW_t^v.
L is a frozen positive, bounded, nonconstant local leverage function; document whether it is synthetic or calibrated. Do not claim market calibration without performing it.
The base variance is already hidden under the shared daily-price observation rule. Calling a model SLV does not make its variance observable.

Hard: the same SLV structure with an additional hidden persistent regime Z_t:
    dv_t = kappa (theta[Z_t] - v_t) dt + xi[Z_t] sqrt(v_t) dW_t^v.
Freeze a finite-state transition law, initial prior and factor correlations. The regime changes long-run variance and volatility-of-volatility; it is not exposed to the learner. Use nondegenerate volatility innovations not almost perfectly correlated with stock returns. Leave enough post-change dates for adaptation.
The additional state must be partially inferable from past prices and persistent enough to affect subsequent hedging; entirely unpredictable extra noise may only raise irreducible risk.
An additional continuous latent factor is a possible future extension, not an extra candidate in this bounded screen. Price jumps are likewise not required in this first comparison.

The expected hierarchy is stable BS hedging, then ordinary stochastic-volatility risk, then additional filtering and regime uncertainty. This is only a hypothesis: measure residual difficulty and specialist gaps. A base-trained SLV model may transfer well to BS; that is a valid negative result.

For this family the fixed baseline trains on base SLV, easy_reference on BS and hard_reference on hidden-regime SLV, as proposed by the user. The adaptive generator can allocate among the predeclared family components. This comparison therefore changes training-distribution/model-family coverage as well as sample allocation; do not describe it as pure reweighting of identical baseline support. Generator initialization must give every candidate component positive probability so exploration is possible; record the exact small off-base masses and KL reference. Do not silently replace the pure-SLV baseline with a mixture. A separate matched-support baseline would require an explicitly scoped follow-up, not an unreported ninth arm.

### Family B: stable easy dynamics versus incompletely hedgeable volatility risk

This is the earlier alternative suggestion, not a rerun of the rejected trinomial models.
Use a common finite stochastic-volatility/SLV scenario menu:
- Easy: stable low-volatility-of-volatility paths, with smaller daily movements and few regime changes. Avoid nearly deterministic prices. The intended specialization is low turnover: aggressive rebalancing costs more than the risk it removes.
- Base: intermediate dynamics. A fixed mixture is dominated by ordinary scenarios and includes small positive mass on every easy/hard menu component.
- Hard: high volatility-of-volatility, an independent component of volatility risk, and an unannounced persistent regime change during trading. Stock alone cannot directly hedge volatility innovations. Keep sufficient post-switch time.
- The hard specialist should improve risk control; the easy specialist should exploit a better cost/rebalancing trade-off. These mechanisms must be checked from turnover, costs, positions and risk trajectories, not asserted from model names.

Use the same positive transaction-cost rate in every domain. Preserve roughly comparable integrated variance where feasible rather than making hard only a larger-volatility copy. Baseline and adaptive sampling have the same menu support in this family, so the comparison more directly tests training allocation. Freeze all mixture weights in advance.

### Pricing, filtration and residual risk for both families

Specify the pricing law and observable information separately for each family/domain, including what model parameters are known and which realized factors are latent. Unknown state is not unknown model specification. Use coherent liability values conditional on the declared information, not true-state oracle marks presented as observable prices. Specialists receive no privileged observations.
With daily observations, stochastic variance requires inference even in base SLV. Any filter for hard must account for both stochastic variance and the regime, rather than assuming either is observed. Validate conditional marks using independent reference calculations; record filtering and pricing approximation errors.
Stock-only hedging avoids a second traded option whose price could reveal latent state, but does not remove the need for validated liability marks for daily-error metrics. Keep those marks out of policy features for this first screen.

Do not rank difficulty using raw pre-premium entropic risk alone. Alongside required raw metrics, compare terminal SD, mean-centered terminal entropic risk and mean-centered CVaR for the corresponding specialists. Report daily-error dispersion, mean error, initial liability value and expected costs separately. Centering removes mean costs too, so centered metrics cannot replace economic risk or cost reporting.
Reject a claimed easy/hard ordering if it only reflects a higher liability value or if terminal residual-risk diagnostics materially contradict it. The previous trinomial gate displayed exactly this failure mode.

## 3. Bounded development gate before adaptive experiments

Use exactly one training seed, initially 20261017, paired across all methods. Use separate PRNG namespaces for training, teacher feedback, benchmark development, checkpoint selection and final evaluation.
Independent evaluation episodes are not additional training seeds.

First train baseline, easy_reference and hard_reference with identical architecture, optimizer, active-time caps and checkpoint opportunities.
Evaluate all three on base, easy and hard development sets.
Check:
1. A credible ordering of residual difficulty: easy below hard for their specialists, assessed using daily-error dispersion AND centered terminal risk/tail diagnostics. Raw liability-value differences are not sufficient; retain mean loss and costs separately.
2. Baseline materially worse than easy_reference on easy.
3. Baseline materially worse than hard_reference on hard.
4. Numerical/pricing errors materially smaller than the observed gaps.
5. The differences are not explained solely by fixed-premium shifts or unequal optimization opportunities.

Before running candidates, record a small bounded candidate list and an absolute minimum useful improvement for each target in normalized risk units.
Calibrate thresholds from declared pilot scales, not from final outcomes. Relative gaps may supplement absolute thresholds only when denominators are stable and positive.
Use paired independent development episodes and episode-resampling intervals to assess evaluation noise; preserve within-episode dependence.
These intervals are conditional on the trained policies. Do not describe them as training-seed uncertainty or general statistical robustness.
A selected candidate must pass a fresh development confirmation set after screening. Retain every rejected candidate and reason.
If neither predeclared family passes within the diagnostic budget, report an uninformative benchmark and stop before the adaptive campaign; do not manufacture a favorable gap or expand the menu automatically. A screening failure does not authorize opening confirmation or final tests.
Freeze distributions, pricing, metric conventions and dataset hashes before opening final tests.

## 4. Identical hedger and controlled comparison

Use the same recurrent architecture, observation history, features, hidden size, action bounds, trading instruments and available information for every arm.
Use the same learner optimizer, learning rate, batch size, block length and gradient clipping.
Use identical initial learner weights from the shared seed; scenario sampling streams are recorded separately.
No method-specific hedger capacity, history removal, privileged state or test-time parameter updates.
Any shared feasibility change must be applied to all arms before scientific runs.

Eight arms:
- baseline: fixed P0 sampling, no teacher.
- easy_reference: fixed Peasy sampling, no teacher.
- hard_reference: fixed Phard sampling, no teacher.
- dpg_hard: adaptive sampling with hard-target attribution.
- soar_hard: adaptive sampling with hard-target improvement feedback.
- selfplay: adaptive sampling with intrinsic historical feedback.
- dpg_mixed: adaptive sampling with balanced easy/hard target attribution.
- soar_mixed: adaptive sampling with balanced easy/hard improvement feedback.

The two specialists are comparators, not theoretical optimal hedges.
No easy-only adaptive teacher arm or extra architecture ablation is required.

## 5. Entropic training objective and premium treatment

Use normalized quantities in inception-spot units and positive numbers for losses.
Let X_theta = (terminal liability - discounted trading gains + all trading/liquidation costs) / S_inception.
Train the hedger to minimize:
    R_gamma(X_theta) = log E[exp(gamma X_theta)] / gamma, gamma > 0.

Freeze one gamma and its units for all methods and domains before comparison.
A constant normalized initial premium p shifts risk by:
    R_gamma(X_theta - p) = R_gamma(X_theta) - p.
Thus the hedge optimization does not need the arbitrary fixed base Black-Scholes premium used by the MSE pilot.
This statement assumes the premium does not change trading constraints or policy information.

Primary comparisons use pre-premium X consistently. If premium-adjusted PnL is also reported, use a documented common convention and label it separately.
Do not claim the minimized liability risk is automatically a utility-indifference price: that requires the no-liability optimization reference.
Implement log-mean-exp stably; check finite values, gradient stability and concentration of exponential weights.
Report the minibatch estimator and finite-batch effects; do not treat a minibatch objective as an exact population risk.

Hard teachers target hard entropic risk.
Mixed teachers use an equal-weight easy/hard objective, with positive fixed domain scales determined only from development and frozen before adaptive training.
Do not divide by a risk value that can be zero or negative; any scaling must have an explicit safe convention.
Selfplay remains intrinsic, but its learner-loss gradients and any expert-iteration component use the revised learner objective.
Audit DPG attribution for a nonlinear batch risk: do not silently replace entropic risk by an average of squared or exponentiated individual losses.
Document exact differentiation and weighting conventions for all teacher adaptations.

## 6. Required evaluation metrics

Report every metric separately on base, easy and hard.

A. Terminal CVaR / expected shortfall:
- Primary tail level alpha = 0.95, fixed across methods.
- Positive losses; empirical upper-tail CVaR computed with fractional boundary mass or the equivalent empirical Rockafellar-Uryasev definition.
- Retain VaR95 and mean terminal loss as diagnostics.
- Use the same pre-premium or explicitly premium-adjusted convention as above.

B. Terminal entropic risk:
- The same frozen gamma as training; numerically stable log-mean-exp.
- Retain effective concentration diagnostics for exponential weights and sample counts.
- No method-specific gamma or post-hoc selection of a favorable risk aversion.

C. Standard deviation of daily hedging error:
Let V_t be the coherently marked liability value, S_t the traded asset price vector, h_t positions held over [t,t+1], and C_t the interval transaction costs.
    e_t = (V_(t+1) - V_t - h_t dot (S_(t+1)-S_t) + C_t) / S_inception.
Assign initial trading and terminal liquidation costs once, with a documented timing convention.
Use V_T = payoff. With discounted units, verify the telescoping identity:
    sum_t e_t = X_theta - V_0 / S_inception.
Report cross-episode SD(e_t) by trading day and its RMS over days as the primary daily-error dispersion summary:
    daily_error_std = sqrt(mean_t Var_episodes(e_t)).
Also retain per-day mean errors so bias is not hidden by dispersion.
Do not substitute terminal shortfall SD for this metric.
For switch experiments, also show event-aligned daily means/SDs around the switch, without exposing switch times to the policy.

Retain terminal and daily raw episode arrays, costs, turnover, positions, sample counts and finite counts.
MSE can remain a secondary diagnostic, clearly labeled; it is no longer the main training or selection objective.
Paired episode intervals preserve entire paths when resampling daily data. They do not account for variation across training seeds.

## 7. Selection, budgets and execution

Use immutable, disjoint feedback, monitor, selection and final-test datasets.
Adaptive checkpoint/configuration selection uses a predefined balanced easy/hard entropic criterion.
Baseline selection uses base development risk; each specialist uses its own domain development risk. Give all arms the same checkpoint schedule and record these different training purposes.
Final-test data cannot drive benchmark design, reward scaling, hyperparameters or checkpoint choice.
If retrospective test curves for retained checkpoints are exported, compute them only after freezing choices and never feed them back into training.

Compare methods at the same active training budget and retain learner updates, simulated paths, teacher work, compile/evaluation/I/O time and paid wall time.
Record teacher feedback counts: mixed feedback may consume more target data and must not hide that cost.
Keep one seed per arm. No seed repeats.

Implementation and numerical work remain on GPU workers; Hetzner is the lightweight control plane.
Both the previous MSE campaign and the entropic-20261018 trinomial gate are closed. The latter rejected both candidates and launched no adaptive training. This plan update does not restart either budget or launch workers.
Before a later execution, establish a distinct authorized campaign ledger, deadline and budget for eight arms and the bounded benchmark gate.
Do not infer fresh spending authorization from the old plan.
Preserve full resumable PyTree states, PRNG states, teacher histories, source/config hashes and failure records.
Only the execution supervisor manages rentals, reservations, export and confirmed shutdown.

## 8. Implementation sequence and acceptance

1. Preserve the pilot and create a new campaign namespace.
2. Specify a bounded market candidate menu, observable information, pricing model and daily accounting.
3. Implement entropic learner/teacher objectives and all three metric pipelines.
4. Validate accounting identities, cash translation, CVaR conventions, finite gradients, causality, conditional pricing/filtering and checkpoint resume on GPU. Resolve the rejected gate's seven accounting tolerance failures through documented precision/implementation work; do not carry forward an all-pass claim or silently loosen tolerances.
5. Run the single-seed baseline/easy-specialist/hard-specialist development gate.
6. Freeze the accepted benchmark, architecture, gamma, scales, datasets and common runtime.
7. Run all eight arms with narrow, symmetric adaptive tuning and continuous export.
8. Freeze selections; evaluate final data; generate plots, tables and a LaTeX report.
9. Audit artifacts and release all paid resources.

Deliverables:
- Frozen benchmark/configuration definitions and rejected-candidate development evidence.
- Shared architecture and initialization provenance for all eight arms.
- Training entropic-loss curves and base/easy/hard evaluation curves for CVaR, entropic risk and daily-error SD.
- Per-day and shock-aligned error plots, selected-checkpoint comparisons and base retention.
- Conditional paired evaluation uncertainty, explicitly labeled single-training-seed evidence.
- Raw episode outcomes, daily errors, all-run metrics, resumable checkpoints, costs and source hashes.
- Honest negative/inconclusive outcomes if no benchmark or adaptive improvement is established.

No claim of across-seed significance, irreducible optimal risk, or adaptation superiority follows from this single-seed design alone.
