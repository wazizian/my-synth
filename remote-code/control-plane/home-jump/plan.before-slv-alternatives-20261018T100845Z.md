# Revised plan: adaptive deep hedging with informative easy and hard markets

Updated 2026-10-18. This supersedes the scientific design of the completed MSE pilot.
Status: planning only; this document update does not launch jobs or authorize new rentals.
Use ONE training seed, shared across methods. No repeated-seed experiment or architecture ablation.

## 1. Research question and changes

Can an adaptive training distribution teach a fixed hedger to hedge shifted markets better than fixed base sampling?
Find genuinely easy-to-hedge and hard-to-hedge situations, with a meaningful baseline-to-specialist performance gap on BOTH domains.
The previous fixed-regime Heston reweighting produced almost no hard-specialist advantage; do not reuse it as an accepted benchmark by default.
Use entropic risk for learner training and target feedback. Report terminal CVaR, terminal entropic risk, and standard deviation of daily hedging error.
Keep the same hedger architecture and learner settings everywhere.

Preserve the completed pilot's results, code snapshots, datasets and checkpoints. New experiments need a separate campaign ID and output namespace; never overwrite results/ from the completed campaign.
The previous authoritative plan is backed up beside this file.

## 2. Benchmark design: difficulty and specialization are separate requirements

Define base P0, easy Peasy and hard Phard before final experiments. Keep payoff, trading dates, information set, hedging instruments, constraints, costs and normalization identical across domains.
Easy must mean lower residual hedging risk under a capable specialist; hard must mean higher residual hedging risk, not merely an arbitrary premium mismatch.
Separately require that training on each target domain benefits the same hedger relative to base training.

Candidate easy dynamics:
- Low and stable volatility with weak volatility-of-volatility and no or very rare regime changes.
- A distinct but interpretable market where a specialized low-turnover hedge can outperform the base-trained policy.
- Low absolute loss alone is insufficient: the easy specialist must beat baseline on easy.

Candidate hard dynamics:
- Standard context followed by an unannounced persistent change in volatility level and volatility-of-volatility during hedging.
- Random switch times leave enough post-switch trading dates for adjustment.
- An optional price jump followed by a persistent change is a bounded alternative candidate.
- Do not rely exclusively on an isolated unpredictable jump: it may mostly create unavoidable instantaneous loss rather than learnable adaptation.

The regime indicator is not a policy input. All policies receive the same causal observations.
Baseline and adaptive generators share a scenario family and support: baseline assigns positive, potentially small probability to shock scenarios.
This tests allocation of training samples, not exclusive access to scenarios withheld from baseline.
Freeze a manageable finite menu so generator probabilities, score terms and KL remain exact and inspectable.

Pricing and information:
- Use one coherent simulator/pricing model for all methods and domains.
- A traded option must be priced consistently with switch/jump risk and the stated observable information. Do not bolt shocks onto spot while retaining incompatible old Heston marks.
- Stock-only hedging is a permissible simpler design choice, applied globally before freezing the shared architecture.
- Even with stock-only hedging, daily liability marks need a declared, validated valuation rule.
- Exclude hidden-state leakage and future switch knowledge from policy features and market observations.

## 3. Bounded development gate before adaptive experiments

Use exactly one training seed, initially 20261017, paired across all methods. Use separate PRNG namespaces for training, teacher feedback, benchmark development, checkpoint selection and final evaluation.
Independent evaluation episodes are not additional training seeds.

First train baseline, easy_reference and hard_reference with identical architecture, optimizer, active-time caps and checkpoint opportunities.
Evaluate all three on base, easy and hard development sets.
Check:
1. A credible ordering of residual difficulty: easy below hard for their specialists, with daily-error dispersion and tail outcomes supporting the interpretation.
2. Baseline materially worse than easy_reference on easy.
3. Baseline materially worse than hard_reference on hard.
4. Numerical/pricing errors materially smaller than the observed gaps.
5. The differences are not explained solely by fixed-premium shifts or unequal optimization opportunities.

Before running candidates, record a small bounded candidate list and an absolute minimum useful improvement for each target in normalized risk units.
Calibrate thresholds from declared pilot scales, not from final outcomes. Relative gaps may supplement absolute thresholds only when denominators are stable and positive.
Use paired independent development episodes and episode-resampling intervals to assess evaluation noise; preserve within-episode dependence.
These intervals are conditional on the trained policies. Do not describe them as training-seed uncertainty or general statistical robustness.
A selected candidate must pass a fresh development confirmation set after screening. Retain every rejected candidate and reason.
If no candidate passes within the diagnostic budget, report an uninformative benchmark and stop before the adaptive campaign; do not manufacture a favorable gap.
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
The previous USD 25/eight-hour campaign is closed. This plan update does not restart its budget or launch workers.
Before a later execution, establish a distinct authorized campaign ledger, deadline and budget for eight arms and the bounded benchmark gate.
Do not infer fresh spending authorization from the old plan.
Preserve full resumable PyTree states, PRNG states, teacher histories, source/config hashes and failure records.
Only the execution supervisor manages rentals, reservations, export and confirmed shutdown.

## 8. Implementation sequence and acceptance

1. Preserve the pilot and create a new campaign namespace.
2. Specify a bounded market candidate menu, observable information, pricing model and daily accounting.
3. Implement entropic learner/teacher objectives and all three metric pipelines.
4. Validate accounting identities, cash translation, CVaR conventions, finite gradients, causality and checkpoint resume on GPU.
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
