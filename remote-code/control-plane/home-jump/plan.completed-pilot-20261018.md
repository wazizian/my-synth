# Execution handoff for adaptive deep hedging data generation across easy and hard markets

This is an implementation and execution brief for one supervising Codex session. Build, validate, tune narrowly, run, and report a fully synthetic experiment using JAX and Equinox. Complete the work in one continuing session, using delegated agents extensively and durable local progress records. The deliverable is working code, saved configurations, metrics, resumable checkpoints, figures, and an honest exploratory report. The execution budget is USD 25 total across all GPUs and a maximum eight hours of campaign wall time. Multiple GPUs may run independent jobs simultaneously. Use only infrastructure accounts and credentials actually supplied for this work; never invent access. This is a fresh execution plan: implementation has not started.

## 1. Fixed scope and new research motivation

The goal is useful synthetic training distribution design, not simply robustness against the hardest market and not worst case adversarial training. A generator is a cooperative teacher: it chooses data that helps the hedger learn. We want to distinguish targeted specialization from useful performance across an easy shifted market and a hard shifted market.

There is no historical data, no language model, no need for labels, and no dataset collection. Given a simulated trajectory, trading strategy, and payoff, compute hedging loss directly.

Fix an option liability, traded hedging instruments, trading constraints, and a family of stochastic market processes. Define three distributions before final runs: base P0, an easy shifted distribution Peasy, and a hard shifted distribution Phard. Evaluate every arm on all three. Low raw error on Peasy by itself is not evidence of a better algorithm: compare methods within each distribution and retain the full easy/hard tradeoff.

The motivating questions are:
1. Can an adaptive training distribution improve hedging on a targeted hard distribution relative to training on P0?
2. Does hard directed feedback also help on a different easy distribution, or specialize at its expense?
3. Does adding easy feedback to hard feedback produce a better easy/hard compromise?
4. Does intrinsic self play find a useful curriculum without direct target feedback?
5. Are detailed DPG attributions worth their computational overhead relative to dataset level SOAR feedback?

The user explicitly selected hard-only and mixed feedback, not easy-only teacher variants. The comparison has exactly seven arms:

| Run ID | Student training distribution | Online generator feedback |
| --- | --- | --- |
| baseline | Fixed P0 | None |
| hard_reference | Fixed Phard with fresh training trajectories | None |
| dpg_hard | Adaptive Pphi initialized to P0 | Example attribution of hard target loss |
| soar_hard | Adaptive Pphi initialized to P0 | Dataset level hard target loss improvement |
| selfplay | Adaptive Pphi initialized to P0 | Intrinsic historical gradient alignment |
| dpg_mixed | Adaptive Pphi initialized to P0 | Example attribution of a balanced easy plus hard target objective |
| soar_mixed | Adaptive Pphi initialized to P0 | Dataset level improvement of that same balanced objective |

The hard_reference is the previous oracle, renamed because it is trained for one distribution, not an oracle for all three and not a mathematical performance bound.

DORAEMON is removed. No repeat seeds, context ablation suite, dimensionality sweep, mixed intrinsic reward sweep, extra easy oracle, historical backtest, or extra arms. Mixed here means easy plus hard target data, not a combination of DPG, SOAR and intrinsic rewards. All results are single seed exploratory evidence.

The recurrent hedger observes spot and traded product histories and reacts without test time parameter updates. This is an ICL style implementation; the experiment does not separately establish an ICL mechanism.

The adaptive methods are financial task adaptations of paper mechanisms. The aim is a valid affordable comparison, not literal reproduction of language model experiments. Freeze all distribution definitions and report easy/hard tradeoffs even when the preferred method loses.

## 2. Autonomy and delegation instructions for the supervising Codex

Proceed without routine clarification requests. Resolve reversible implementation choices using this brief and record decisions. Do not stop after writing code or a plan: continue through validation, the budgeted experiment, artifact export, and shutdown of rented compute when access and authorization exist. If access, credentials, or account authorization are missing, prepare code/configuration and lightweight orchestration locally, identify the exact blocker, and do not invent access or leave a paid instance idle while waiting. Do not run numerical preparation or CPU smoke tests on the small supervisor server.

Use agents as much as is productive. The supervisor owns the scientific protocol, integration, central multi GPU experiment queue, money, deadline, and final conclusions. Delegate bounded tasks with explicit file ownership. Agents should develop and review in parallel. Independent trials may train concurrently on distinct GPUs, with one training process per GPU. Agent concurrency must never create untracked rental spend or competing jobs on the same device.

Initial delegation:

| Agent | Responsibility | Suggested ownership |
| --- | --- | --- |
| Market agent | Heston simulation, coherent option marks, pricing tables, numerical checks | src/market.py, src/pricing.py, tests/test_market.py |
| Learner agent | Equinox recurrent hedger, causal rollout, PnL, checkpoint state | src/hedger.py, src/objective.py, tests/test_hedger.py |
| Teacher agent | Generator distribution, DPG, SOAR, intrinsic self play | src/generator.py, src/teachers.py, tests/test_teachers.py |
| Runner agent | Budget watchdog, metrics, resumability, experiment queue | src/runner.py, src/checkpoint.py, src/metrics.py, scripts/ |
| Review agent | Read only mathematical and implementation audit; later figures/report | review notes first; reporting files after integration |

Use fewer agents if the runtime has fewer slots. Agents may delegate further within available slots, but avoid duplicate work. Give agents common interfaces before they edit. Have each return a short summary: files changed, commands checked, important assumptions, remaining issues. Ask for concise outputs rather than full files or long logs.

Only the supervisor launches paid GPU training, changes scientific defaults, commits integration changes, or rents/terminates instances. Agents must not independently install conflicting dependency versions, launch training sweeps, modify someone else's modules, or create cloud instances.

Maintain these supervisor records from the beginning:

- SUPERVISOR_STATE.md: current phase, frozen decisions, blockers, completed checks, next actions, all active instance statuses, eight hour deadline, spent and remaining dollars. Keep under about 150 lines.
- TASKS.md: agent assignments, ownership, interface contracts, completion status.
- DECISIONS.md: short explanations for changes from this brief.
- runs/manifest.jsonl: every trial, configuration hash, checkpoint lineage, status, and artifact paths.
- budget/ledger.jsonl: per instance paid elapsed time, quoted prices, aggregate cost, outstanding job reservations, and reserve.
- queue/state.json: worker/GPU assignments, runnable jobs, leases, deadlines, completed trials, and pending artifact syncs.

Update the state at every phase transition and after failures. On context compaction, read these files and continue; do not restart development or rerun completed experiments. Keep the session alive while runs execute, check logs at reasonable intervals, and provide occasional useful progress updates. Do not claim an uninterrupted session can survive service failure: checkpoints and state must make recovery possible.

## 3. Two hard limits and parallel GPU scheduling

The campaign has two independent limits: USD 25 total spend across all machines and eight hours of wall time from campaign execution start. Eight hours is not eight GPU-hours. Development, compilation, tuning, training, evaluation, artifact export and shutdown must fit the wall clock. Do not schedule a fifty hour sequential campaign.

Use up to three matched RTX 4090 24 GB GPU workers by default. Start the first worker early, as soon as source scaffolding is ready, for numerical preparation, integration and GPU smoke/profiling; activate additional workers only when jobs are ready. A fourth GPU is allowed only when it shortens the critical path and the central cost projection remains below the cap. Use independent single GPU runs, not distributed data parallel training of this small hedger. One process per GPU; if a multi GPU host is used, isolate device visibility and bound JAX memory per process. Prefer the same GPU model for comparable time budgets.

The supervisor server is very small. It is a control plane only: edit code, coordinate agents, submit SSH jobs, read small log summaries, and maintain manifests/budget state. Do not install a numerical stack, import JAX for backend initialization, simulate trajectories, construct pricing tables, execute numerical tests, or aggregate large arrays there. All smoke tests, correctness checks, dataset generation, pricing preparation, model compilation, training and model evaluation run on rented GPU workers. Use their host CPUs for unavoidable compilation, serialization and lightweight orchestration; this does not authorize CPU test execution on the supervisor. Run large metric reductions/plots on a worker before shutdown, and transfer compact summaries plus preserved raw artifact files without loading them into supervisor RAM.

Every numerical entrypoint must assert that its selected JAX backend/device is GPU before running. Fail fast with a clear error if JAX falls back to CPU; do not silently run a CPU smoke test. Keep float64 reference checks tiny and explicitly GPU-placed. Numerical correctness checks are moved, not removed. Set JAX/CUDA memory limits on workers, not by attempting GPU initialization on the supervisor.

Read actual offers and record all charges, reliability, host CPU/RAM, storage, transfer costs, rental start and minimum billing. Do not assume previous quoted prices are current. Examples only: three GPUs at USD 0.50/hour for six paid hours cost USD 9 before extras; four GPUs at USD 0.70/hour for seven hours cost USD 19.60 before extras. These arithmetic examples are not price guarantees or authorization to exhaust the credit.

Keep USD 3 reserved for recovery, export, storage and shutdown. Normal experiment work has a USD 22 ceiling. If actual price makes the proposed parallelism infeasible, reduce concurrency or job time, not the completeness of metric retention. Do not let agents create workers independently.

Default spending envelopes:

| Work | Maximum allocation |
| --- | ---: |
| GPU environment, compilation, profiling, smoke checks | 3 |
| Shared hedger tuning and baseline/reference diagnostics | 4 |
| Adaptive method tuning | 7 |
| Seven selected continuations and final evaluation | 8 |
| Recovery, storage, transfer and shutdown reserve | 3 |

Unused allocations may move between phases, with an explicit ledger entry. No phase is a spending target.

Default eight hour schedule, adjustable after profiling:

| Elapsed campaign time | Work |
| --- | --- |
| 0:00–1:00 | Parallel source implementation and interface freeze; start the first GPU worker early for pricing preparation and GPU numerical checks |
| 1:00–1:45 | Complete remote GPU integration/smoke/profile and baseline/reference diagnostics; activate other workers only for ready work |
| 1:45–3:15 | Parallel short tuning jobs, validate easy/hard datasets and metric pipelines |
| 3:15–6:30 | Freeze selected configurations and execute/continue seven arms across workers |
| 6:30–7:15 | Complete evaluation for every retained trial and selected final checkpoints; perform worker-side aggregation/plots and synchronize artifacts |
| 7:15–7:45 | Export, verify result/checkpoint integrity, shut down all paid workers |
| 7:45–8:00 | Lightweight report writing from exported summaries and recovery margin; no numerical jobs on the supervisor |

If implementation or remote GPU pricing preparation overruns, shorten optional tuning first. Do not respond by moving numerical work onto the small supervisor. Never compress correctness into invalid shortcuts or skip an arm silently. Start all seven arms early enough to obtain at least one valid checkpoint before extending any favorite arm. Use a work conserving central queue; do not reserve a dedicated idle GPU for each method.

Profile actual kernels before choosing per-run lengths. Initial scheduler targets: 10–15 minute short screens, 30–45 minute promotions, and approximately 45–75 cumulative active training minutes per selected arm, subject to measured throughput and remaining resources. These are provisional scheduling budgets, not scientific convergence guarantees. Seven arms at 60 minutes each need seven GPU-hours, which parallel workers can supply within the deadline.

Implement a central budget watchdog. The authoritative campaign estimate is the sum over every paid worker of rate times rented elapsed time, plus other known charges. Idle time and compilation count. Avoid double counting charges for a multi GPU host. Individual processes must not each assume they own the remaining USD 25. Reserve projected job cost transactionally before launch, reconcile actual cost afterward, and poll provider balance when available.

At estimated USD 22, stop new experiment launches and begin controlled export unless the only outstanding operations are already budgeted finalization. At USD 24 estimated spend, terminate compute work and perform only affordable emergency saving/shutdown. Set worker hard deadlines before the eight hour campaign deadline so loss of the supervisor cannot leave indefinite rentals. Releasing a process is not releasing a rental. Verify all instances/volumes have reached their intended terminal billing state after export; never delete the sole results copy.

Source preparation and lightweight control-plane bookkeeping may happen on the supervisor. All numerical preparation and validation start on the first rented GPU and count toward its paid time, the USD 25 cap, and the eight hour deadline. Build/cache pricing tables once there and transfer the immutable cache to other workers instead of recomputing it. The supervisor must produce an honest partial report if a deadline or validity blocker prevents completion.

## 4. One manageable stochastic market

Use one Heston underlying and one traded European call, with a second European call as the liability. Work in discounted units with rates and dividends zero. For this synthetic experiment take the physical and pricing dynamics equal; there is no risk premium inference problem in the pilot.

    dS_t = S_t sqrt(v_t) dW1_t
    dv_t = kappa (vbar - v_t) dt + xi sqrt(v_t) dW2_t
    corr(dW1, dW2) = rho

Starting proposal for a finite candidate regime menu:

- kappa = 2 per year.
- vbar in {0.04, 0.09}.
- xi in {0.20, 0.40, 0.60}.
- rho in {-0.30, -0.80}.
- This gives 12 regimes, each defining a genuinely stochastic path law.
- Initial variance can start at vbar at the beginning of the context; continue that same variance process into trading. Do not reset variance at time zero.

Use a finite regime menu deliberately: it makes generator densities exact, entropy/KL cheap, the search interpretable, and pricing caches manageable. This is not a test of unrestricted neural process synthesis. Keep the menu frozen for the main comparison.

Fix base and shifted distributions separately from the learner:
- q0: mass 0.70 on the intermediate regime (vbar=0.04, xi=0.40, rho=-0.80), remaining 0.30 uniformly over the other 11.
- qeasy: mass 0.80 on (vbar=0.04, xi=0.20, rho=-0.30), remaining 0.20 uniformly over the other 11.
- qhard: mass 0.80 on (vbar=0.09, xi=0.60, rho=-0.80), remaining 0.20 uniformly over the other 11.

All three have full support. Easy and hard are proposed regime descriptions, not outcomes to enforce by cherry picking. The easy shift reduces volatility of volatility and changes leverage at the same long run variance as the dominant base regime; this avoids making the easy comparison only a reduction in payoff scale. Correlation changes can affect hedge effectiveness in either direction, so confirm numerical behavior in development diagnostics without claiming a theorem of difficulty.

These are shifts in mixture weights within common support, not unseen model families. No benchmark distribution is adaptively chosen to maximize a hedger's loss. The generator can sample the full menu but cannot change the benchmark evaluation distributions. Freeze distributions after the one bounded development check in Section 11.

Generate 64 daily context observations and 20 daily trading intervals, dt=1/252 year. Fix S at the start of context to 1. Simulate one continuous path from the context through trading. Liability is an ATM call at inception: K_liability = S_0, maturity 20 trading days. The traded call has fixed strike 1.05 relative to the simulator's initial price unit and maturity 60 trading days after inception, so it remains alive at liquidation. Its context prices must refer to this same fixed contract, not a sequence of newly rolled ATM options.

Use a documented positivity preserving Heston discretization, initially full truncation for variance with log Euler spot and four substeps per daily interval. Make the internal random draws explicit and deterministic given keys. Check stability and weak consistency on the rented GPU against a finer step size, using small vectorized batches and synchronized timing. Feller failures in this menu are allowed but require a working nonnegative discretization; do not claim variance is strictly positive or that simulation is exact.

### Coherent traded option marks

Price the traded call from the same Heston parameters and current (S,v), under the stated pricing dynamics. Do not substitute Black Scholes with sqrt(v) and call that a coherent Heston option market.

Precompute a Heston call price lookup table for each of the 12 regimes on the first rented GPU, using vectorized JAX quadrature and bounded chunks. Do not build tables or run a pricing smoke test on the supervisor server. Exploit homogeneity: store C/K as a function of S/K, variance, and remaining maturity on the required daily grid. Interpolate during GPU simulation. A starting table is 96 log moneyness nodes and 48 variance nodes, with maturity nodes matching observations. Choose bounds from simulated tail coverage, validate interpolation on randomly sampled states against a direct float64 Fourier/quadrature pricer on the rented GPU, and expand only if needed. Generate in chunks to avoid large temporary arrays.

Record interpolation error, maximum sampled out of grid distance, and frequency of out of range states. Do not silently clip all out of grid prices. Expand the table once during setup or use a documented direct pricing fallback for rare cases. Pricing errors must be materially below the hedging differences being studied. Validate intrinsic/upper bounds, monotonicity in S, put call parity when relevant, and the xi approaching zero limit. Include a discounted martingale drift check with confidence/tolerance appropriate to simulation discretization and interpolation; finite differences need not be exact martingales.

If pricing setup becomes the bottleneck, delegate code review and optimize/chunk remote GPU table construction first. Share the resulting cache across workers. Do not launch CPU numerical work on the small server. Do not quietly change to an inconsistent option price process to save time. If no validated market can be implemented within setup limits, report that blocker rather than spend the remaining credit on invalid results.

## 5. History conditioned hedger and objective

Use JAX, Equinox, and Optax. No PyTorch. Start with a single layer GRU of hidden width 64 plus a small output MLP. This is a history conditioned / ICL style hedger: weights are frozen during evaluation, while hidden state evolves with observed prices. Do not claim that the comparison separately proves an ICL mechanism, since no context ablations are planned and option prices themselves reveal state information.

At each observed date supply causal features built from spot S and traded call C: log spot level relative to a fixed episode reference, spot log return, call price divided by the reference spot, call price change divided by the same reference, time features, and a context/trading indicator. During trading also supply liability moneyness/time, known instrument specifications, and previous holdings. Use a fixed feature schema and zero fill quantities that are unavailable during the pre trading context; never access future prices. A hedge call can approach zero, so do not use unprotected log call returns.

Warm the hidden state on the context with no trading. At trading time t, observe prices at t, choose holdings h_t, pay costs on changes from h_{t-1}, then earn increments from t to t+1. Reset hidden state between episodes. Hide regime index, Heston parameters, and latent variance. Use previous positions as inputs so transaction costs can be managed. Bound spot and option positions using scaled tanh outputs, initially absolute holdings at most 2 for each one unit liability. Freeze bounds across methods.

Use zero initial holdings and final liquidation of both hedges. Define gains G=sum_t h_t dot (Y_{t+1}-Y_t), where Y=(S,C). Costs are proportional notional costs at initial trade, every subsequent change, and liquidation. Initial default rates: 5 basis points for spot and 10 basis points for the call, applied identically in every arm. No price floors in the cost formula other than correct nonnegative option prices.

Fix the initial cash endowment p0 by a single public base convention, for example a Black Scholes ATM premium using the base mixture's mean long run variance and the liability maturity. This is a common benchmark endowment, not a claim about each regime's fair price. It uses no hidden episode parameters. Normalize terminal shortfall by inception spot:

    z_theta(X) = [payoff(X) - p0 - G_theta(X) + costs_theta(X)] / S_0
    ell(theta,X) = z_theta(X)^2
    L_D(theta) = mean_X_in_D ell(theta,X)

Primary training and target feedback objective is MSE of normalized terminal shortfall. All arms use it. This replaces the earlier entropic risk suggestion: exponential shortfall moments can be infinite in lognormal type models and empirical exponential losses can be unstable. Do not spend this pilot debugging that unrelated issue. Report signed PnL and upper shortfall tail metrics in addition to MSE; distinguish signed loss from absolute error.

## 6. Generator, keys, and independent actions

Use categorical regime probabilities with trainable logits a, initialized to log q0. To retain exploration use

    q_phi(j) = (1-epsilon) softmax(a)_j + epsilon q0(j), epsilon=0.05.

Use this same generator family and exploration floor in both DPG variants, both SOAR variants, and intrinsic self play. It is stochastic over regimes and over conditional trajectory noise. Do not learn the Brownian noise seeds. Gradients of log q must include the mixture floor correctly.

Crucial: sample a separate c_i independently for every trajectory, then sample X_i given c_i. Do not sample one regime for the whole batch in DPG. For normalized example weights, uniformly scaling all weights leaves student training unchanged, so summed weight attributions can cancel exactly; one shared generator action would then provide no useful attribution.

The policy score for trajectory i is grad_phi log q_phi(c_i), not the likelihood of its full stochastic path. Since the conditional simulator is fixed, the generator needs no path likelihood. Sampled regimes and trajectories are constants inside the teacher score loss. All teacher rewards/advantages are stop gradient when multiplying log probabilities. Do not accidentally add pathwise derivatives through the simulator.

Separate PRNG streams for initialization, regimes, simulator noise, target data, development evaluation, final testing, and teacher updates. Never reuse a target/test key in generated training data. Give all arms identical initial hedger weights and matched random number streams where meaningful; trajectories need not be identical when the regime policies differ.

## 7. Feedback, monitoring, selection and sealed test datasets

Create immutable independent episode sets and content hashes before meaningful adaptive training. Split role and domain explicitly. None is historical data.

| Role | Base episodes | Easy episodes | Hard episodes | Permitted use |
| --- | ---: | ---: | ---: | --- |
| Feedback F | None | 256 | 256 | Assigned teacher objective only |
| Monitor M | 512 | 512 | 512 | Save curves on every run; not student training |
| Selection V | 1024 | 1024 | 1024 | Sparse declared hyperparameter/checkpoint selection |
| Final test T | 8192 | 8192 | 8192 | Sealed until final configurations/checkpoints are frozen |

Use separate keys even when two roles share a distribution. Monitor and selection sets must not overlap feedback or test sets. Data storage is cheap relative to rerunning training; save episodes or exact reproducible keys plus immutable simulator/table versions. Evaluation is streamed in fixed batches.

Teacher access is:
- dpg_hard/soar_hard: F_hard only.
- dpg_mixed/soar_mixed: F_easy and F_hard, equal episode counts.
- selfplay: no target feedback.
- baseline/hard_reference: no generator.
All student gradient updates use newly generated training trajectories, except hard_reference which samples fresh Phard paths. Students never directly train on feedback/monitor/selection/test episodes.

### Fixed mixed objective and financial scales

Let L_j(theta) be raw MSE of normalized shortfall on domain j. Preserve raw losses always. For a balanced relative objective choose fixed positive scales s_easy and s_hard once, before tuning, as the zero-holdings strategy MSE on the corresponding feedback set, with a declared small numerical floor. These are not the current learner's losses. The exact scales and reference strategy are saved in objective_scales.json.

    F_hard(theta) = L_Fhard(theta) / s_hard
    F_mixed(theta) = 0.5 L_Feasy(theta)/s_easy + 0.5 L_Fhard(theta)/s_hard

This primary mixed objective equally weights loss relative to a fixed financial reference. It is not the same as an equal weight raw dollar MSE objective. Save the raw arithmetic average as an additional metric so the distinction remains visible. Do not tune mixture weights or update scales during training.

Use these same frozen scales for monitoring, selection and test summaries. A base reference scale s_base may be computed from the zero-holdings strategy on M_base for reporting only. Do not feed base reporting loss into the target teacher objective.

The mixed variants use twice as many distinct target feedback episodes as hard-only variants. Disclose this information difference. A feedback budget matched study is outside this pilot; do not imply the comparison isolates the effect of mixture choice at identical target information.

### Selection policy and information claims

Use the same primary selection score for all arm configurations: 0.5 L_Veasy/s_easy + 0.5 L_Vhard/s_hard, with base loss retained as a separate diagnostic. This matches the new broad easy/hard motivation. Hard-only refers to online teacher feedback, not complete ignorance of easy data: the experiment's hyperparameter selection observes easy development performance. Similarly selfplay has no target reward, but is not a claim of entirely target-blind model selection. State this explicitly.

No selector reads final tests or chooses its preferred easy/hard weighting after seeing outcomes. Preserve both domain components at every selection call, not only the aggregate. Fix the final checkpoint policy before final evaluation; do not report each method's separately best test checkpoint.

No repeat seeds. Use one master seed and fixed splits. Bootstrap test episodes only for conditional evaluation uncertainty, never to imply independent training seed robustness. If evaluation costs exceed projections, reduce dataset size once before reading method rankings, identically across arms, retaining at least a base/easy/hard monitoring pass for every trial.

## 8. Common online training protocol

Use one persistent learner per configuration run, initialized from scratch. All seven selected arms must have independent learner states. No baseline pretraining for adaptive arms. Carry hedger weights, Adam moments, and step count across all training blocks.

Starting shapes: batch size 256; K=4 minibatch learner steps per generator update. Each block has K*256 independently generated trajectories, reshaped into K batches. Use float32 for training and small explicit float64 GPU pricing/reference checks; consumer GPU float64 throughput is limited, so keep those checks bounded. No mixed precision until the entire implementation is correct; there is no budget for a precision study.

For each outer block:

1. Freeze the generator distribution used to sample all actions in that block.
2. Sample regimes and complete context/trading trajectories.
3. Perform K student updates on generated data.
4. Compute the method specific teacher feedback.
5. Update the generator once.
6. Keep the post block learner and optimizer state, with gradients detached across the block boundary.

Baseline and hard_reference run equivalent learner blocks without generator work. Compare primarily at equal cumulative active GPU time on matched hardware, including method overhead. Track learner steps, paths, device type and dollar cost too. Count compilation/setup in total expenditure but show steady training time separately. Stop by budget/deadline, not only by a nominal step count.

For DPG and SOAR add the declared common beta KL(q_phi || q0) penalty to the teacher minimization loss. It is differentiable analytically for this finite menu. Log the policy and KL terms separately. Fix whether the policy loss uses a sum or mean before tuning; do not vary that convention across hard/mixed variants. Reward normalization or rescaling must not silently change the effective relative KL coefficient. For selfplay, log its policy, prior penalty and expert iteration terms separately as well.

### DPG adaptation

Use weighted minibatch student MSE, with w_i initialized to 1:

    L_batch(theta,w) = sum_i w_i ell(theta,X_i) / sum_i w_i
    r_i = - d/dw_i F_target(theta_post(w)) evaluated at w=1.

For dpg_hard use F_target=F_hard; for dpg_mixed use F_target=F_mixed from Section 7. Differentiate both easy and hard components through the same student training window. Save component attributions r_i_easy and r_i_hard when inexpensive by separate VJPs, otherwise save both component losses and the exact combined attribution; mark component attribution unavailable instead of inventing it. Do not run two separate student training trajectories for the mixed reward.

Backpropagate through K actual Optax Adam steps, including its carried moments and count. Stop gradient the block's starting learner and optimizer state. Each example weight belongs only to the update in which that example is used. Verify that the update returned as the persistent learner is exactly the unweighted w=1 update.

Use a score function teacher loss proportional to -sum_i stopgrad(r_i) log q_phi(c_i). The canonical reward is the raw metagradient; do not silently replace it by gradient norms. Establish a fixed scaling convention for its sum/mean and normalize only using a previous block running scale if necessary; account for this in teacher learning rate. Avoid current batch baselines that introduce unintended dependence. Plain policy gradient is the clean pilot implementation; do not describe it as a literal GRPO reproduction.

Important failure check: one step Adam can suppress uniform gradient magnitude changes, and normalized weights imply sum_i r_i approximately zero. Neither implies that all individual attributions should vanish. Validate selected components by finite differences on a tiny deterministic window. If Adam attributions vanish numerically, first inspect stop gradients and carried state; do not replace the method without documenting the reason.

### SOAR adaptation

Measure the assigned objective before and after the block on exactly the same feedback episodes:

    R = F_target(theta_pre) - F_target(theta_post)

For soar_hard use F_target=F_hard; for soar_mixed use F_target=F_mixed. Save easy and hard raw pre/post losses and component improvements separately in mixed runs. The same student update receives one combined reward, not two independent student runs.

For independently sampled block actions c_1,...,c_N, the score is sum_i grad log q_phi(c_i). Use teacher loss

    - stopgrad(R-b_pre) sum_i log q_phi(c_i),

up to an explicitly documented fixed scaling absorbed in the learning rate. b_pre is an EMA reward baseline from previous blocks, updated after computing this teacher gradient. One scalar reward shared by many actions can be noisy; that is a property to measure, not a reason to grant extra hidden compute. Log advantage scale and actual distribution movement.

No student reset and no parallel teacher candidate branches. This is an online SOAR style risk improvement estimator, not original SOAR RLOO. Do not call MSE improvement the original paper's literal accuracy reward.

### Intrinsic self play adaptation

At outer round e, use the current pre block learner theta_e and historical checkpoint theta_floor(e/2). For each synthetic example use squared shortfall ell_i and

    delta_e = theta_floor(e/2) - theta_e
    P_e = learner_lr / (sqrt(vhat_e) + adam_epsilon)
    r_i = abs( grad_theta ell_i(theta_e) dot [P_e * delta_e] ).

Flattening is conceptual; implement PyTree dot products or a JVP. Compute all example directional derivatives with a single JVP of the batched loss vector along P_e*delta_e, without materializing an example-by-parameter gradient matrix. Use the actual learner Adam second moment with bias correction. At e=0 define reward zero; q stays at q0 until a usable historical movement exists. Exclude integer state and nonparameter leaves from P_e.

The reward is absolute alignment, not signed improvement and not a finite loss difference. Self play uses neither F_easy nor F_hard. Use normalized rewards, KL to q0, and a reward weighted log probability term for generator expert iteration, following the paper's mechanism at the regime action level. In this finite menu, replay items are regime IDs and metadata rather than executable programs. Use newly simulated paths for ongoing reward estimates. Log and stop gradient the rewards and normalization. Record every simplification relative to the paper, especially the absence of program mutation and off policy program replay. Do not add an unrelated ALP reward.

Save detached learner parameter snapshots indexed by outer round so floor(e/2) is available exactly. Store them outside the autodiff graph, using the rented worker's host/disk storage for old states. Do not accumulate model snapshot banks in supervisor RAM. The main resumable checkpoint must retain or reference the historical snapshot bank. Cap the planned number of blocks from measured throughput and budget so snapshot I/O/storage remain modest. Snapshot compression or retaining only still-needed indices is allowed if exact lookback retrieval is preserved. No graph through past snapshots.

## 9. Engineering and JAX requirements

Use Equinox Modules for the hedger and appropriate PyTrees for state. Use eqx.filter_jit, filtered gradients, jax.vmap over episodes, and lax.scan over both context and trading time. Scan a short block of student updates for DPG where practical. Use JVP for intrinsic rewards. Avoid Python loops over trajectories or per episode option pricing during training.

Define pure functions with explicit inputs and outputs:

    sample_regimes(generator_state, key, batch_shape) -> ids, log_probs
    simulate(regime_ids, key, market_config, price_tables) -> episode_batch
    rollout(hedger, episode_batch, task_config) -> shortfall, diagnostics
    student_block(hedger, optimizer_state, batch, weights) -> updated_state, losses
    teacher_step(method, generator_state, student_pre, student_post, batch, auxiliary) -> generator_state, metrics
    evaluate(hedger, dataset, config) -> aggregate_metrics

Keep batch sizes, K, feature dimensions and architecture static. Pass learning rates and scalar coefficients as array inputs when appropriate to avoid recompilation. Do not recreate equivalent jitted functions inside loops. Enable a persistent compilation cache in a private project directory. Measure compile time separately and time kernels only after block_until_ready; asynchronous launch timing is not runtime. Count all compilation time toward dollar consumption.

Pin a working JAX/jaxlib/Equinox/Optax stack compatible with the rented GPU. Use current official installation instructions at execution time; do not blindly upgrade repeatedly. Save package versions, GPU/driver/CUDA information, and a dependency lock/export. Make dependencies runnable from a fresh clone. Set modest CPU thread counts when necessary and keep one training process per GPU. Run independent queued jobs on separate GPUs and use GPU-specific compilation cache directories or a safely managed compatible cache. Use a lean local logger, not a mandatory external tracking service.

## 10. Required checks before meaningful GPU expenditure

Run all small numerical tests and the integrated smoke test on the first rented GPU. Start by asserting a GPU JAX backend; use tiny fixed batches for cheap correctness checks. There is no CPU smoke-test phase, and the supervisor must not execute these numerical tests:

1. A constant holding on a two date path gives the analytically expected gain.
2. Zero positions give zero gains and zero transaction costs.
3. Initial trade, rebalancing, and liquidation costs are counted exactly once.
4. Changing future observations cannot affect earlier actions.
5. Hidden state resets between episodes; context itself does not trade.
6. Target/test PRNG streams and episode IDs never appear in student training.
7. q probabilities sum to one, include the exploration floor, and their log gradients match finite differences on logits.
8. Option pricing and simulation pass the checks in Section 4.
9. DPG weight derivatives match finite differences on a tiny K step problem, including optimizer state.
10. SOAR reward is zero for an intentionally zero learner learning rate and its sign agrees with target loss changes.
11. Self play JVP rewards match explicit small model gradient dot products.
12. Checkpoint resume reproduces the next training update and PRNG sequence to numerical tolerance.
13. Budget watchdog refuses an over budget job and can gracefully checkpoint on a termination signal.

These tests protect scientific validity and money. Do not build a large generic testing framework. Save commands/results in validation.md. Fix failures before a sweep.

## 11. Timing and distribution validity gate

Rent the first GPU early for preparation and numerical checks; do not rent the full worker pool before an integrated GPU smoke test passes. Compile/profile one worker, then launch only ready independent jobs on additional matched workers. Record compile seconds, milliseconds per learner block, peak VRAM, feedback evaluations, teacher overhead, simulated/evaluated paths per second, and checkpoint overhead for each method family. Synchronize timed JAX work.

Use measured rates to reserve wall time and aggregate dollars for all seven arms, all monitoring curves and final test passes. Choose a common affordable active runtime for selected runs; mixed feedback and DPG overhead reduce learner updates within that time rather than being silently free. Do not launch tuning candidates whose completion would jeopardize a valid run for every arm.

Train short baseline and hard_reference runs using the same learner settings. Starting Optax Adam learning rate is 1e-3 with global gradient clipping norm 1; use 3e-4 only if the default is unstable or unproductive. Keep architecture and other learner settings shared. Evaluate base, easy and hard monitor/selection metrics from initialization onward.

Required gate:
- Market prices and causal accounting are valid.
- Student losses and teacher signals are finite and nontrivial.
- The three distributions differ as declared; easy is not simply a relabeling of base.
- Both reference learners can learn without hidden simulator state.
- Differences are not dominated by numerical pricing errors or accidental changes in payoff/cost units.

A hard target baseline/reference gap is diagnostically useful, but no longer an acceptance requirement. Do not force a ten percent gap or discard the campaign merely because the base learner transfers well. The new experiment also asks about easier shifted markets and broad utility. Retain a weak or absent gap as a result. One bounded pre-final correction to an obviously uninformative distribution definition is allowed using development diagnostics, within time; never optimize target severity against final test performance. Regenerate/hash all affected data and objective scales after a definition change.

Do not train additional easy oracles. Use zero-holdings loss and declared process properties to contextualize scale and the easy label. Without a separately optimized easy reference, do not claim to know the irreducible hedging error of that market.

## 12. Narrow tuning and seven arm final comparison

Use one master seed per configuration. No independent seed repeats. Resuming a screened candidate is its existing lineage, not a new trial. Preserve every rejected and failed configuration.

First freeze shared hedger architecture, learner optimizer, batch size, K and accounting after the baseline/reference diagnostic. Default K=4. A global fallback K=1 is allowed only for feasibility after checking DPG signal. No separate window length sweeps.

Generator Adam learning rate starts with at most two values, 1e-3 and 1e-2 after documented reward scaling. Start KL=1e-2 and exploration floor=0.05, intrinsic expert iteration coefficient=1. These are pilot choices, not paper defaults. Allow one corrective setting change for numerical failure or immediate collapse, not an unrestricted sweep.

Minimum queue before long extensions:
- One baseline and one hard_reference diagnostic, reusable as final lineages.
- One short valid configuration for each of dpg_hard, dpg_mixed, soar_hard, soar_mixed and selfplay.
- A base/easy/hard monitoring record and resumable checkpoint for each.

Then screen the second generator learning rate only if time and total budget allow. Share implementation and compiled shapes between hard/mixed variants; their learner states remain independent. Use equal short time caps across adaptive candidates and selection by the same frozen easy/hard score. Do not manually favor a candidate after looking at sealed tests.

With three workers, an example work conserving queue is:
- Worker A: baseline/reference diagnostic and then available short trials.
- Worker B after validation: DPG hard/mixed candidates.
- Worker C after validation: SOAR hard/mixed and selfplay candidates.
Rebalance after each job; no idle dedicated method worker. The scheduler owns job leases so two agents cannot accidentally run the same trial twice. A fourth worker may clear a measured backlog if both hard limits permit.

Screen for approximately 10–15 active minutes, promote for 30–45 minutes, then extend one selected lineage per arm to the common final active runtime (provisional 45–75 minutes). Profile-based budgets override these provisional values. If a time cap is too short for any learning, reduce tuning breadth and extend all selected arms rather than produce meaningless five minute rankings.

Mixed selection uses normalized equal easy/hard loss, with all components logged. Base retention is reported, not an extra teacher reward. Freeze a final manifest with selected config IDs, checkpoint policy, target definitions, objective scales and continuation caps before opening tests.

Compare cumulative active compute at common checkpoints where possible; include lineage work already spent on a winning configuration. Track tuning cost separately for each arm and campaign totals. If different GPU models must be used, do not equate their GPU-hours: retain cost, device, updates and time, and qualify comparisons.

Final test evaluation runs after selection is frozen, on selected checkpoints and also retained trial checkpoints for the all-runs metric archive. Launch it from a non-selecting reporting stage; never feed these archived test values back into the queue, early stopping or hyperparameter choice. Ensure remaining time covers every retained checkpoint. If an objective failure yields no valid checkpoint, archive status and null test metrics with a reason rather than deleting the trial.

No additional scientific arms, repeats or ablations. Stop and export on either deadline or budget cap.

## 13. Preserve all metrics for every run

Metric retention is mandatory for every arm, hyperparameter configuration, smoke/pilot training run, promoted lineage, stopped trial and failed trial. Never retain only the winning run or overwrite earlier trial directories. Keep method-specific rewards alongside comparable base/easy/hard outcomes.

Write append-only train_metrics.jsonl, evaluation_metrics.jsonl and a combined CSV/Parquet export. A training record does not need to evaluate three datasets every step: separate event types and preserve the most recent evaluation timestamp. Never fill a missing fresh measurement with a stale number while pretending it was recomputed.

Every event contains:
- campaign/run/trial/parent/checkpoint IDs, method and target mode (none, hard, mixed, intrinsic);
- config/benchmark/dataset/pricing-table hashes, code commit, master seed;
- worker, GPU model, device, instance, timestamps, campaign elapsed seconds;
- learner steps, outer rounds, synthetic path counts, feedback episode evaluation counts;
- cumulative active training, compile, teacher, feedback, monitoring, selection, checkpoint and paid elapsed times;
- instance/campaign dollar estimates and remaining budget/deadline.

### Common evaluation schema

For every evaluated domain (base, easy, hard) and split role (monitor, selection, final_test), save:
- raw normalized shortfall MSE and reference-scaled MSE;
- mean signed shortfall, mean signed PnL, shortfall standard deviation and absolute error;
- empirical shortfall VaR95 and ES95, sample count and finite count;
- spot/option turnover, total and per instrument costs, liquidation costs;
- average absolute positions, saturation fraction, nonfinite count;
- data hash and checkpoint ID used.

Also save easy/hard raw mean and fixed-scale balanced objective, and a base/easy/hard reporting aggregate if desired with its weights named. Never collapse them into a single best loss. Negative signed PnL and positive shortfall conventions must be explicit.

For all training events save synthetic loss, learner gradient norm, clipping and nonfinite diagnostics. For every adaptive run save all 12 generator probabilities, entropy, KL to q0, easy-regime mass and hard-regime mass, teacher gradient norm, reward statistics and actual probability movement. Baseline/reference use their fixed sampling probabilities; teacher-only fields are null with a not_applicable marker.

DPG: raw component target losses, combined objective, attribution norm/sum/sign fractions and attribution scaling.
SOAR: raw pre/post component losses, component improvements, combined reward, previous EMA baseline and advantage.
Selfplay: raw alignment reward statistics, lookback index, parameter movement norm, preconditioner summaries, normalized reward and expert iteration term. No target teacher losses should be fabricated for intrinsic updates.
All runs: pricing out-of-range counters, numerical failures and simulator diagnostics.

### Frequency without wasting the eight hours

- Training scalar records every approximately 50 learner steps or 30–60 seconds, using batched device transfers.
- Mandatory monitor evaluations on all three domains at initialization, at fixed 2–5 minute active-time intervals selected from profiling, at promotions, and at run termination.
- Selection evaluations at short trial termination and sparse fixed continuation checkpoints.
- Final tests only in the post-selection reporting stage.
- If monitoring exceeds about five percent of active run time, increase its interval or reduce the common monitor size once. Do not drop easy or hard curves. Save the choice and sample counts.
- Every early-stopped candidate gets a terminal three-domain monitor pass from its last valid checkpoint.
- On failure preserve the last valid metrics/checkpoint and failure timestamp, exception, nonfinite counts, budget and status. If evaluation is impossible, write explicit null values and reason. Do not hide the run.

Retain per-episode monitor loss arrays at evaluation events where feasible, and per-episode final test shortfall, PnL, costs and turnover for every retained evaluated checkpoint. Compressed arrays make later comparisons possible without rerunning a model. A summary metrics file is not a replacement for these final raw outcomes.

Run a lightweight metadata-only completeness audit before shutdown/reporting; directory/JSON checks may run on the supervisor, but array integrity checks and numerical reductions must run on a rented worker before it shuts down. Verify each run_id has config, status, training events, base/easy/hard monitor events, terminal/latest checkpoint reference, and either final three-domain results or a documented failure. The expected final comparison has seven selected arms; the archive also includes all non-selected trials. Saving metrics on failed/tuning runs is not optional.

Use identical evaluation paths across methods. Optional bootstrap episode differences must run on a rented worker, with bounded arrays and saved results, not on the supervisor; skip this optional computation if time is tight, and do not imply independent training seed uncertainty. The single seed limitation remains.

## 14. Checkpoints and recovery

Use Equinox tree serialization or a comparably transparent PyTree serialization format, plus explicit JSON configuration metadata. Save full state, not only model weights:

- Hedger parameters and static architecture specification.
- Learner Optax state, Adam count, scheduler state if any.
- Generator parameters and optimizer state.
- All PRNG keys/counters.
- Outer round, learner step, and data/market/pricing table hashes.
- Reward baselines, running scales, and teacher auxiliary state.
- Exact self play lookback snapshot bank or durable references to it.
- Cumulative runtime, per worker/campaign cost ledger position, deadline, and trial lineage.
- Feedback target mode, fixed easy/hard scales, and exact feedback/monitor/selection/test manifests.
- Latest base/easy/hard evaluation event IDs and associated checkpoint hashes.

Atomic write: write a temporary checkpoint, flush, and rename. Keep latest, previous valid latest, best selection checkpoint, and final. Save at least every five minutes, at stage changes, on graceful stop, and on SIGTERM when possible. Do not serialize full training computation graphs or static price tables repeatedly. Checkpoints must reference the correct cached table and data files by content hash.

Resume with the same dependency environment and rebuild a model skeleton before deserializing Equinox leaves. Verify one actual resume during setup. Resume must not reset Adam state or replay target/test random streams. Copy checkpoints off every rented machine at each major phase, not only at the end. Use per-run immutable names and atomic publication to avoid parallel workers overwriting each other. Keep trial metrics for rejected configurations; reclaim bulky caches rather than scientific outputs. Preserve the code commit and config used by every saved state.

## 15. Repository layout and execution interface

Suggested layout:

    README.md
    SUPERVISOR_STATE.md
    TASKS.md
    DECISIONS.md
    validation.md
    pyproject.toml
    dependency-lock-or-export.txt
    configs/base.yaml
    configs/methods/{baseline,hard_reference,dpg_hard,dpg_mixed,soar_hard,soar_mixed,selfplay}.yaml
    src/{market,pricing,hedger,objective,generator,teachers,runner,checkpoint,metrics}.py
    scripts/{prepare_market,smoke,profile,run_campaign,evaluate,make_report}.py
    tests/
    data/{feedback_easy,feedback_hard,monitor_base,monitor_easy,monitor_hard,selection_base,selection_easy,selection_hard,test_base,test_easy,test_hard}/
    cache/pricing/
    cache/jax/
    runs/<run_id>/{config.json,train_metrics.jsonl,evaluation_metrics.jsonl,status.json,episode_metrics,checkpoints,stdout.log}
    budget/ledger.jsonl
    queue/state.json
    objective_scales.json
    results/{final_metrics.csv,episode_losses.npz,figures,report.md}

A single campaign command should run the validated multi GPU queue and enforce the global dollar and eight hour wall-clock limits. Provide resume and dry run modes. Do not hardcode secrets, SSH keys, account tokens or machine IDs in source. Make dataset roles explicit in function arguments and filenames. The supervisor should be able to inspect status and remaining cost without reading verbose logs.

## 16. Completion criteria and final interpretation

Finish within eight hours when seven affordable selected runs have completed, all retained valid trial checkpoints have their required evaluations, metrics/checkpoints are exported, completeness checks pass, and every paid worker is released. If a hard limit or validity blocker prevents full completion, save an honest partial campaign with explicit missing arms and reasons. Do not silently change the experiment to three winners.

Produce:
1. Working JAX/Equinox/Optax code, frozen configs and dependency export.
2. Full trial manifest including rejected, interrupted and failed configurations.
3. Train/evaluation metrics and raw final episode arrays for every retained run.
4. Full resumable learner/generator/optimizer/key state, including selfplay historical snapshots.
5. All dataset/scales/pricing definitions and hashes plus numerical validation.
6. Per-worker costs, aggregate dollar ledger, actual campaign duration and confirmed shutdown status.
7. A selected-arm result table with base/easy/hard raw and scaled MSE, balanced easy/hard objective, tail outcomes, costs, steps and active compute.
8. Easy and hard monitor/selection curves versus active compute, base retention curves, generator probability trajectories, and an easy-versus-hard final outcome plot. Final test curves are not used for selection.
9. An all-trials table with the same domain metrics, statuses, runtime/cost and configuration lineage.

Primary interpretation is broader useful learning:
- Compare hard-only versus mixed within DPG and within SOAR.
- Determine whether easy loss falls, rises or remains unchanged when hard loss improves.
- Compare intrinsic curriculum with target-guided generation.
- Assess whether gains justify generator overhead and whether baseline transfer already works well.
- Report base retention independently.
Do not reduce the conclusion to worst-case robustness, average only the favorable distribution, or claim all distribution shifts are covered.

Report improvement over baseline separately in every domain:
    Delta_j(method) = L_j(baseline) - L_j(method), j in {base,easy,hard}.
Show raw losses too. A lower raw easy loss than hard loss mostly describes different tasks, not relative algorithm quality.

Optional hard reference gap closure remains secondary:
    closure_hard = [L_hard(baseline)-L_hard(method)] /
                   [L_hard(baseline)-L_hard(hard_reference)].
Use only with a positive nontrivial denominator and never clip. There is no easy oracle in this budget. Hard_reference can lose on easy or base and is not an overall bound.

State one seed, narrow tuning, common-support distribution shifts, finite generator menu, numerical pricing approximations and online paper adaptations. Hard-only means no easy online teacher reward; common easy/hard hyperparameter selection still uses easy development information. Mixed arms use a larger target feedback pool. Intrinsic selfplay receives no target reward but is development-selected. No repeated-seed confidence claim is allowed.

The correct conclusion can be negative, inconclusive, or a clear specialization tradeoff. Preserve unfavorable and failed trials. Do not spend extra funds or continue past the deadline to obtain a preferred outcome.

## 17. References and implementation documentation

Read the relevant methods and current official APIs before implementation. Do not copy original language model pipelines wholesale.

- DPG: Synthetic Data for any Differentiable Target, https://arxiv.org/abs/2604.08423 . Metagradients of example weights supply generator policy rewards; the approximation theory does not establish exactness for every optimizer or persistent training setup.
- SOAR: Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability, https://arxiv.org/abs/2601.18778 . Grounded teacher rewards use student improvement on hard problems; this pilot replaces accuracy with hedging MSE and removes reset/branch structure.
- Intrinsic self play: Self-Play Pretraining with Zero Data, https://arxiv.org/abs/2609.30063 . Use its absolute AdamW preconditioned alignment and lookback definition; stochastic process regimes replace programs.
- Deep Hedging, https://arxiv.org/abs/1802.03042 . Background for trading policy optimization with frictions; the pilot uses squared shortfall rather than asserting fidelity to a particular paper risk measure.
- JAX benchmarking, https://docs.jax.dev/en/latest/benchmarking.html . Synchronize timings and distinguish compilation from execution; follow redirects to current official documentation.
- JAX persistent cache, https://docs.jax.dev/en/latest/persistent_compilation_cache.html . Keep cache private and reuse compatible executables.
- Equinox transformations, https://docs.kidger.site/equinox/api/transformations/ . Filtered JIT/grad and PyTree transformations.
- Equinox serialization, https://docs.kidger.site/equinox/api/serialisation/ . Save array leaves together with enough configuration to rebuild the skeleton and restore optimizers.

## 18. Supervisor opening instruction

Start fresh and implement this complete revised plan, not the superseded hard-only robustness plan. The scientific goal is useful synthetic curricula across easy and hard market shifts. Run exactly seven arms: baseline, hard_reference, dpg_hard, soar_hard, selfplay, dpg_mixed, soar_mixed. Mixed uses fixed balanced easy-plus-hard target feedback. Keep base, easy and hard losses/financial metrics for every run, including all tuning and stopped trials.

Use JAX, Equinox and Optax. The small supervisor is only a control plane: no CPU smoke tests or numerical workloads. Start the first rented GPU early; run pricing preparation, all numerical tests, simulation, compilation, training, evaluation and substantial aggregation on GPU workers, with explicit GPU backend checks. Delegate implementation, numerical checks, runner and report work extensively to agents with disjoint ownership. Keep the main supervisor context focused on integration, science, the central job queue, deadline and money; use durable state files so context compaction is not a restart. Proceed through implementation, checks, short tuning, selected runs, all-run evaluation, artifact export and confirmed cloud shutdown without routine user questions.

Multiple GPUs may run independent jobs concurrently, but one central supervisor owns all rentals and budget reservations. Enforce USD 25 total across all workers and eight hours wall time, with checkpoint/export margin. One seed only, no repeat seeds or ablation suite. Preserve complete metrics, per-episode results, configs and resumable checkpoints continuously. If credentials or infrastructure access are absent, finish source/configuration and lightweight orchestration preparation only and identify the exact blocker; do not substitute supervisor CPU runs. Never claim unexecuted results.

