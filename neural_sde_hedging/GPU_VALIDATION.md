# Required GPU verification before a scientific training run

Status: **GPU smoke passed 2026-10-19**: eleven automated cases plus two-round,
126-day DPG/SOAR/self-play runs and separate-process checkpoint loading. Evidence:
[smoke report](../gpu-smoke-20261019/README.md). The broader checks below are a
scientific acceptance checklist; a smoke pass does not establish every item.
No numerical computation may run locally or on Hetzner's one-vCPU control plane.
Further checks require an authorized GPU worker, declared budget, detached
execution and exported evidence.

Record source hashes, package versions, device, float precision, complete configuration, random keys, measured tolerances and actual outputs. Do not substitute a passing smoke test for the checks below.

## Neural-SDE law and likelihood

- Check one-step transition log density against an independently written multivariate Gaussian calculation, including initial distribution, drift times dt, covariance times dt, inverse/log determinant and joint latent-observed ordering.
- For paths sampled under a frozen teacher, compare simulator-recorded density to rescoring that exact path. Include non-diagonal diffusion, latent factors, smallest admissible diffusion and long horizons.
- Verify current versus proposal path likelihood ratios on small analytic Gaussian SDE cases. Demonstrate support compatibility. Frozen parameter snapshots must define the same initial law/discretization and full-rank transitions.
- At identical teachers ratios must be one. Changing latent factor parameters must change both generated latent paths and observable conditional dynamics. Confirm that no latent coordinates or teacher labels reach hedger observations.
- Verify the chosen discretization, truncation and payoff cap define the intended finite entropic population objective. A finite Monte Carlo estimate is insufficient evidence of population integrability.

## Differentiable learner and DPG

- Instantiate actual Equinox hedger partitions and Optax states. Run the full callback graph with array/None parameter PyTrees, integer Adam count, recurrent state and nested tuples. Both eager GPU and top-level JIT results must agree.
- At every attribution evaluation compare all-one weighted risk and resulting parameter/optimizer trajectory to the ordinary inner learner. Include clipping and several carried Adam updates. Confirm the incoming state is detached and the initial state is bitwise unchanged afterward.
- Independently check selected positive example-weight derivatives by central finite differences through the complete learner. Compare risk sign, scale and directional derivatives. Check nonzero metagradients through Adam moments, not only through the final model.
- Include the zero-initial-moment case and near-zero gradients. Learner Adam must have positive eps_root for differentiable square-root moments; self-play preconditioning must use the same epsilon conventions.
- Verify repeated DPG outer calls restart at the declared fixed model **and optimizer**, not the previous adapted student. Cross-group batching must preserve correct example indexing; separately batched groups must each reset independently.
- Verify GRPO candidate grouping is across group axis for each common prompt/context, and group normalization has the intended result for constant rewards and unequal rewards. Check signed negative attributions remain negative before normalization.
- With one on-policy update, old/current ratio equals one and PPO clipping is initially inactive. This is a valid first gradient, **not evidence that clipping stabilizes the run**. If multiple optimization epochs are desired, freeze rewards/behavior likelihoods and test the actual repeated-epoch loop and clipped ratios. The present runner's single update must be described accurately.
- KL options require a separate numerical gradient check against an analytic continuous Gaussian KL. A language-model GRPO sampled KL surrogate must not be described as the exact continuous-path KL gradient. Default beta zero does not establish regularization.

## SOAR

- Verify every group/repeat starts with the same baseline arrays, moments and count. Verify the same held-out feedback paths are used for every branch and baseline evaluation, with no teacher-sampled evaluation replacement.
- Repeats need meaningful learner randomness: permuting a complete full batch before a permutation-invariant reduction does not provide four independent training outcomes. Use the declared bootstrap/minibatch training procedure, then check distinct keys produce distinct batches and trajectories. All repeats retain the same initialization.
- Check rewards equal baseline target risk minus post-branch target risk. Average repeats before constructing dataset RLOO advantages; advantages should sum to zero over datasets. Every row receives its own dataset's advantage.
- Test no-promotion, promotion, reward ties and even-number repeat medians. Failed branches must never modify baseline. On promotion, choose best mean-reward dataset and a declared median-ranked repeat; archive the actual winning dataset. Check smoothing state and risk-unit threshold against the recorded protocol.
- Optional KL uses a sampling-law score correction and is locally on-policy. Verify its gradient analytically; stale-path reuse would require an additional specified estimator.

## Self-play, replay, mutations and reuse

- At zero movement, alignment reward must be zero. For a small hand-computable learner compare forward-mode JVP to explicit gradient dot Adam-preconditioned historical movement. Confirm halfway historical checkpoint indexing and optimizer bias correction, including count zero and actual eps_root.
- Verify the entropic adaptation computes B*softmax(gamma*raw_loss)*JVP(raw_loss). It must not differentiate the softmax weights as part of the per-example JVP. Compare the sum/mean relationship with the gradient of empirical entropic risk.
- Check all three pool sources train the learner and enter expert iteration. Mutation rows must contribute **zero** policy-gradient term while contributing their valid expert-iteration term. Normalization must use the full pool.
- Replayed generator snapshots must be executed with fresh innovations; recompute current reward and original-snapshot path density on that new path. Never substitute an old path density or stored archive reward for a newly executed path.
- Verify replay sampling is uniform without replacement from eligible earlier non-replay records, including empty/undersized banks and repeated program identities. Check mutation parents are uniform over occupied niches rather than all entries.
- Verify archive admission requires positive reward, per-niche top-eight retention, .97 per-round decay, and the declared volatility-persistence/amplitude niche mapping. These financial descriptors and parameter mutations are adaptations of the paper's program-space archive.
- Check stopped sequence importance ratios clip log ratios to [-20,20], fresh ratios equal one, prior penalty appears in stopped policy advantage, and positive expert weights sum to one unless all rewards are zero. All-zero pools must stay finite and preserve a declared prior-only policy update convention.
- Confirm replay likelihoods score joint observed/latent paths consistently. Importance correction is componentwise over each stored full-support proposal; its correctness does not mean finite-sample variance is small. Log-ratio clipping introduces bias; report the fraction clipped.

## Restart, fairness, and final evaluation

- Interrupt and resume each method mid-run on the GPU. Compare to uninterrupted execution: RNG, student/teacher parameters, all Adam states, baseline and smoothing state, promotion archive, replay records/elites/behavior snapshots, learner history and round counts must match within declared determinism tolerances.
- Exercise atomic-write interruption and stale/incomplete temporary files; verify latest committed state survives and missing historical checkpoints cause an explicit failure, not reset.
- Check configuration/source/feedback hashes reject incompatible resume. Verify source hashing covers the actual loaded files and persisted immutable prior. Confirm storage bound/pruning preserves every future required halfway snapshot and proposal density.
- Demonstrate deadlines, terminal status, disconnect robustness, log flushing, artifact export and worker shutdown. A status file saying complete merely because the time limit expired must not imply convergence or benchmark success.
- Compare methods using identical hedger architecture, initial weights, contract, observations, training objective and frozen evaluation data. Report method-specific extra learner/teacher computations, feedback exposure, reuse and wall-clock budget; no multi-initialization campaign is authorized.
- Run sealed evaluation only after protocol and specialist gates are frozen. Report CVaR, entropic risk, daily hedge-error SD with consistent conventions; keep development selection separate from final results.
