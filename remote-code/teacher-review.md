# Teacher mechanism review

Reviewed 2026-10-17 against the execution brief. Source-only checks completed on the supervisor; GPU numerical validation remains required. `python -m py_compile src/generator.py src/teachers.py tests/test_teachers.py` passed; this does not establish numerical correctness.

## Reference findings

- [DPG, section 3](https://arxiv.org/html/2604.08423v1): example-weight metagradients become generator rewards. Its formal approximation result assumes SGD and regularity conditions; it does not prove exact persistent-Adam policy gradients. This adaptation uses normalized weighted MSE, negative target-loss derivatives, carried Adam moments/count, one persistent learner trajectory, raw rewards and a summed categorical score. It omits GRPO and learner resets. The same mixed objective is differentiated through one window; component losses are retained, separate component attributions are explicitly unavailable.
- [SOAR, section 3](https://arxiv.org/html/2601.18778v1): teacher reward measures downstream student progress on hard grounding data. The paper uses RLOO branches and promotion/reset logic. Here a persistent learner takes one block; MSE decrease on identical pre/post feedback episodes supplies a scalar reward. Its advantage subtracts the previous EMA, then updates that EMA. Hard/mixed variants use the same summed action score and exact prior KL.
- [Self-play, section 2.2 equations 2–5](https://arxiv.org/html/2609.30063v1): absolute alignment uses the halfway historical checkpoint and bias-corrected AdamW diagonal step operator. Forward differentiation avoids per-example gradient storage. Batch standardized policy advantages and positive-reward sum-normalized expert iteration match the declared mechanism. Here regimes replace programs, prior KL is evaluated exactly over the finite menu, and fresh paths replace mutation/replay pools. No off-policy ratios are needed for fresh samples. No target feedback enters the intrinsic API.
- [Equinox transformations](https://docs.kidger.site/equinox/api/transformations/): filtered JVP handles static leaves and symbolic zero tangents. Filtered value-and-grad is used for DPG auxiliary learner trees, including callable leaves in Equinox modules.

## Interfaces and runner obligations

`dpg_step` returns `(generator, student_post, block_losses, metrics)`. Its block callback accepts `(student_state, batch, weights)` and returns `((model, optimizer_state), losses)`. Use the actual learner block callback, not a substitute optimizer. Starting learner/state and examples are detached; the returned post-state is the same all-one weighted update.

`make_target_objective(mode, loss_fn, feedback_hard, scales, feedback_easy=None)` returns scalar objective and raw/scaled component callbacks. Hard mode rejects easy data. `soar_objective_step` takes pre/post states, this objective, sampled IDs and previous EMA. Pass component callback to both target-directed methods.

`selfplay_step` takes the pre-block learner and its actual optimizer state, historical model at index `outer_round//2`, newly generated flat batch and IDs, actual learner LR, and outer round. Its loss callback must return one scalar per sampled action. Save detached snapshots before each round and preserve them in checkpoints. Round zero or zero historical movement preserves generator parameters AND optimizer momentum. Learner training still proceeds. The reward preconditioner follows the brief's explicit formula (no `eps_root` term), while the learner's tiny positive `eps_root` stabilizes DPG higher derivatives.

Generator starts exactly at q0 via logits `log(q0)` and mixes 5% q0 into its softmax, guaranteeing full support. DPG/SOAR score reduction is sum, with unscaled DPG attribution; self-play uses mean standardized policy advantages and sum-normalized EI. Keep these conventions fixed through tuning. Teacher metric logs now include attribution sign fractions/scaling, self-play normalized reward moments and whether its update was applied.

## GPU checks to run

Run `pytest -q tests/test_teachers.py` on a rented GPU only. The module fails if the default backend is not GPU. Checks cover categorical floor/score finite differences, carried clipped-Adam multistep DPG finite differences and all post-state leaves, per-minibatch zero attribution sums, SOAR reward sign/previous baseline, explicit self-play gradient-dot equivalence, static/integer parameter filtering, zero-movement momentum preservation, fixed mixed normalization, hard feedback isolation, and reward stop-gradient.

Remaining integration validation: actual recurrent learner DPG block under filtered JIT; mixed component accounting; snapshot indexing/checkpoint resume; worker-side numerical outcomes. No tests or model computations were executed on the supervisor.
