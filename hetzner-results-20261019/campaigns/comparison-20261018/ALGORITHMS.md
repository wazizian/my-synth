# Persistent adaptive comparison

Recovered from root `PLAN.md` sections 7–8, `src/teachers.py`,
`src/generator.py`, and the entropic extension in
`campaigns/slv-20261018/teacher_fragment.py`. This campaign's reusable implementation
is `adaptive.py`; each market adapter supplies simulations and terminal losses.
No numerical validation has been performed on Hetzner. Numerical auditing and
profiling must occur on GPU workers before production training.

## Shared learner and generator

Each learner persists its model, clipped-Adam moments, step counter and random
state across blocks. A block contains K sequential minibatches with B paths per
minibatch. Inputs are pytrees with leading shape `[K,B,...]`. The market-specific
callback `losses_fn(model,batch)` returns B terminal losses. Every method uses the
same empirical entropic learner objective, `logmeanexp(gamma*loss)/gamma`, with
the campaign's fixed gamma. Intermediate gradients and optimizer moments are
never reset for teacher evaluation.

The generator is categorical over the fixed easy/base/hard scenario menu. Its
positive reference prior is frozen separately from the pure-base baseline.
The root controller selected uniform `[1/3,1/3,1/3]` and exploration epsilon .05.
`q=(1-epsilon)*softmax(logits)+epsilon*prior`. This deliberately changes scenario
coverage relative to the baseline, which never sees easy or hard. Adaptive gains
therefore measure both coverage and allocation, not allocation alone.

Each learner path gets an independent categorical scenario ID. IDs and sampler
noise are detached. Generator Adam uses the frozen campaign rate; its objective
includes exact categorical `0.01*KL(q||prior)`. Do not replace the summed
DPG/SOAR score with a mean, standardize their rewards, or change teacher rates
after inspecting final results.

## DPG

At the pre-block state, introduce positive virtual weights `w[k,b]` and define
each learner minibatch objective as

`R_w = (logsumexp(gamma*loss + log(w)) - log(sum(w)))/gamma`.

Differentiate the held-out target risk after all K actual learner updates with
respect to each weight at all-one weights. Reward is the raw negative derivative.
The differentiation includes every update's clipping, carried Adam moments,
bias correction and all dependencies on preceding updates within this block.
Incoming model/optimizer state and sampled paths are detached before the block.
The actual all-one post-block learner is retained; no virtual student replaces it.

The normalization denominator is necessary: multiplying every weight by the same
constant must leave the block unchanged. Consequently each minibatch's weight
derivatives sum to zero, up to numerical precision. This is a useful GPU audit
invariant, alongside finite differences of the complete nonlinear target.
An average of weighted individual exponentials is not this objective.

The teacher objective is `-sum(reward*log(q[id])) + 0.01*KL(q||prior)`.
For hard arms the target is held-out hard empirical entropic risk. For mixed
arms it is `0.5*(R_easy/s_easy + R_hard/s_hard)` with positive scales frozen
before teacher training. Feedback, selection and final streams remain disjoint.

`dpg_attributions(student,batches,block_fn,target_fn,weight_shape)` returns
`(post_student, learner_values, rewards, target_after)`.

## SOAR

Evaluate the same frozen feedback batch before and after the real persistent
learner block. The scalar improvement is `target_before-target_after`.
Subtract the pre-update EMA baseline; broadcast this scalar advantage to every
path ID in the block. The EMA becomes `.9*old + .1*improvement`.
Update the generator with the same summed score and exact KL as DPG. SOAR
does not differentiate through the learner trajectory.

`soar_reward(pre_model,post_model,target_fn,baseline)` returns
`(advantage,new_baseline,metrics)`. Call `generator_step` with the advantage
broadcast to the complete ID shape.

## Selfplay

Keep pre-block learner snapshots indexed by outer round. At round t, retrieve
snapshot `floor(t/2)`. Form historical-minus-current parameter movement and
multiply leafwise by the *actual learner* Adam preconditioner
`learner_lr/(sqrt(nu/(1-beta2**count)+adam_eps_root)+epsilon)`.
The shared optimizer migration sets adam_eps_root=1e-16, and selfplay must use
that same value. The default function argument remains zero for earlier audits.

A single JVP per minibatch returns directional terminal-loss derivatives. For
the entropic learner, intrinsic reward for path i is
`abs(B*softmax(gamma*loss)[i]*directional_loss[i])`. Normalize the softmax within
each original learner minibatch, not across the entire K-block. This is the
per-path entropic-gradient contribution on a mean scale, rather than an MSE
gradient or raw exponential. No easy/hard feedback is accepted by this function.

The teacher objective is mean centered/std-normalized reward policy gradient
plus positive-reward expert iteration normalized by total reward, plus the same
prior KL. The expert coefficient is one. At t=0, or exactly zero parameter
movement, preserve both generator logits and generator optimizer state. Do not
advance generator Adam merely because a zero reward was observed.

Historical snapshot arrays are rebound to the current frozen parameter-tree
structure after checking leaf counts, shapes and dtypes. This avoids private
Equinox Missing-sentinel identities changing after separate pickle loads;
historical numerical parameters are preserved exactly.

`selfplay_rewards(model,historical,opt_state,batches,losses_fn,learner_lr,t)`
returns rewards shaped `[K,B]` and movement metrics. Use
`generator_step(...,selfplay=True,apply_update=(t>0)&(movement>0))`.

## Integration and GPU audits

`weighted_block((model,opt_state),batches,weights,losses_fn,optimizer)` returns
`((new_model,new_opt_state), minibatch_risks)`. By default it rematerializes each
learner step to bound DPG autodiff storage. Market rollouts should additionally
rematerialize their daily scan body when necessary; this changes storage and
recomputation, not the learner or teacher mathematics.

GPU validation must verify all-one weighted/unweighted equivalence; exact
all-one DPG post-state vs ordinary block; DPG finite differences with a nonlinear
batch entropic objective and carried nonzero Adam moments; per-minibatch weight
derivative sums; generator score gradient and fixed prior floor; SOAR baseline
timing; selfplay JVP/entropic contribution identity; no generator change at round
zero; and checkpoint replay including generator state, EMA, historical snapshots,
optimizer, PRNG and cumulative active time. Profile the full market shapes before
freezing runtime/storage caps. Actual empirical-risk gradients remain finite
minibatch estimators, not exact population entropic gradients.
