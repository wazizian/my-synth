# Hard targets and hard specialist

The target law is independent of the trainable teacher. `--target-regime hard`
is the default; it never falls back to the initial neural prior.

- Generate **1,024 fixed hard training paths** once, from a dedicated seed.
- DPG and SOAR draw **128 paths without replacement** from that pool each outer
  iteration, shared by all comparisons in that iteration. They still generate
  their synthetic learner-training data using their teacher; the hard subset
  supplies the downstream objective/reward.
- At each evaluation (every 16 rounds and the last completed round), simulate
  **2,048 fresh hard paths**. These never enter teacher rewards or learner
  training. Evaluation uses a separate RNG namespace and round index, so the
  same round has common evaluation paths across arms and reproducible resume.
- `--method hard-specialist` trains the same initialized hedger directly on
  subsets of the same static hard training pool. It never adapts the teacher.
  Its inner training batch defaults to 128 (`groups * items`), with the same
  optimizer and inner updates as the other arms.

Pool size, feedback size and evaluation size are configurable. The pool is
recreated deterministically on resume; source/config fingerprints prevent
silently changing it. An optional `--feedback-paths` file replaces the static
hard pool, but its hard-regime provenance is the caller's responsibility.
`--validation-paths` is rejected for hard mode: evaluation must sample fresh.

Options restore the physical persistent-shock dynamics in the volatility
diagnostic: base volatility 20%, independent OU factors with physical half-lives
40 and 3 days, annual factor noise .35, and an imposed log(2) shock to each factor
at day 35. The stock uses variance `.04 * exp(factor_1 + factor_2)` and zero
physical excess drift. The policy sees stock history, not these factors.
The diagnostic's separate Q/rolling-option-IV calculation is not part of this
stock-only hedge; this does not establish a priced IV/RV benchmark or stochastic
shock arrivals. Horizons below 36 steps do not contain a shock.

Continuous credit has a **new one-factor persistent-hazard analogue**, not the
historical two-factor credit-curve benchmark: a centered OU factor with a 40-day
half-life and annual noise .35, log(2) shock at day 35, and positive hazard via
softplus. Existing fixed-Q quote tables and the default-integrated CDS ledger
are retained. This credit law and the options law have not passed an economic
hardness or specialist-performance gate. The discrete credit adapter requires
an explicit non-hard target mode.

Example hard specialist on an already provisioned GPU:

```bash
python -m neural_sde_hedging.training \
  --method hard-specialist --problem options --target-regime hard \
  --hard-pool-size 1024 --feedback-size 128 --validation-size 2048 \
  --output runs/hard-specialist --seconds 3600 --payoff-cap 1
```

The prior pause remains in force for neural-teacher methods; their entry points
still require `NEURAL_SDE_ENABLED=1`. The non-neural hard specialist does not
require that switch. The existing four-GPU launcher propagates hard target mode
but retains its four original arms; the specialist is a separate training job.
No new GPU allocation or training run has been started. Only Ruff and
compilation were run; numerical checks are added to the GPU suite and unrun.

`--target-regime prior` explicitly retains the old neural-prior demonstration,
and `--target-regime external` retains fixed externally provided training and
validation files for historical/fitting experiments. Neither is the default.
