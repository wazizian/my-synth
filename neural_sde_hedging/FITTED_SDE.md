# Hard-regime fitted SDE baseline

`--method fitted-sde` fits the same neural SDE architecture used by the adaptive
methods to **hard-regime training paths**, freezes it, and trains the common
hedger on fresh simulated paths. It supports options and continuous credit;
the discrete credit grid model is excluded.

For every daily transition, minimize the Euler Gaussian negative log likelihood:

\[
  -\frac{1}{BT}\sum_{b,t}
  \log\mathcal N\!\left(x_{b,t+1};
      x_{b,t}+\mu_\phi(x_{b,t},t)\Delta t,
      L_\phi(x_{b,t},t)L_\phi(x_{b,t},t)^\top\Delta t\right).
\]

This includes the diffusion log determinant; it is not squared prediction error.
Each fit update samples whole paths with replacement and uses **every transition**
on those paths. `--fit-steps` counts optimizer updates, not simulated days.
Use `--teacher-lr` for fitting; gradients are clipped at norm 1 with the existing
AdamW optimizer. Defaults (1000 updates, 32 paths/update) are unvalidated.

Input `--hard-paths` is an NPZ with `states[B,T+1,D]`:
options use log spot plus both simulator latent factors (`D=3`), continuous
credit uses the log hazard factor (`D=1`). Paths must be finite and start at the
simulator's fixed initial state. This is a **full-state fitting baseline** with
privileged latent training data. Price-only fitting needs a separate latent
inference method. The likelihood identifies physical drift and diffusion;
it does not separately identify risk-neutral drift and risk premium.

Supply a separate `--validation-paths` file from the hard regime. Validation
never enters the fit or hedger updates. By default, feedback diagnostics sample
the hard training file; an explicit `--feedback-paths` can override this.
Files with identical hashes are rejected for fitting versus validation, but
independence and absence of partial path overlap remain the caller's responsibility.
There is no fallback that silently calls the initial prior a hard regime.

Example on an already provisioned GPU:

```bash
python -m neural_sde_hedging.training \
  --method fitted-sde --problem options --output runs/options-fitted \
  --hard-paths data/hard-train.npz --validation-paths data/hard-validation.npz \
  --fit-steps 1000 --fit-batch-size 32 --teacher-lr 0.001 \
  --seconds 3600 --rounds 1000
```

For credit add `--problem credit --credit-simulation continuous` instead of
`--problem options`, and supply credit states with the corresponding horizon.
The existing fixed neural-Q pricing reference and marginalized default ledger
are retained; this estimates a physical training proposal, not a new calibrated
pricing model.

Fitting and hedger training share the time budget. Each fit update checkpoints
parameters, moments, RNG, dataset hash, and progress; resumption finishes fitting
before training the hedger. `fit-progress.jsonl` records transition NLL separately
from hedging risks in `progress.jsonl`. `fitted-generator.pkl` preserves the frozen
fit for inspection. No teacher rewards, replay, or teacher updates occur during
hedger training. The existing four-GPU launcher still launches its original four
arms; run this additional arm separately.

The fixed daily Euler scan is kept: it directly matches the Gaussian likelihood
and existing replay/pathwise code. A Diffrax solver would add integration setup
without shortening this implementation, and a different solver would require
reconsidering this transition likelihood.

Only Ruff and compilation have been run. Added GPU checks cover the Gaussian
formula, diffusion gradients, fitting/training, and resumption; they remain unrun.
