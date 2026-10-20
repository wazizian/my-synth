# Deep hedging with distinct benchmark markets

The default runner is `--market distinct`. Its economic markets and hedgers
restore the 2026-10-18 distinct-regime experiment. The short-lived day-35
plotting-diagnostic adapter is no longer the default.

## Fixed market laws

Options have three separate regimes:

| Regime | Physical factor half-lives (days) | Q half-lives (days) | Daily jump probability |
|---|---|---|---|
| Easy | (3,3) | (3,3) | 0 |
| Base | (3,3) | (40,40) | 1/63 |
| Hard | (40,3) | (1,40) | 1/63 |

All have factor noise .35, baseline variance .04, variance capped at 1, and
simultaneous log(2) jumps to both factors at the **end** of an interval. Stock
returns use current factors before that interval's jump. There is no imposed
shock date. Q pricing includes future random jumps. Rolling ATM IV and European
call pricing are available in `option_benchmark.py`.

Credit restores the two-factor finite-grid model, original per-regime calibrated
production grids and original grid-consistent ZCB/CDS prices. A long five-year
ZCB is hedged using daily new six-month CDS and sales of existing vintages.
See [CREDIT.md](CREDIT.md) and the versioned production configs in
`credit_hedging/configs/distinct_regimes.json`.

## Training and evaluation

- `baseline` trains on **fresh base-regime** paths each round (ordinary for credit).
- `easy-specialist` and `hard-specialist` train on **fresh own-regime** paths.
- DPG/SOAR keep a static 1,024-path hard feedback pool and uniformly resample
  128 paths per iteration, shared across their within-round comparisons. Their
  synthetic learner data comes from their teacher, not from this feedback pool.
- Scheduled evaluation draws **2,048 fresh paths from each of easy/base/hard**.
  Evaluation randomness is independent of learner training and the static pool;
  matching rounds use common draws across arms. No evaluation data enters rewards.
- Every arm within a problem uses the same architecture and initialization, with
  one initialization seed. Learner updates use the historical clip-1/Adam form.
  Current batch sizes, learning rates, budgets and inner step counts are CLI
  settings, not a reproduction of the old LR-selection campaign.

The options learner is the original GRU32 with zero output head, holdings in
**[0,1] shares**, and features current log price, previous log price, remaining
time and previous shares. It hedges an ordinary uncapped ATM call. The credit
learner is the original float64 target/retention GRU32 initialized to unit
replenishment and full retention. No latent state or regime ID reaches a policy.

Teacher algorithms remain DPG/GRPO, SOAR/dataset-RLOO and self-play with
replay/alignment. The option teacher remains a neural-SDE synthetic proposal.
The restored credit teacher is a neural physical-transition proposal on the
**hard two-factor grid**, initialized to the true hard transition law; its
likelihood includes both factor cell probabilities and daily default/survival.
It does not change the contractual Q price tables. There is no categorical
mixture generator. Credit grid/default draws are discrete: pathwise meta-learning
is explicitly rejected for this benchmark. The fitted-SDE variant is also
rejected here pending the requested price-only fitting work.

## Metrics and pricing

Report terminal entropic risk, fractional CVaR95 and population terminal SD.
Credit daily error SD comes from the exact daily marked ledger. Options daily
marked error uses Q mark increments minus stock gains plus trading costs,
including final liquidation. It is computed only at scheduled evaluation,
using a declared subset (default 64) and Monte Carlo Q paths (default 4,096).
Quote standard-error diagnostics and telescoping error are logged; quote noise
can affect daily SD. Initial option Q premiums are fixed across arms and rounds
within each regime, with their Monte Carlo standard errors recorded.

Entropic ESS/max-weight diagnostics remain. Neither a restored regime label nor
finite static checks demonstrate the desired economic difficulty or specialist
performance gap. The original unbounded payoff/share setup also has no guarantee
of finite population exponential moments; finite-sample risk is not such a proof.

## Running

Baseline and specialists do not require the neural-teacher opt-in:

```bash
python -m neural_sde_hedging.training --market distinct --problem options \
  --method baseline --output runs/options-base --seconds 3600
python -m neural_sde_hedging.training --market distinct --problem options \
  --method easy-specialist --output runs/options-easy --seconds 3600
python -m neural_sde_hedging.training --market distinct --problem credit \
  --method hard-specialist --output runs/credit-hard --seconds 3600
```

Only run these on an already provisioned GPU under an authorized budget.
Neural-teacher methods remain paused unless explicitly enabled with
`NEURAL_SDE_ENABLED=1`. No new rental or numerical job has been started.
Checkpoints preserve learner/teacher/optimizer/RNG and source/config fingerprints.
Production hard-credit pricing alone requires roughly 10 GB of GPU memory for
its bond table; `--credit-grid-size` and `--credit-fast-grid-size` are explicit
smoke approximations, not the calibrated production discretization.

`--market neural-demo` retains the earlier experimental adapters; see
[NEURAL_DEMO.md](NEURAL_DEMO.md). It is not the distinct-regime experiment.
The four-GPU launcher uses the restored default for its original four arms;
specialists remain separate jobs. Only Ruff and py_compile have been run on
these changes. New numerical equivalence tests against archived source are
GPU-only and remain unrun.
