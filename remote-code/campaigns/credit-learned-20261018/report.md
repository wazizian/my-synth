# Learned credit baseline and specialists

The learned baseline beats the development-tuned fixed-notional hedge on P0: entropic risk 0.00237519 versus 0.00381202, about38% lower, with paired95% gain interval [0.00116122,0.00173631]. The easy and hard specialists do not establish additional specialization gains. The hard-trained policy has the lowest point-estimate risk across domains; the intended easy/hard residual-risk ordering is not established.

Each arm independently selected its best learning rate from the same search. An additional same-rate check was cancelled at the user's request before numerical computation; no same-rate result is claimed.

Three recurrent hedgers were trained on the unchanged credit markets: exact P0 mixture, easy only and hard only. All arms used identical initialization, GRU(382,32), target/retention action map, entropic gamma10, batch160, clipping1 and the same three learning rates and 600-active-second budgets. Each trial started exactly at one-unit expiry replenishment; the initial checkpoint was eligible for selection. This warm start and action parameterization are shared improvements to the baseline, not privileged specialist information.

All selected policies were frozen using separate development episodes before this fresh paired evaluation of8192 episodes per component. Base is the exact .05 easy/.90 ordinary/.05 hard empirical mixture. One training initialization; paired bootstrap intervals measure evaluation uncertainty only. No adaptive teacher ran.

| Evaluation domain | Hedge | Entropic | CVaR95 | MSE | Daily SD | Mean cost |
|---|---|---:|---:|---:|---:|---:|
| easy | baseline | 0.00344468 | 0.05746958 | 0.00045942 | 0.00059487 | 0.00081719 |
| easy | easy | 0.00354014 | 0.05761193 | 0.00049662 | 0.00062469 | 0.00080159 |
| easy | hard | 0.00325814 | 0.05558298 | 0.00043012 | 0.00057642 | 0.00081514 |
| easy | replenish_1.0 | 0.00935111 | 0.07412708 | 0.00200934 | 0.00125137 | 0.00089382 |
| ordinary | baseline | 0.00232835 | 0.04333878 | 0.00028466 | 0.00047989 | 0.00081553 |
| ordinary | easy | 0.00251929 | 0.04646416 | 0.00033326 | 0.00052155 | 0.00080230 |
| ordinary | hard | 0.00228152 | 0.04044139 | 0.00025905 | 0.00045560 | 0.00081304 |
| ordinary | replenish_1.0 | 0.00864186 | 0.05961944 | 0.00191376 | 0.00122209 | 0.00089150 |
| hard | baseline | 0.00214249 | 0.04084114 | 0.00025195 | 0.00045031 | 0.00081341 |
| hard | easy | 0.00234119 | 0.04411652 | 0.00030237 | 0.00049673 | 0.00080088 |
| hard | hard | 0.00198967 | 0.03797454 | 0.00022676 | 0.00042765 | 0.00081364 |
| hard | replenish_1.0 | 0.00805074 | 0.05557279 | 0.00185404 | 0.00120802 | 0.00089213 |
| base | baseline | 0.00237519 | 0.04408272 | 0.00029176 | 0.00048487 | 0.00081551 |
| base | easy | 0.00256168 | 0.04699698 | 0.00033988 | 0.00052599 | 0.00080219 |
| base | hard | 0.00231602 | 0.04124288 | 0.00026599 | 0.00046106 | 0.00081318 |
| base | replenish_1.0 | 0.00864798 | 0.06027837 | 0.00191555 | 0.00122289 | 0.00089164 |

On its P0 training distribution, the neural baseline has entropic risk 0.00237519 versus 0.00381202 for the development-tuned static hedge (target 0.85). Gain 0.00143683, paired95% interval [0.00116122, 0.00173631].

## Specialization and static comparators

| Target | Specialist gain over neural baseline | Paired95% interval | Lower bound above0.001? |
|---|---:|---|---|
| easy | -0.00009546 | [-0.00032987, 0.00013976] | False |
| hard | 0.00015282 | [-0.00000261, 0.00031647] | False |

Static protection levels were selected independently on development data from the frozen menu, not from final outcomes. The same full-calendar vintage ledger and costs apply.

- baseline: selected static target 0.85.
- easy: selected static target 0.85.
- hard: selected static target 0.85.

| Target | Learned hedge | Entropic gain over its development-tuned static comparator | Paired95% interval |
|---|---|---:|---|
| easy | baseline | 0.00155750 | [0.00118613, 0.00190587] |
| easy | easy | 0.00146204 | [0.00104259, 0.00188779] |
| hard | baseline | 0.00165407 | [0.00128406, 0.00199549] |
| hard | hard | 0.00180689 | [0.00136533, 0.00223229] |

Hard-minus-easy specialist residual diagnostics: `{"terminal_std": -0.007231915404824211, "centered_entropic": -0.0015085712752280292, "centered_cvar95": -0.019595492543057647, "daily_error_std": -0.00019703038368817168}`. A larger loss level alone does not establish difficulty.

## Training and selection

| Arm | Selected learning rate | Selected update | Selected active seconds | Trial terminal steps |
|---|---:|---:|---:|---|
| baseline | 0.0003 | 1056 | 600.55 | 0.0001: 1058, 0.0003: 1056, 0.001: 1058 |
| easy | 0.001 | 1366 | 600.04 | 0.0001: 1363, 0.0003: 1367, 0.001: 1366 |
| hard | 0.001 | 2274 | 600.14 | 0.0001: 2273, 0.0003: 2274, 0.001: 2274 |

Training uses160 paths per step. Baseline uses fixed8/144/8 stratification and the weighted exponential moment; this estimates mixture entropic risk rather than averaging component risks. The finite-minibatch log-mean-exp objective has sampling bias and is not an exact population gradient. Initial compilation includes one retained learner update, excluded from active timing equally for all trials. Compilation, evaluation and I/O are additional paid costs; equal active time may yield differing update counts. All tuning trials and resumable model/optimizer/PRNG checkpoints are retained.

## Numerical scope

The preceding campaign validated the unchanged production grids and pricing convention against refined grids and boundary widening. All41 GPU tests passed before training, including causal accounting, gradients, exact initial static-rule equivalence and checkpoint replay. Selected-policy full-calendar raw records passed accounting/finite/bound checks at1e-10. This is a daily finite-state model with custom daily-coupon CDS, not a continuous-time error certification or ISDA market model. Policies see quotes and inventory, not simulator labels or bond marks. Quotes may reveal state information.

Risk metrics are positive discounted loss per bond face; CVaR95 uses fractional tail mass. Daily population SD includes absorbing zeros after default. Opening-trade turnover excludes final liquidation notional; costs include liquidation. Initial bond value, mean and centered risks, default/survival outcomes and entropic ESS are retained in metrics.json. Raw selected-policy records include every daily loss, discounted cashflow/cost, purchase, sale and total notional, plus scenario paths. They do not contain full per-vintage cubes; replay uses retained policies, sources and paths. No across-seed or global-optimality claim is supported.

![All tuning curves](figures/tuning-curves.png)

![Easy risk](figures/easy-risk.png)

![Hard risk](figures/hard-risk.png)

Provider-reported campaign cost at closure: $1.172, including infrastructure failures and the cancelled setup. All campaign GPU instances are confirmed shut down; no reservations remain. Charges may settle later.

Final artifact/protocol audit: 2957 checks passed. See `artifact_audit.json`, `completion_audit.json`, `provider_billing.json` and the `launchers/` output directories. The cancelled additional check lost no scientific results because cancellation occurred during dependency setup; its expected retrieval failure and setup logs are retained.
