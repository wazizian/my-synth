# Tasks and interfaces

Authoritative plan /home/jump/plan.md. Numerical work only on rented GPU workers.

- Market agent: src/market.py, src/pricing.py, tests/test_market.py, scripts/prepare_market.py. GPU rewrite/source audit complete; numerical validation pending. Now read-only orchestration audit.
- Teacher agent: src/generator.py, src/teachers.py, tests/test_teachers.py source audit complete. Now owns src/runner.py, scripts/run_trial.py for persistent seven-arm training/resume/evaluation.
- Learner agent: src/hedger.py, src/objective.py, src/checkpoint.py, src/metrics.py and owned tests source audit complete. Also scripts/make_report.py, scripts/audit_artifacts.py, README.md implemented; GPU/report execution pending.
- Supervisor: datasets/protocol, GPU runtime guards, central budget/queue, rentals, integration, decisions, final evidence and provider shutdown.

Interfaces:
Episode dictionary includes spot/call (B,85); trade dates64..83, liquidation84.
student_block returns ((model,opt_state), losses[K]); optional return_diagnostics=True adds third diagnostics result.
Teacher block callback retains default two-result contract.
train_one(root, config, active_seconds, resume=True) uses cumulative target runtime.
evaluate_checkpoint(root,run_id,checkpoint_path,role='final_test') archives all domains.
Final selection shape selected:{arm:{run_id,checkpoint_id}}, benchmark_hash, objective_scales_hash.
No agent may rent/terminate compute or launch experiments independently.


Final closure 2026-10-18T02:29:46.975528+00:00

All delegated and supervisor tasks completed. Final report, all169checkpoint evaluations, raw-array checks, recovery audit, provider billing and zero-rental shutdown verified. No pending tasks.


## Revised next-campaign plan (2026-10-18)

See PLAN.md and /home/jump/plan.md. Planning only; no new jobs launched. One shared training seed, eight arms including easy_reference, informative easy/hard specialist gaps, identical hedger architecture, entropic training, CVaR/entropic/daily-error-SD evaluation. Completed campaign closure above remains historical.


## Next design revision: BS/SLV alternatives (2026-10-18)

PLAN.md and /home/jump/plan.md now specify two bounded alternatives: A, BS easy / one-factor SLV base / hidden-regime SLV hard; B, stable low-turnover easy / intermediate base / stochastic-volatility and persistent-switch hard with shared menu support. One seed, identical stock-only hedger, eight arms, entropic training and CVaR/entropic/daily-error-SD metrics. The trinomial gate is closed and rejected; this is a plan update only. Read the plan for residual-risk acceptance and pricing/information safeguards.


## Selected next proposal: ZCB / rolling CDS credit-curve risk (2026-10-18)

The authoritative plan.md and PLAN.md now specify a financed long 5y ZCB and rolling daily-issued 6m CDS vintages. Easy: stable persistent credit factor; base: ordinary mean reversion; hard: slow credit quality plus temporary distress. One seed, same architecture, economic pre-screen then specialist gate. Old equity campaigns remain closed. Local audited source must be transferred explicitly; no new training or rentals launched by this plan update.


## Credit source uploaded (2026-10-18)

Audited one-factor source and tests: /home/jump/deep-hedging/credit-hedging/. All 15 transferred file hashes verified; see SOURCE_MANIFEST.json and credit_hedging/AUDIT.md. No worker launch or dependency installation. Two-factor extension remains planned.

## Parallel price-history-only volatility experiment (2026-10-18)

User authorized the multiple-cost, longer-training, pure-base versus easy/hard specialist screen on the uploaded volatility benchmark, parallel to credit. No RV/IV/quotes/latent factors as policy inputs. Campaign: `campaigns/volatility-shocks-20261018/`; see frozen `DESIGN.md`, `STATE.json`, and `monitor.latest.json`. Original demo GPU audit and coherent stochastic-arrival/pricing numerical gate precede training. No outcomes yet.
