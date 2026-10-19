# Frozen eight-arm final evaluation

Only invoke production main after all eight selections are frozen. `final_evaluation.py` is GPU-only. The control-plane `package_evaluation.py MANIFEST WORKER` reads source and artifact bytes without importing project code. It includes adaptive.py so full trainer checkpoints can unpickle generator classes.

Manifest fields:

```json
{
  "benchmark": "credit",
  "source": "/absolute/campaign/source",
  "bp": 3,
  "frozen": true,
  "seed": 520261017,
  "episodes": 8192,
  "bootstrap": 1000,
  "models": {
    "baseline": {"checkpoint": "/selected.pkl", "selection": "/selection.json", "sha256": "...", "kind": "pickle"}
  },
  "trial_curves": [{"arm": "baseline", "lr": 0.0001,
    "selection": [{"active_seconds": 60, "path": "/selection-60.json"}],
    "monitor": [{"active_seconds": 60, "path": "/monitor-60.json"}]}]
}
```

`models` must contain exactly baseline, easy_reference, hard_reference, dpg_hard, soar_hard, selfplay, dpg_mixed, soar_mixed. Pickle loads payload.student[0]. Optional legacy kinds are credit-eqx (requires selected lr for optimizer skeleton) and option-eqx (model only). Each selection must not claim final-test usage; hashes must match before any final episode is drawn.

Option manifests set benchmark=option, source=volatility campaign root, and initial_premiums={easy:...,base:...,hard:...} using the converged numerical Q diagnostic (including future arrivals). Final terminal seed520261017, final option daily diagnostic920261017, Q seed948/949. Reference stream820, early option screen821 and confirmation822 are separate. Audit entry points use997/998; they never call production main or open520/920.

Outputs include results.json, whole-episode terminal NPZs per domain, online daily aggregates, credit default/ledger audits and256-episode vintage/holding summaries, option256-episode independent Q marked errors with4096 quote paths and independent replicate RMS errors, terminal/daily plots, and all-LR per-domain development learning curves when trial_curves is supplied. Credit raw daily losses and alive/default/state records are exported as atomic lossless128-path NPZ blocks with seed/fold lineage, SHA256 manifest and readback/accounting checks. Raw option daily errors, holdings, gains and costs are archived for the marked diagnostic subset. Full raw credit vintage tensors are not exported;256-episode vintage aggregates remain available. Conditional paired bootstrap is not across-training-seed uncertainty. Credit turnover is explicitly discounted turnover inferred from cost/transaction-cost.

Dedicated evaluator audit is under final-evaluator-audit, source hashes frozen there. Validate its audit/validation.json, artifacts.json, final sync and API-confirmed destruction before claiming the evaluator validated.

Raw option diagnostic `terminal_loss_*` arrays equal the sum of that subset's daily marked errors, using its independent Monte Carlo mark at issuance. The8192-episode terminal test arrays instead subtract the frozen converged initial premium. These are distinct declared streams; the diagnostic initial-price shift does not affect cross-episode daily standard deviations. The marks and price-replicate errors are archived alongside the diagnostic paths.
