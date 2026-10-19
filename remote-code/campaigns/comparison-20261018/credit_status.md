# Credit campaign: completed and reconciled

Authoritative audit: 2026-10-19T10:43:54.087898+00:00. The original pipeline error/status and recovery history are preserved; no new rental or scientific recomputation was performed after the deadline.

- All eight arms completed all three learning-rate trials: **24/24 trials**, selected checkpoints retained, terminal exit0 and no active launcher markers.
- The sealed final evaluation completed with seed520261017 and8192 episodes per domain. `results.json` and `complete.json` both record completion. Markdown/LaTeX reports, plots and all-LR learning curves are exported.
- Independent recovery completed the final SHA-verified export: **234 files**, including **192 raw daily archives**. Current file presence/size and raw-manifest/export-digest agreement have zero discrepancies. Large payload hashes were not recomputed on the control host.
- All three complete selfplay history archives have exact worker SHA/readback proofs on private volume56629013; current ownership/state was verified. Archive manifests and small checkpoints remain local. Retention is USD0.32 per30days for120GB; review2027-01-16, contract expiry2027-04-12. This volume is intentionally retained, with no attached GPU.
- Fresh provider API returned **zero instances**. Available credit at10:41 was USD7.1377; final financial totals are maintained in the root financial supplement.

The pipeline stopped after an export timeout/rsync12, then correctly refused a scientific retry. The independent recovery service finished export verification and deleted the evaluation worker afterward. Therefore its old `waiting_for_eight_frozen_training_selections` status is stale; `credit-pipeline-reconciliation.json` is the authoritative completion receipt.

All six original specialist gates failed. GPU freeze selected ordinary-credit10bp under the predeclared ranking; this comparison is **exploratory**. Claims concern the coherent finite-state production law, not a certificate of continuous-model production-grid convergence. Intervals are conditional on one fitted training initialization.

Selfplay recovery retained the original run/worker, restored checkpoint model/optimizer/generator/RNG unchanged, and charged7.55641655 known lost active seconds. The71.70605612-second uncertain failed-overhead bound is separately disclosed and is not active training time. Original/intermediate/final checkpoints, raw lost records and receipts are preserved.

Evidence: `credit-pipeline-reconciliation.json`; `credit-final/evaluation/launcher/runs/20261019T021407Z-6f5fb002/export-verified-manifest.json`; the corresponding `outputs/evaluation/report.md`, `results.json`, `raw-record-manifest.json`; selfplay `external-artifacts.json` and `external-readback.json`. Earlier status history is archived alongside this file.
