# Validation record

Supervisor runs only metadata/source checks. No local numerical tests were executed in this resumed session.

- Python AST/source syntax checks: passed for submitted source and orchestration.
- `python scripts/check_control_plane.py`: passed over-budget and over-deadline rejection, reservation and reconciliation.
- Provider API: no pre-existing instances; first stale offer rejected before creation; replacement instance 56458749 entered running status.
- GPU numerical pytest, coherent pricing preparation, dataset preparation: pending worker outputs.
- Scientific readiness, seven-arm training, checkpoint resume and final artifact completeness: unproven.

Official API references reviewed: https://docs.jax.dev/en/latest/benchmarking.html , https://docs.jax.dev/en/latest/persistent_compilation_cache.html , https://docs.jax.dev/en/latest/installation.html . Detailed method/reference audits are in teacher-review.md and learner-review.md.

GPU validation03 (instance 56459935, RTX4090, 2026-10-17): pinned environment installed and real preflight loss fell 4.821553 → 0.105911. Pytest: 18 passed, 2 failed in 67.48s. Both failures were stale test assertions treating nested Optax tuple.count as Adam count; preceding actual update/finite-difference comparisons passed. Current tests use find_adam_state. Full corrected GPU rerun required. Logs/artifacts synced; destruction confirmed. Pricing preparation and integrated smoke were not reached.

Infrastructure validation02: download failure/slow route, no numerical experiment started; explicitly terminated and API-confirmed destroyed. Local logs retained; final remote fetch failed after destruction (no scientific outputs existed). Subsequent offers exclude the observed failing region and prefer available European hosts.

GPU validation05 rejected fallback:24/1440 nonfinite stressprices, max coarse/fine discrepancy0.002086; reference convergence8e-6 also failed. In-grid bitwise outputs/tablehash preserved. No training launched. Export and provider destruction confirmed. Correction underway; no tolerance relaxation.

Validation08: GPU fallback stress PASSED all1440 states, independent converged doubleprecision reference, losslessquadrature transport, unchanged table/in-grid values, float32learneroutputs. Fixeddata generation and integratedchecks nowrunning.

Fallback08 exactevidence: maxabsolutepriceerror 2.3697464257566025e-07, referenceconvergence 9.414691248821327e-14; all1440finite, float32outputs and defaultx64False verified. Integratedsevenarms+actualselfplayresume PASSED. RealSIGTERM savednewcheckpoint, baseline6→472learnersteps, childexit0/statusstopped. Productionprofiling underway.

Telemetry scope review: training pricing_out_of_bounds_count counts path-time states in sampled simulation blocks (~30-active-second log cadence), NOT cumulative training totals; max_grid_distance is likewise sampled. All blocks nevertheless check synchronized spot/call finiteness before learner updates, and fallback convergence/bounds failures become NaN and terminate. Evaluation dataset masks are complete; repeated evaluations recount the same fixed states. Final report MUST disclose that total training fallback frequency and maximum training excursion were not recorded; never infer zero fallback use from logged zeros. See results/telemetry_scope_review.json. No source edits or scientific changes.


Final closure 2026-10-18T02:29:46.975528+00:00

Final GPU reporting passed1191 raw event checks and metadata completeness across21lineages/169checkpoints/507finaldomain evaluations. Byte/source recovery audit passed610229historyreferences. All paid workers exported and destroyed; see results/campaign_signoff.json.
