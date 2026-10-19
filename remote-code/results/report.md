# Adaptive deep hedging exploratory campaign

**Post-campaign cleanup (2026-10-18T08:59:12.985351+00:00):** Code, reports, figures and all canonical metrics were retained. Datasets, checkpoints, selfplay history, worker exports and caches were intentionally deleted at the user’s request. Archive-completeness and recovery claims below describe the pre-cleanup campaign. See [cleanup manifest](cleanup_manifest.json).

Archive complete: seven selected arms, 21 retained hedger lineages, 169 checkpoints and 507 final checkpoint-domain evaluations. GPU raw-event validation and metadata completeness passed. Source/checkpoint recovery audit passed. Smoke and profile lineages are retained separately from the seven scientific arms; infrastructure-only preflight jobs are recorded in worker/budget evidence.

| Arm | Base MSE | Easy MSE | Hard MSE | Balanced scaled loss | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 6.65735e-05 | 4.24845e-05 | 0.000199676 | 0.055596 | completed |
| hard_reference | 7.21253e-05 | 4.68484e-05 | 0.000199484 | 0.057629 | completed |
| dpg_hard | 7.26195e-05 | 4.62924e-05 | 0.000203204 | 0.0580265 | completed |
| soar_hard | 6.81072e-05 | 4.31532e-05 | 0.000199486 | 0.055879 | completed |
| selfplay | 7.17161e-05 | 4.67711e-05 | 0.000203214 | 0.058255 | completed |
| dpg_mixed | 7.16131e-05 | 4.56085e-05 | 0.00020071 | 0.0572595 | completed |
| soar_mixed | 6.84847e-05 | 4.37201e-05 | 0.00019908 | 0.0560755 | completed |

## Selected-result interpretation

The fixed base-distribution baseline has the lowest observed base MSE, easy MSE, and balanced scaled easy/hard objective. No adaptive arm demonstrates broad improvement in this single-seed run. SOAR mixed has the lowest measured hard MSE, but its advantage over baseline is only 5.96e-7 and comes with worse base and easy losses. SOAR hard and the hard reference have still smaller hard-MSE advantages. These are measured differences, not established statistically or numerically robust improvements.

Mixed feedback improves DPG over hard-only DPG in all three domains, yet both have higher MSE than baseline in all three selected-test domains. Within SOAR, mixed feedback improves hard MSE while worsening base and easy MSE relative to hard-only feedback; its balanced objective is also worse. Both SOAR variants outperform selfplay on all three selected-test MSEs. Selfplay's average PnL is closer to zero, but its overall squared error is higher. Base retention is therefore unfavorable for every adaptive arm, rather than an unreported cost of specialization.

The hard reference is a fixed hard-distribution comparator, not an overall bound. Its hard MSE is nearly identical to SOAR hard, with worse base/easy MSE. Hard-reference gap closure is deliberately not reported: the baseline-minus-reference denominator is only 1.92e-7 and is too small for a useful stable ratio.

Each final domain contains 8,192 immutable episodes. “Raw MSE” means squared shortfall normalized by inception spot, before division by the fixed domain reference scale; it is not dollar-squared error. Selection was frozen at 2026-10-18 01:54:09 UTC using the balanced development objective, before any final-test evaluation. All selected checkpoints are terminal 2,700-second checkpoints except SOAR mixed, whose selected checkpoint is at 1,800.06 active seconds; its lineage still completed the common 2,700-second budget.

Positive Delta below means baseline MSE minus method MSE. Values are copied from GPU-generated final_metrics.csv.

| Arm | Delta base | Delta easy | Delta hard |
| --- | ---: | ---: | ---: |
| baseline | 0 | 0 | 0 |
| hard_reference | -5.55177e-06 | -4.36393e-06 | 1.92464e-07 |
| dpg_hard | -6.04597e-06 | -3.80793e-06 | -3.52762e-06 |
| soar_hard | -1.53368e-06 | -6.68668e-07 | 1.90019e-07 |
| selfplay | -5.14254e-06 | -4.28656e-06 | -3.5378e-06 |
| dpg_mixed | -5.03957e-06 | -3.12397e-06 | -1.034e-06 |
| soar_mixed | -1.91121e-06 | -1.23556e-06 | 5.95872e-07 |

Tail rankings differ from MSE rankings. Hard reference has the lowest hard-domain ES95 (0.0387722); DPG mixed has lower hard ES95 (0.0393256) than baseline (0.0395784) despite higher MSE. SOAR mixed's hard ES95 is worse than SOAR hard's. Full signed PnL, VaR/ES, transaction costs, turnover, holdings and saturation measurements are retained in the CSVs and raw episode archives; MSE alone is not a claim of dominance across financial outcomes.

At the end of training, targeted generators remain close to q0: dominant base-component mass is 0.71056/0.70414 for hard/mixed DPG and 0.71282/0.70489 for hard/mixed SOAR, versus initial 0.70; corresponding KL values are 0.000569/0.000895/0.002107/0.000604. Selfplay moves substantially (base mass 0.035, KL 2.257), concentrating on three other components. These are terminal distributions, not necessarily the distribution at the selected checkpoint. Their trajectories and the differing update counts contextualize the outcome without identifying its cause.

The equal active-time budget yields the following terminal-lineage learner update counts: 261,348 baseline updates, 263,456 hard-reference updates, 83,676/75,064 hard/mixed DPG updates, 210,012/197,184 hard/mixed SOAR updates, and 240,900 selfplay updates. DPG's integrated computation buys fewer learner steps and no observed improvement over baseline here; the experiment does not show its detailed attribution cost to be worthwhile under these settings. Selfplay also uses 555.5 seconds of checkpoint/history I/O and retains 4.25 GB of exact history. Active time excludes that I/O, compilation and evaluation, although all consume paid wall time. Workers share the RTX 4090 model but not controlled hosts/clocks, so timing differences cannot all be attributed to teacher mechanisms.

## Measured compute

| Run | Compile s | ms/block | Active paths/s | Sampled VRAM MiB | Allocator peak in-use MiB | Allocator peak pool MiB | Checkpoint/history I/O s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline-lr001 | 16.6474 | 41.3243 | 24779.6 | missing | 256.612 | 258 | 0.449285 |
| dpg_hard-lr001 | 55.0781 | 129.074 | 7933.46 | missing | 849.021 | 1538 | 0.507096 |
| dpg_mixed-lr001 | 26.5704 | 143.884 | 7116.83 | missing | 922.697 | 1538 | 0.895392 |
| hard_reference-lr001 | 22.8587 | 40.994 | 24979.3 | missing | 237.21 | 514 | 0.416315 |
| profile_baseline | 20.3752 | 40.7971 | 25099.8 | 959 | missing | missing | 0.0858157 |
| profile_dpg_hard | 26.4943 | 100.386 | 10200.6 | 1993 | missing | missing | 0.0571558 |
| profile_dpg_mixed | 13.8866 | 103.038 | 9938.09 | 2001 | missing | missing | 0.0527882 |
| profile_hard_reference | 14.522 | 41.3815 | 24745.4 | 961 | missing | missing | 0.0770507 |
| profile_selfplay | 8.7778 | 40.7709 | 25116 | 1995 | missing | missing | 2.22472 |
| profile_soar_hard | 5.83293 | 42.3327 | 24189.3 | 1993 | missing | missing | 0.0659873 |
| profile_soar_mixed | 6.58918 | 41.9379 | 24417.1 | 2001 | missing | missing | 0.0568441 |
| selfplay-lr001 | 27.1182 | 44.8321 | 22840.8 | missing | 256.573 | 514 | 555.528 |
| smoke_baseline | 20.6509 | 133.519 | 119.833 | missing | missing | missing | 0.0909062 |
| smoke_dpg_hard | 19.4171 | 502.443 | 31.8444 | missing | missing | missing | 0.0591853 |
| smoke_dpg_mixed | 10.9426 | 439.021 | 36.4447 | missing | missing | missing | 0.0495104 |
| smoke_hard_reference | 9.35909 | 292.619 | 54.6786 | missing | missing | missing | 0.0617813 |
| smoke_selfplay | 16.4047 | 470.344 | 34.0176 | missing | missing | missing | 0.547766 |
| smoke_soar_hard | 3.51869 | 397.615 | 40.2399 | missing | missing | missing | 0.0530049 |
| smoke_soar_mixed | 4.46711 | 388.05 | 41.2318 | missing | missing | missing | 0.0661306 |
| soar_hard-lr001 | 37.3705 | 51.426 | 19912.1 | missing | 256.615 | 514 | 0.543178 |
| soar_mixed-lr001 | 13.2486 | 54.7717 | 18695.8 | missing | 256.791 | 514 | 3.32491 |

Teacher time includes student updates; isolated teacher overhead is unmeasured. Evaluation throughput includes summary and archive I/O. Sampled nvidia-smi VRAM values remain lower bounds. Allocator peaks come from device.memory_stats() process-lifetime high-water counters: they include earlier sequential runs and evaluation in the same process, are not isolated per-trial peaks, and do not represent total device VRAM. Unsupported counters remain missing; profile.json retains observation process/device provenance. Profile shapes are recorded in profile.csv; tiny smoke timing does not establish production throughput. Profile-derived block counts and storage projections are estimates. The runner separately enforces immutable selfplay_max_blocks, selfplay_history_limit_bytes (12 GiB default), and selfplay_storage_reserve_bytes (1 GiB default), checking projected storage before training and block/storage availability during execution. Consult each run configuration and stop_reason for the applied limits and any early stop.

lineage_active_training_seconds records the maximum cumulative active training time observed for the run; active_training_seconds is its backward-compatible alias. checkpoint_active_training_seconds comes from the final-test events for that exact checkpoint and may be earlier than the lineage endpoint. Missing or inconsistent checkpoint timing remains null and is identified by checkpoint_active_time_status. Primary monitor/selection figures show the seven selected lineages; all_trials_<domain>_<split>.png preserves every retained trial curve. The CSV tables retain scaled and raw losses, signed shortfall/PnL, tail outcomes, transaction costs, turnover, positions, sample counts, lineage and available runtime/cost metadata. Missing observations remain missing. Positive domain improvement means baseline MSE minus method MSE; compare methods within each domain. Easy/hard raw levels alone do not establish algorithm quality.

All seven lineages reached the common budget; no arm is missing. The observed hard-market specialization does not establish broad improvement.

Selected-checkpoint pricing sensitivity status: complete. See selected_pricing_sensitivity.json and its referenced paired-outcome report. This uses the same 32 fixed monitor episodes per domain for all seven frozen checkpoints; it does not bound pricing error on the full datasets or rare tails.


The final paired check repriced 32 fixed monitor episodes per domain for all seven frozen policies. MSE changed 0.022%–0.978%; baseline-relative gaps changed 0.083%–20.13%, with no sign reversals across 18 comparisons. Direct quadratures agreed within 5.01e-15; maximum interpolated-versus-direct mark differences were 1.90e-4, 1.15e-4 and 3.25e-4 for base/easy/hard. Hard-domain gap changes were +1.13e-7 for SOAR mixed and -1.63e-7 for hard reference. Those are material relative to full-test gaps of 5.96e-7 and 1.92e-7, although effects cannot be transferred between samples. This bounded check does not certify full-test differences, rare tails, or the effect of retraining with exact prices. See [paired report](pricing_sensitivity/20261018T022347Z-f2135dd7/report.json).

Training price-range counts and maximum grid excursions are sampled block diagnostics at roughly 30-active-second intervals, not cumulative training totals. Counts refer to path-time pricing states, not episodes. Every block checks synchronized price finiteness before learning; failed fallback convergence/bounds checks produce nonfinite prices and stop execution. Nevertheless total training fallback frequency and maximum training excursion were not recorded. Evaluation-dataset out-of-bounds masks are complete, and repeated evaluations recount the same fixed states. Logged zeros must not be interpreted as proof of zero training fallback use. See [telemetry scope review](telemetry_scope_review.json).

## Limitations

One training seed (20261017); narrow tuning (generator learning rate .001 only; the optional .01 candidate was uniformly omitted for all adaptive methods by the predeclared storage/export reserve check, not by outcomes); synthetic common-support shifts in a finite 12-regime Heston menu. Numerical pricing/interpolation and time discretization are approximations. Pricing validation must be assessed against measured between-method differences. The recurrent policy is history conditioned; no separate ICL mechanism is established. Hard-only teacher updates omit easy target reward, but configuration selection uses easy development data. Mixed feedback has twice as many distinct target episodes. Intrinsic selfplay is development-selected. Financial adaptations of DPG, SOAR and selfplay are not literal language-model reproductions. No independent-seed uncertainty claim is supported.

## Cost and release verification

Campaign completed from 2026-10-17T21:54:00+00:00 to 2026-10-18T02:29:46.975528+00:00 in **4h 35m 46s**, within the eight-hour limit. All seven selected lineages completed 45 active training minutes. Report-worker export and provider shutdown confirmation succeeded; there are **zero active campaign instances**, zero unreported rental IDs and zero outstanding reservations. No standalone volumes were created; instance-attached storage was released with the destroyed instances.

Provider-reported campaign charges are **USD 3.681** as of 2026-10-18T02:28:09.203212+00:00, below the USD 25 cap. Charges include the provider's reported GPU/disk/bandwidth amounts, are rounded and may settle after shutdown. The separate conservative control-plane total is USD 12.579, including safety allowances; it is not an additional charge. See [per-worker charges](provenance/provider_worker_costs.csv), [provider snapshot](../budget/provider_charges_final.json), [budget ledger](../budget/ledger.jsonl), and [final signoff](campaign_signoff.json).

Original table cost fields describe cumulative evaluation-worker estimates, not allocated training/tuning cost per trial, and must not be summed across domains or checkpoints. Shared setup, compilation and transfer costs remain at worker/campaign scope. The [provenance supplement](provenance/checkpoint_evaluation_provenance.csv) distinguishes each checkpoint's creator commit/source hash from the evaluator build; the original per-domain source columns identify the evaluator. Its [scope and hashes](provenance/cost_and_provenance_scope.json) document these distinctions.

GPU checks passed 20/20 unit tests, all-seven integrated smokes and production profiles, actual selfplay resume and SIGTERM checkpoint recovery. All 169 retained checkpoints have final evaluations in all three domains. Strict worker-side checks validated raw episode schemas, counts, finiteness, PnL signs and recomputed summaries. The [recovery audit](final_recovery_audit.json) verified 169 checkpoints and 610,229 history references; 577 missing historical source working files were reconstructed from retained, verified Git objects across 12 snapshots ([evidence](materialize_sources_applied.json)). No historical scientific bytes were changed. Failed development checks and no-instance stale-offer attempts remain in worker/queue/budget records, with exact replacement provenance retained.

## Artifacts

- final_metrics.csv: all seven selected-arm slots with per-domain improvement.
- all_trials.csv: every retained trial/checkpoint and its status, including failures.
- all_events.csv: all available training and evaluation events.
- episode_losses.npz: lossless merged named raw episode arrays.
- array_integrity.json: worker-side array shapes and finite counts.
- raw_event_validation.json: strict event/raw schema, count, finiteness, sign, summary and aggregate checks; explicit failures remain recorded.
- selected_pricing_sensitivity.json: bounded frozen-selected pricing comparison and paired-archive reference.
- figures/: selected-lineage monitor/selection curves, separate all_trials_* curves, all-run generator trajectories, and easy/hard final outcomes.
- artifact_audit.json: metadata completeness failures, if any.

- campaign_signoff.json: final duration, budget, shutdown and validation signoff.
- provenance/: separate checkpoint/evaluator provenance and provider per-worker charges.
- final_recovery_audit.json: checkpoint/history/source byte integrity and recoverability evidence.
- telemetry_scope_review.json and screening_gate.json: retained supervisor gate and telemetry-scope evidence.
- report.worker_generated.md: original worker-generated report before supervisor interpretation/signoff.
