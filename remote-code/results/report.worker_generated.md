# Adaptive deep hedging exploratory campaign

Archive status: metadata completeness passed.

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

PENDING SUPERVISOR INTERPRETATION: After checking the frozen selected results and numerical validity, describe hard-only versus mixed within DPG and SOAR, easy/hard tradeoffs, base retention, intrinsic versus target-guided learning, and whether gains justify measured compute. Include unfavorable or inconclusive outcomes and any incomplete arms. No favorable conclusion is inferred automatically.

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

Inspect hard-only versus mixed separately within DPG and SOAR, including base retention. Compare intrinsic curriculum to target-guided methods together with their active compute and teacher overhead. A lower hard loss with higher easy loss is specialization, not broad improvement. This report does not infer a favorable result from incomplete data.

Selected-checkpoint pricing sensitivity status: complete. See selected_pricing_sensitivity.json and its referenced paired-outcome report. This uses the same 32 fixed monitor episodes per domain for all seven frozen checkpoints; it does not bound pricing error on the full datasets or rare tails.

## Limitations

One training seed; narrow tuning; synthetic common-support shifts in a finite 12-regime Heston menu. Numerical pricing/interpolation and time discretization are approximations. Pricing validation must be assessed against measured between-method differences. The recurrent policy is history conditioned; no separate ICL mechanism is established. Hard-only teacher updates omit easy target reward, but configuration selection uses easy development data. Mixed feedback has twice as many distinct target episodes. Intrinsic selfplay is development-selected. Financial adaptations of DPG, SOAR and selfplay are not literal language-model reproductions. No independent-seed uncertainty claim is supported.

## Cost and release verification

Supervisor must reconcile provider charges, elapsed campaign duration and terminal billing state before final signoff. See budget/ledger.jsonl and SUPERVISOR_STATE.md; process termination alone does not prove rental release.

Retained ledger events: 64. Provider reconciliation/shutdown signoff: PENDING SUPERVISOR VERIFICATION.

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
