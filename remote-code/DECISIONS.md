# Decisions

- Project isolated at /home/jump/deep-hedging; infrastructure reuses existing
  /home/jump/vast-tools only after budget and lifecycle review.
- Master seed 20261017, one seed throughout. No additional ablations.
- CPU stack initially matches proven GPU JAX0.11.2/Equinox0.13.8 pins.
- User asks execution of plan, whose delegation instructions authorize the
  three bounded implementation agents; supervisor retains integration/rental.

- 21:54 UTC restart of execution against explicitly revised seven-arm plan.
  Old implementation was incomplete, with no campaign rental or trial data.
  Eight-hour clock starts now and includes all revised development.
- Correct old $23/$2 caps to $22/$3; base dominant regime index3, easy index0.
- Repository git binary/metadata unavailable; retain deterministic source hash
  and source archive, with code_commit explicitly null until git is available.

- Current /home/jump/plan.md supersedes old CPU-preparation instructions; all numerical work is GPU-only. Existing 21:54 start/deadline retained conservatively. No previous numerical artifacts are accepted as validated.

- Provider quote check returned verified RTX4090 offers around $0.41–0.43/hour with explicit storage/transfer prices saved in budget/offers.json. First attempted offer vanished before creation; no rental was created. Retried using a fresh provider offer.
- Parallel jobs use isolated copies of reviewed Vast lifecycle tools under workers/<worker>, centrally reserved by src/budget.py. This implements the plan’s explicit multi-GPU authorization while retaining one managed job per worker state file.
- JAX official benchmarking/cache/installation docs reviewed: synchronized timings, first-call compilation separate, private persistent cache, pinned CUDA13 stack.

- GPU validation04 passed all20 numerical unit tests. The once-expanded Heston table passed its gate (max sampled error0.00018023, RMSE0.00004311, zero out-of-grid in12,288 validation episodes). However the independent hard final-test generation hit a rare out-of-grid state. Benchmark mixtures/keys stay fixed. A dual-quadrature float32 GPU fallback is being independently stress-validated; table identity and in-grid pricing arithmetic remain unchanged. No method training occurred before this correction.
- Runtime worker provenance is separate from immutable trial configs, permitting continuation on another matched GPU while preserving actual rental IDs/rates/timestamps.
- Source-only assumption of independent cleanup was insufficient in SSH environment; provider instance-local watchdog is installed at onstart. validation04 confirmed an armed watchdog status. On supervisor loss it stops compute and retains storage unless export acknowledged; normal control-plane cleanup still exports and confirms destruction. Source: https://docs.vast.ai/guides/instances/docker-environment .

Validation07 composite quadrature resolves reference aliasing (independent doubleprecision rules agree within9.42e-14), but float32 coarse/fine comparison rejects28/1440 stressstates (max1.24e-5). Keep gates; improve bounded fallback precision. No training permitted yet. validation06 stale offer caused no rental; validation07exported/destroyed.

Campaign handoffs preserve only our trusted generated compilation cache across matched RTX4090 workers with identical pinned stack. Files copied as opaque bytes on supervisor; eachworker usesprivate0700cache. Actual cachehit/lowering time remains measured and charged.

Storage correction after resident-table profile: selfplay323newrounds/9.966s,70,536bytes/snapshot projectsabout6.1GB actual2700s history;2xcombined-profile blockcap projects9.34GB. Raised bounded workerhistoryallowance8→12GiB (40GBworker disk), unchanged model/training. Futureartifactimports use immutablehardlinks and atomicmutablecopies; future rsync uses immutable-onlylinkbase. No-inplace. Oldterminatedworker duplicates and completedhandoff archives are reclaimed onlyafterbytehashverification and preservingallmetadata. Optional secondLR nowrequiresdiskproof including fullhandoff+2GiBreserve; omitalloptional candidates ifnotaffordable. This is an explicit postdiagnostic source migration; currentdiagnosticworkercontinues its pinnedsource.

Provider billing reconciliation uses official per-instance charge endpoint GET/api/v0/charges with UTCdayfilter andpagination (https://docs.vast.ai/api-reference/billing/show-charges), including GPU/disk/bandwidth. Provisional provider_charges_latest.json records onlycampaigninstanceIDs; conservative budgetreservations remain authoritative duringlivework tocoverbillinglag. No inference from presumedinitialcreditneeded.

2026-10-17T23:32:26.765132+00:00: Diagnostics completed and GPU56463419 exported/destroyed. Development gate passed from reference learning, all7 profiles and paired pricing sensitivity (MSE shifts0.066%-0.464%, gap shifts1.42%-2.32%, fixed32monitor/domain, no signchanges). Shared B256/K4/H64/Adam.001 settings frozen. Explicit source migration queue/source_migrations/20261017T233158Z.json records operational storage/transfers/allocator/report-integrity-only changes; model/teacher/pricing/data mathematics unchanged. Byte-identical immutable dedup reclaimed~819MB, preserving all paths and contents. Upcoming screens5adaptivearms on3workers; finaltests sealed.

2026-10-17T23:57:17.391818+00:00: All7arms completed600active seconds with validfinitecheckpoints; allscreenworkers exported/released. Screening evidence results/screening_gate.json. Applied operational exportgrowth migration queue/source_migrations/20261017T235706Z.json (only scripts/run_campaign.py), preserving allscientificsettings. Shared finalcap2700cumulativeactive seconds. Optional2ndLR storageproof fails; retainoneLRperadaptivearm. Nextphaseextensions reusesfullsavedstate.

2026-10-17T23:59:49.181260+00:00: Extensioncontroller session2515 running. Original extensions-0 failedbeforecreation staleoffer54132961, zerocharge/no training. Provider control script /home/jump/vast-tools/smoke.py now filters JSON blacklist /home/jump/campaign-control/stale_offer_ids.json; originalbytesbackup+change record retained outsideproject. Launched identical baseline/soar_hard/soar_mixed lease as extensions-retry0 instance56469209 session44970; externallease record /home/jump/campaign-control/extensions-retry0.json. Othergroups extensions-1 andextensions-2 live/disjoint. IMPORTANT controller2515 will eventuallyexit1 fororiginalextensions-0 failure; do notrerun itsarmjobs. After replacement44970 completes+export/release, import its3runs via run_campaign.copy_result and verify group_status; preserveoriginalfailureattempt record thenmarkextensions-0 recoveredcomplete linkingreplacement, setphasecomplete. Do this onlyaftercontroller2515 ends toavoidstateoverwrite. No finaltests opened.

Telemetry scope review: training pricing_out_of_bounds_count counts path-time states in sampled simulation blocks (~30-active-second log cadence), NOT cumulative training totals; max_grid_distance is likewise sampled. All blocks nevertheless check synchronized spot/call finiteness before learner updates, and fallback convergence/bounds failures become NaN and terminate. Evaluation dataset masks are complete; repeated evaluations recount the same fixed states. Final report MUST disclose that total training fallback frequency and maximum training excursion were not recorded; never infer zero fallback use from logged zeros. See results/telemetry_scope_review.json. No source edits or scientific changes.


Final closure 2026-10-18T02:29:46.975528+00:00

Finalization preserved worker report and added supervisor interpretation. External archive trimmer learned to verify safe immutable tar hardlinks; source working files recovered only from SHA1/SHA256-verified retained Git objects. These were artifact-recovery changes, not scientific-source changes. Provider charges and evaluator/checkpoint provenance are separated in results/provenance; no invented trial-cost allocation. Small hard-reference denominator is not used for gap closure.
