# Adaptive deep hedging across easy and hard markets

This repository implements the synthetic seven-arm comparison in `/home/jump/plan.md`: `baseline`, `hard_reference`, `dpg_hard`, `soar_hard`, `selfplay`, `dpg_mixed`, and `soar_mixed`. The recurrent Equinox hedger trades spot and a coherently priced Heston call, pays proportional transaction costs and liquidates both positions. Every method minimizes squared terminal shortfall normalized by inception spot. Base, easy and hard distributions are frozen mixtures over the same 12 regimes.

The small supervisor is a control plane. **Do not install/import the numerical stack, run pytest, generate data, train, evaluate or plot there.** Numerical entrypoints assert a GPU backend; CPU fallback is a hard failure. All numerical work below runs on a supervisor-authorized GPU worker. Only the supervisor may rent/release workers or reserve spend. Total campaign limits are USD 25 and eight hours, with USD 3 reserved for recovery/export/shutdown.

## Worker setup and validation

Use the pinned dependency environment from `pyproject.toml` / `requirements-worker.txt`; preserve the worker's actual dependency export. Set JAX CUDA memory limits before starting Python, isolate one training process per GPU, and use a private compilation cache.

From the repository root on the GPU worker:

```bash
python scripts/validate_worker.py
python -m pytest tests/test_market.py tests/test_hedger.py tests/test_checkpoint.py tests/test_teachers.py
python scripts/prepare_market.py
python scripts/prepare_data.py
```

Inspect pricing and simulation validation before training. The cached pricing table, immutable dataset manifests and objective scales must be shared by content hash across workers. Validation failure blocks scientific runs; never replace coherent Heston marks with Black–Scholes marks. `src/protocol.py` and `configs/` define dataset roles, seed namespaces, learner settings and the seven methods.

## Control-plane orchestration

The supervisor maintains `SUPERVISOR_STATE.md`, `TASKS.md`, `DECISIONS.md`, `runs/manifest.jsonl`, `budget/ledger.jsonl` and `queue/state.json`. Consult the current state before resuming; never relaunch a job merely because a log observation timed out.

A source snapshot can be packaged without importing numerical libraries:

```bash
python scripts/package_job.py --entry scripts/validate_worker.py --output /tmp/validation_job.py
python scripts/launch_worker.py --worker validation --job /tmp/validation_job.py --seconds 600 --dry-run
```

Actual worker launch requires supplied infrastructure credentials and central budget reservation. `--dry-run` estimates the reservation and creates no rental. Running a job and terminating its Python process does not release the rental: confirm provider terminal billing state after artifacts have been exported. Worker commands must carry hard deadlines and save full resumable state before shutdown.

## Evaluation, checkpoints and reporting

Feedback, monitoring, selection and final tests have disjoint immutable manifests. Teacher updates receive only their authorized feedback: hard, balanced fixed-scale easy/hard, or intrinsic selfplay. Selection uses the same frozen balanced objective for adaptive candidates. Freeze `results/frozen_selection.json` before opening final tests. Preserve final evaluations for every retained trial checkpoint, including rejected and stopped configurations. Failed trials require explicit reasons and null measurements, rather than deletion or fabricated scores.

Checkpoints save arbitrary complete numerical state using atomic Equinox serialization plus JSON metadata. Rebuild the same skeleton/environment when restoring and validate dataset/pricing/config hashes. State must include learner/generator optimizer moments and counts, all PRNG streams, exact selfplay history, reward baselines, counters, runtime/cost state and lineage. `latest.json`, `previous.json`, `best.json` and `final.json` point to immutable bundles. The GPU resume test verifies that the next Adam update after restoration is unchanged.

After all workers' retained runs have been synchronized to one GPU worker:

```bash
python scripts/make_report.py --root /path/to/campaign
```

The report produces `results/final_metrics.csv`, `all_trials.csv`, `all_events.csv`, `episode_losses.npz`, `array_integrity.json`, figures and `report.md`. It retains raw and scaled MSE, signed PnL/shortfall, tail outcomes, costs, turnover and available runtime/lineage fields. It never runs selection or modifies teacher state. Missing results remain visibly missing.

The metadata-only completeness audit may run on the supervisor after export:

```bash
python scripts/audit_artifacts.py --root /path/to/campaign
```

Exit status 1 means requirements remain missing; inspect `results/artifact_audit.json`. The metadata audit does not replace GPU-side array integrity checks. Supervisor signoff must fill actual cost, duration and confirmed instance/volume shutdown status in the final report.

## Interpretation

This is a single-seed exploratory synthetic study with narrow tuning and common-support mixture shifts. Compare methods within each domain, report base retention independently, and describe easy/hard specialization honestly. Mixed teacher feedback uses more target episodes; intrinsic selfplay still uses development selection. Pricing approximation error must be small relative to reported differences. No separate ICL mechanism, unseen-model robustness, independent-seed uncertainty or favorable outcome is assumed.

## Completed campaign

All seven selected arms completed 45 active minutes; 169 checkpoints across 21 retained lineages were evaluated on all three final domains. GPU raw-array validation, artifact completeness and checkpoint/source recovery audits passed. The single-seed experiment found no broad adaptive improvement: baseline had the best base/easy and balanced MSE; SOAR mixed had a small hard-MSE advantage with worse base/easy performance and a numerical-pricing caveat.

Start with [the final report](results/report.md), [selected metrics](results/final_metrics.csv), [all retained checkpoints](results/all_trials.csv), [figures](results/figures), and [campaign signoff](results/campaign_signoff.json). The report records actual provider-observed cost, campaign duration and confirmed rental shutdown, with explicit billing scope. [Checkpoint/evaluator provenance and per-worker charges](results/provenance) supplement the original table fields. All runs, immutable datasets, exact selfplay history and verified historical source snapshots remain available locally. No further worker launch is needed to review the completed artifacts.
