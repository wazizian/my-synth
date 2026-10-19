# Changes and stopped validation — 2026-10-19

Stopped at the user's request. Both campaign GPUs (56763444 and 56765799) were
confirmed deleted through the provider API. The validation supervisor, readout
waiter and independent cleanup timer are stopped. Four preexisting stopped
instances were left untouched. The supervisor's failed terminal state reflects
the requested interruption, not a failed numerical test.

## Code changes

### Four-H100 launcher work

- `neural_sde_hedging/four_gpu.py`: added an autonomous launcher for four
  independent jobs: baseline, DPG, SOAR and self-play, one GPU per job.
  It snapshots source/configuration, detaches a supervisor and cleanup watchdog,
  runs GPU tests and calibration before the pilot, exports artifacts, records
  GPU utilization/memory and enforces deadlines, disk limits and a compute-cost
  envelope. Provider deletion is supplied as an explicit command.
- Added optional `--hyperparameter-profile` support for per-method learner LR,
  teacher LR, items per group and warmup. Profiles are checked and embedded in
  campaign configuration. No completed validation profile was produced.
- `neural_sde_hedging/FOUR_GPU.md` and `README.md`: added launch instructions,
  cleanup/export requirements and limitations. No four-H100 job was launched.

### Training and diagnostics

- `neural_sde_hedging/training.py`: added `baseline` mode. The existing GRU32
  learner trains on fresh paths from the fixed initial neural-SDE prior; the
  baseline teacher is never updated.
- Added per-round reward mean, standard deviation and maximum absolute reward;
  raw teacher gradient norm; and feedback entropic effective sample size.
- All arms use the existing GRU32 architecture and initialization seed.
  Training defaults remain learner LR 0.0003, teacher LR 0.001, warmup 32,
  four groups of 32 paths and eight inner updates. Alternative values were
  trial configurations, not a replacement of the defaults.
- `tests/test_neural_sde_gpu.py`: added the baseline to the full training/resume
  cases, including an assertion that its teacher parameters stay unchanged.

### Hyperparameter study

- `neural_sde_hedging/hyperparameter_validation.py`: added one-seed screening
  and independent confirmation, using a capped call and the initial-prior target.
  Planned 16 screens of 64 rounds, followed by selected/default configurations
  for 256 rounds. Learner rates: 0.0003/0.001; batch sizes: 128/256;
  teacher rates: 0.0001/0.0003/0.001; DPG warmup: 8/32.
- Separated feedback (512 paths), development selection (2,048 paths) and
  confirmation (8,192 paths). Added risk metrics, paired path-bootstrap
  intervals, learning-curve/CSV output and bounded full-state recovery exports.
- Added a local archive-handling refinement to avoid duplicating the largest
  selected self-play bank. This refinement was **not executed**: a guard rejected
  updating the already-started worker, preserving its submitted source snapshot.
- `gpu-hyperparams-20261019/`: added packaging/worker scripts, copied Vast
  orchestration helpers and recorded the study in `PROTOCOL.md`. The remote
  retry helper filters for faster networking and standard 24-GiB GPUs after
  the first host timed out downloading CUDA dependencies.

### DPG diagnostic, prepared but not run

- `neural_sde_hedging/teacher_readout.py`: added a frozen-generator comparison.
  Identically initialized hedgers would receive matched 96/512-update budgets
  using the prior, default DPG teacher and selected DPG teacher, evaluated on
  a new 8,192-path set. No teacher updates or hyperparameter reselection.
- `gpu-hyperparams-20261019/launch_readout.py`: added a detached waiter to launch
  that diagnostic on the existing GPU once DPG confirmation checkpoints existed.
  It was stopped while waiting; the diagnostic never started.

## What was verified

- Ruff and `py_compile` passed for changed Python files.
- **12 GPU integration cases passed**, including baseline, adaptive-method
  training/resume and unchanged baseline teacher checks.
- **12 screening trials completed**: four baseline, six DPG, two SOAR.
  The third SOAR screen was interrupted. No self-play hyperparameter screen,
  256-round confirmation or frozen-generator readout completed.
- The four-H100 launcher/profile path and readout script have static checks only.
  These were not numerically validated by the integration suite.

## Preliminary findings — not final recommendations

- At 64 rounds, development risk favored learner LR 0.001 and batch 256 over
  the original baseline settings. Longer independent confirmation is missing.
- DPG's eight-update reset learner stayed near initialization. Warmup 8 helped
  relative to 32; teacher-rate differences were small. Its raw learner risk
  should not be ranked against a continuously trained baseline without stating
  the different learner budgets. This motivated the unexecuted readout check.
- SOAR produced valid promotions and finite results in the completed trials.
  Its screening and confirmation remain incomplete.
- **The defaults are not yet fully validated or replaced.** No claim of
  convergence, superiority or performance on easy/base/hard or credit follows.

## Saved evidence

- Local: `gpu-hyperparams-20261019/partial-results/` (158 SHA256-verified files).
- Portable archive: `gpu-hyperparams-20261019/partial-results.zip`, 2,441,526 bytes.
  SHA256: `6d77833dbdbe506f6d127fbb4e61a0f569f58ba598f67c9765f4dc19d9a7d0fb`.
- Remote: `/home/jump/neural-sde-hyperparams-20261019/`.
- Active attempt: `runs/20261019T174738Z-f7c0546f/outputs/` contains curves as
  JSONL, completed trial results/configurations/learner parameters, GPU tests
  and rotating recovery checkpoint. `STOPPED.json` records provider cleanup
  confirmation and successful final export.
- The earlier attempt `20261019T172908Z-376a2597` contains the dependency-download
  timeout evidence. No numerical trial ran on that first worker.

The saved submitted worker and its source manifest are authoritative for what
actually executed; later local launcher/archive/readout additions are distinguished
above. No numerical computation or tests ran locally or on Hetzner's control plane.
