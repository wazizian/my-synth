# Autonomous four-H100 pilot

Run this **on the allocated GPU node**, from the project root using the GPU
Python environment. It does not rent a machine or install dependencies.
No campaign was launched while writing this launcher.

```bash
python -m neural_sde_hedging.four_gpu launch \
  --output /workspace/runs/neural-sde-pilot-001 \
  --pilot-hours 2 \
  --hourly-rate "$TOTAL_NODE_USD_PER_HOUR" \
  --max-cost "$COMPUTE_BUDGET_USD" \
  --export-to 'archive-host:/absolute/path/neural-sde-pilot-001/' \
  --cleanup-argv '["/absolute/path/delete-this-instance"]'
```

Replace the export destination and deletion executable before use. Create the
remote archive directory first and configure noninteractive SSH authentication.
The deletion executable must delete **this allocation through the provider API**,
be safe to repeat, and return zero only after deletion is confirmed. Keep tokens
in its protected environment/configuration, not in command arguments. `poweroff`
does not necessarily stop rental billing. Configure a provider-side expiry or
an external control-plane deletion timer too: an on-node watchdog cannot survive
node failure. Do not use this with a cleanup command that merely prints success.

The node needs exactly four H100s, `nvidia-smi`, `rsync`, Python/pip and the
already installed GPU dependencies used by the smoke run: JAX CUDA 13 0.11.2,
Equinox 0.13.8, Optax 0.2.8, Chex, NumPy and pytest. The tested package freeze is
in `gpu-smoke-20261019/evidence/runs/20261019T140245Z-5c010296/outputs/requirements.lock`.
No extra packages are installed by the launcher. Output parent directory must
exist. The archive host must have sufficient storage; Hetzner's nearly full
control-plane disk is unsuitable without making room first.

## Execution

1. Validate four H100s, positive budgets and at least 10 GiB free; snapshot source,
   package versions, GPU inventory and file hashes. Reject reused output paths.
2. Start a detached watchdog and detached supervisor; both survive SSH logout.
3. Require an initial artifact export, then run the eleven GPU smoke cases on
   GPU 0 (ten-minute test deadline).
4. Run all four arms concurrently for sixteen calibration rounds, with a
   fifteen-minute soft budget plus two-minute hard grace. Any failure, nonfinite
   progress, missing checkpoint or incomplete calibration stops the campaign.
5. Start fresh pilot states on the same four GPUs, with the same configuration
   and seed. Calibration is not included in pilot training or used to choose a
   different learning rate. Each pilot gets two hours by default, including its
   compilation and checkpoint I/O, plus a two-minute hard grace for the last round.
6. Export artifacts every minute, summarize, export again, and invoke provider
   deletion. Three consecutive export failures, low disk, an artifact cap or any
   failed worker stops all remaining workers and triggers final export/deletion.

| GPU | Arm | Training distribution |
| --- | --- | --- |
| 0 | Baseline | Fixed initial neural-SDE prior; teacher never updated |
| 1 | DPG | Learned neural-SDE teacher, reset learner attribution |
| 2 | SOAR | Learned neural-SDE teacher, branched learner/promotion |
| 3 | Self-play | Learned neural-SDE teacher with replay and mutations |

One seed (`20261017`), identical GRU hedger initialization and architecture,
126-day horizon, capped-call payoff (`--payoff-cap 1`), entropic gamma 10,
learner LR 0.0003, teacher LR 0.001, 32-update learner warmup, weight decay 0.01,
gradient clipping 1, four groups of 32 paths, eight inner updates and 1,024 fresh per-round
feedback paths, with 2,048 fixed validation paths evaluated every 16 rounds. These are starting settings, not tuned hyperparameters.
To use a frozen per-method configuration from the GPU validation study, add
`--hyperparameter-profile /path/selected-configurations.json`. Review its
independent confirmation results first; development selection alone is not
evidence of improvement. The profile is embedded in `campaign.json` and applies
identically to calibration and pilot phases.
Each arm has one visible GPU, selected by UUID; this is independent-job
parallelism, not distributed training. Method-specific compute differs.

Default feedback consists of fresh per-round draws from the initial neural-SDE prior,
shared across arms. This is a demonstration target. For an intended scientific
target, pass both `--feedback-paths /path/training.npz` and `--validation-paths /path/validation.npz`; the file is copied and
hashed, must contain `states[episodes,127,3]`, and must not be the sealed final
test set. No final test set is opened by this pilot.

## Outputs and limits

`campaign.json` contains the absolute deadline, configuration and conservative
compute cost envelope. `calibration-summary.json` and `pilot-summary.json`
contain first/last risks, reward scales and mean post-compilation round times.
Each method has `progress.jsonl`, `latest.pkl`, configuration and status files;
self-play also saves historical learners. `*.log` captures GPU tests, training,
supervisor, watchdog and cleanup. The exporter includes logs and checkpoints,
excludes temporary files and never deletes archive-host files.

Calibration only gates **finite execution and throughput**, not learning quality.
Inspect entropic/CVaR/terminal-standard-deviation curves before interpreting
results. Daily marked hedge-error standard deviation is not available in this
adapter; it is not replaced by another metric. No plots or sealed evaluation
are generated by this launcher.

The supervisor enforces a wall-clock compute envelope using the supplied total
node hourly rate, including setup, shutdown grace and deletion retries. This
does not meter provider billing, storage or network charges. Successful cleanup
depends on your deletion executable and provider availability. The independent
watchdog retries deletion at the absolute deadline plus three minutes even if
the supervisor dies. The external/provider expiry is the backstop for node loss.

Training commits checkpoints every round and fails on incompatible resume.
The supervisor does not blindly restart failed numerical jobs or change their
hyperparameters. To recover an interrupted supervisor **before its original
deadline**, run the snapshot supervisor with the existing configuration:

```bash
cd /workspace/runs/neural-sde-pilot-001/code
python -m neural_sde_hedging.four_gpu supervise \
  --config /workspace/runs/neural-sde-pilot-001/campaign.json
```

Do not restart it after provider deletion has begun. The campaign lock prevents
two supervisors; training locks prevent duplicate writers. Original time budgets,
source and configuration stay fixed. Run recovery detached using your node's
service manager if needed.

Validation of the new launcher and baseline branch: Ruff and `py_compile` only.
Their numerical GPU validation occurs in calibration; the preceding eleven-case
suite covers the previously smoke-tested adaptive methods.
