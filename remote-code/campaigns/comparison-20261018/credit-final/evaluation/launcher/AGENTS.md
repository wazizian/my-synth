Use this directory for the user's Vast.ai GPU jobs. Read USAGE.txt for commands,
limits, and recovery. The user prefers uv for all Python environment and package
management. Do not invoke pip or seed it in new environments; uv's `uv pip`
commands are acceptable and run entirely inside uv.

- Credentials remain at ~/vast_ai_api.txt and ~/.ssh/id_ed25519_vastai (+ .pub).
  Never print them or upload private credentials to a rented instance.
- Use job.py for CUDA JAX/jaxlib/Equinox jobs; smoke.py is the original PyTorch
  hardware test and supplies the shared rental/cleanup functions.
- Default hardware is one verified RTX 4090 with 24 GB VRAM. Respect the launcher
  price cap and runtime limit. Do not leave an instance idle after finishing.
- One managed job at a time. If active.json exists, inspect/recover its instance
  before starting a new rental. Use smoke.py --cleanup for explicit cleanup.
- Always preserve metrics/checkpoints under JOB_OUTPUT_DIR, JOB_METRICS_DIR,
  and JOB_CHECKPOINT_DIR. Flush metrics. Write checkpoints to temporary names
  containing .tmp, then atomically rename them to completed names.
- Use --require-artifact for the actual job's expected metric/checkpoint globs.
  Inspect sync-status.json, artifacts.json, and the fetched files before reporting
  success. A live job failure can still have valuable locally synced artifacts.
- Confirm instance destruction through the API before reporting completion.
  If cleanup cannot be confirmed, retain active.json and report the instance ID.
- Run output remains in runs/<timestamp>-<id>/ on this Hetzner server. Never
  remove prior runs/checkpoints as part of routine cleanup.
- Document runtime/package changes in USAGE.txt and validate substantive GPU
  setup changes with a short real job when the user has authorized rentals.

Known platform details: list instances through /api/v1/instances/ with pagination;
other operations currently use v0. SSH uses -F /dev/null because the host's system
SSH config has invalid ownership/permissions. Instance startup creates
/root/.no_auto_tmux so batch SSH does not open an interactive tmux session.
