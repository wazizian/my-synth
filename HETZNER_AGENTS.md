# Instructions for ChatGPT / Codex on Hetzner

These rules apply to work under /home/jump, including deep-hedging and campaign-control. Hetzner is a control plane with only one vCPU.

## No local computation or tests

- Never run numerical computation, simulations, training, evaluation, benchmarks, plotting, data processing, or tests on this Hetzner host. This includes smoke tests, pytest, dependency-import checks and CPU fallbacks, however small they seem.
- The only permitted local code checks are Ruff and Python's py_compile, scoped to the changed files. Do not invoke project code to validate it locally.
- Run all other computation and tests on a provisioned GPU worker, within the user's existing authorization and budget. A missing GPU is not permission to run on Hetzner's CPU.
- Lightweight editing, file inspection, SSH, transfers and orchestration are allowed. Keep control scripts lightweight; never import numerical libraries or execute workloads as part of orchestration. Avoid resource-heavy parallel processes on this one-vCPU host.

## Disk space

- Check free space before dependency setup, large transfers or artifact exports, and monitor it during campaigns.
- Keep environments, package/model caches, datasets, numerical intermediates and full raw results on workers or appropriate external storage whenever possible. Transfer only necessary artifacts to Hetzner and avoid duplicate copies.
- Use bounded logs, incremental exports and explicit artifact manifests. Check expected sizes before transfers; avoid unbounded log growth and repeated full archives.
- Preserve required results and recovery checkpoints. Delete only known disposable files or files explicitly authorized for cleanup; verify required exports before removing a worker or its only artifact copy.

## GPU jobs must survive SSH disconnections

- Never run a GPU job with its lifetime tied to an interactive SSH connection, a foreground SSH command, or a local tool session.
- Start the worker job under a durable supervisor (for example systemd, a batch scheduler, tmux, or correctly detached nohup with stdin closed and stdout/stderr redirected). Verify it remains running after the launch connection exits.
- Persist each job's run ID, provider instance ID, worker address, supervisor/session name, source/config hashes, log location, checkpoint location and output location before relying on the job.
- Write progress and exit status to durable files. Save periodic atomic checkpoints containing model, optimizer, RNG and training progress; support resume without resetting the run or its budget accounting.
- Use reconnectable, bounded monitoring. An SSH timeout or lost tool session is not evidence that the worker stopped. Reconnect and inspect the persisted job state before retrying, relaunching or renting another worker.
- Make launch/recovery idempotent: check for an existing run first and prevent duplicate workers or jobs. Resume from verified checkpoints after actual failures.
- Budget limits, runtime watchdogs and safe completion/cleanup must operate independently of the SSH connection. Do not let the loss of a monitoring session leave an unbounded paid instance running.
- Export incrementally, verify final artifacts, record terminal status and confirm provider shutdown when the job is finished. Allow bounded export recovery before cleanup; never leave a paid worker waiting indefinitely for a disconnected client to acknowledge an export.
