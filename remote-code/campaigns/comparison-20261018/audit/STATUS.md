# Algorithm GPU audit

Only py_compile was run on Hetzner. All numerical checks run on managed GPU workers.

First attempt ask53061818 disappeared before rental; no instance was created.
GPU56620824 (RTX4090) ran the generic nonlinear entropic/replay/teacher checks.
The first full-credit attempt failed because a closed-over concatenated full
bond quote tensor required another 9.05GiB allocation. This was a memory/layout
failure before a full DPG step, not evidence about DPG scientific viability.
Final outputs were retrieved and provider destruction confirmed.
Run: `launcher/runs/20261018T212348Z-5257e221`.

The adapter now passes arrays dynamically and keeps only CDS quotes and issuance
bond prices for terminal training. Its terminal-only ledger calls the original
book_step with discarded dummy bond marks. It requires GPU equivalence checks
against the full original rollout; final daily evaluation still uses real marks.

Second audit launched on GPU56621676 (RTX4090, offer38381893, $0.4281/hour),
local session7849, worker timeout2400seconds. It tests production-domain ledger
equivalence, stronger finite differences, unchanged default Adam first, and only
diagnoses eps_root1e-16 if default Adam gives nonfinite full-shape derivatives.
No optimizer migration is authorized by this diagnostic alone.
Durable run metadata and progress live under `launcher/active.json` and
`launcher/runs/*/outputs/audit.partial.json`; final result is audit.json.

Second audit generic GPU checks passed: nondegenerate nonlinear entropic DPG
finite difference 0.00045777061795718166 vs autodiff
0.00045777061781878645. All-one actual carried-Adam post-state equivalence,
per-minibatch derivative sums, prior floor, SOAR EMA timing, entropic selfplay
JVP identity, round-zero optimizer preservation and checkpoint replay passed.
Full production credit-shape audit completed successfully. Default eps_root=0
gave nonfinite DPG derivatives; eps_root=1e-16 was finite. Production-domain
terminal ledger equivalence, all-one actual post-state equivalence, weight
derivative sums, all three finite teacher updates, actual trainer checkpoint/load
and exact replay all passed. Warm full DPG block: 8.065829seconds.
Final sync completed with six successful syncs and no failures. GPU56621676
destruction confirmed. See OPTIMIZER_MIGRATION.md for the shared approved change.

Actual updated eight-arm trainer execution/replay audit launched on GPU56622688,
offer42448592 ($0.4304/hour), local controller41975, timeout2400seconds.
This runs every arm at full production block/path dimensions with boundaries
[0,1] and selection128, then repeats the completed run to verify no extra active
time, no changed learner/teacher/EMA/PRNG state and exact selection replay.
Final artifact is trainer-audit.json; until then production trainer approval is
pending. Historical source/config hashes remain preserved per run.

The first actual-trainer audit on GPU56622688 passed baseline and dpg_hard
execution/resume, then exposed a selfplay historical-snapshot static-metadata
identity mismatch after a separate pickle load. The failure was exported and
worker destruction confirmed. Historical numerical arrays are now rebound to
the frozen current parameter-tree structure with layout checks. Selfplay's
preconditioner was also corrected to include the shared Adam eps_root1e-16.

Current-source audit: GPU56623182, offer48982788, $0.4644/hour,
controller87745, run20261018T214659Z-193844b4. All eight arms use independent
monitor128 and full completed-state replay. Selfplay runs35active seconds and
must exercise at least three outer rounds plus a real generator update. Its
actual-Adam entropic JVP identity is checked independently. The audit-only
stopping-rule extension happened while the instance was still loading, before
source upload/execution; remote job SHA was verified against the effective
submitted payload. Original payload and explicit migration manifest are retained.

All eight actual credit trainer arms passed execution, monitor and completed-run
replay on GPU56623182; selfplay exercised eight rounds and64updates. That worker
was exported and destruction confirmed.

Final current-source proof completed on GPU56624000,
run20261018T215418Z-abef1547. Credit baseline and selfplay passed; selfplay ran
14rounds/112updates with one trace per trainer-step function and exact replay.
Option baseline, dpg_mixed and selfplay passed with the frozen premium-aware
batch dictionaries; option selfplay ran34rounds/272updates. Independent monitor
streams, complete boundary checkpoints and zero-extra-training completed resume
all passed. Actual Adam selfplay JVP identity errors were3.33e-15 (credit) and
6.66e-16 (option). Final sync had7successes/0failures; GPU destruction confirmed.

`APPROVAL.json` records exact current implementation hashes and artifact paths.
This approves implementation validation only: specialist benchmark admission and
final performance claims still require their separate gate/evaluation evidence.
