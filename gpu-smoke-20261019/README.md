# Neural-SDE GPU smoke results — 2026-10-19

**Final result: PASS.** All checks ran on a rented RTX 4090, never locally or
on Hetzner's one-vCPU control plane. JAX/JAXlib 0.11.2, Equinox 0.13.8.

| Final stage | Result |
|---|---|
| GPU verification suite | 11 passed; 0 failures/errors/skips, 40.31 s |
| DPG, 126-day grid | 2 rounds complete; cross-entrypoint checkpoint load passed |
| SOAR, 126-day grid | 2 rounds complete; cross-entrypoint checkpoint load passed |
| Self-play, 126-day grid | 2 rounds complete; cross-entrypoint checkpoint load passed |

Final source: ten package/test files independently SHA256 matched the current
local source. Runs used one common initialization seed, identical GRU32 hedgers,
two groups × four paths, two inner updates, and explicitly capped call payoff.
These are integration tests, not evidence of convergence or superior hedging.
The ordinary-call population-entropic issue is not resolved by these tests.

The first GPU suite passed 9/11 and exposed unpicklable activation functions in
the self-play bank. Fixed by serializing executable SDE programs as architecture
metadata plus numerical leaves. A canonical imported checkpoint codec avoids
pickle references to `__main__`; separate Python processes successfully loaded
the final CLI-produced checkpoints for all three methods. Parameter/moment
partitions remain partitions on restoration.

Final evidence is in
`evidence/runs/20261019T140245Z-5c010296/outputs/`: `pytest.xml`, `pytest.log`,
`summary.json`, method logs/configurations/progress/checkpoints, GPU information
and dependency lock. Earlier attempts and failures are retained separately.

`evidence.zip` contains 132 files, 1,070,511 uncompressed bytes. Every extracted
file matches `evidence/DOWNLOAD_MANIFEST.json`. Archive SHA256:
`30b1f512f09354944e9e6b9dcb2865fff657e138f9f95b44abc71b2b276854f6`.

Budget cap was $1.25. Final provider snapshot reports **$0.307** across instances
56729103, 56731481 and 56738201, with **$16.2273 account credit** remaining.
Charges can settle after shutdown. All three smoke instances were confirmed
destroyed; no active smoke instances remain. Four preexisting stopped instances
were left untouched. Two stale-offer attempts created no instances.

The supervisor ran under user systemd with linger enabled, a worker runtime
timeout, periodic exports and a separate cleanup timer. The final unused timer
was stopped after provider-confirmed destruction. The optional worker-local
watchdog was not relied upon: provider token environment variables were absent
from the inspected SSH environment.
