# Credit hedging source handoff

See [implementation and run instructions](credit_hedging/README.md), [audit](credit_hedging/AUDIT.md), and [next experiment plan](credit_hedging/PLAN.md). Run commands from this directory on a suitable numerical worker. SOURCE_MANIFEST.json records verified source hashes. No dependencies were installed or jobs launched by this upload.

## Credit-curve campaign implementation

The upload manifest describes the original handoff, not subsequent edits. An
untouched, verified copy is retained in
`../campaigns/credit-curve-20261018/source-original/`. The authoritative executed
plan snapshot, frozen design, source hashes, ledger and progress records are in
that campaign directory. The older audit above remains a historical one-factor
audit; it does not certify the current campaign's results.

`curve_market.py` adds independent slow and fast positive-intensity OU factors.
Simulation and pricing share a finite-state daily transition law. Separate
factor matrices avoid constructing a dense joint transition matrix. Production
price tables use one preallocated maturity buffer, retaining float64 arithmetic
and the numerically validated grid. The repaired full hard grid (1,201 by 801)
was built successfully on an RTX 4090 and matched the independent reference at
the tested states. All 39 current GPU tests passed during that repair validation.
This is not a uniform state-space or continuous-time error guarantee.

An informed dealer supplies executable par quotes and held-vintage marks. The
policy receives those quotes, time and inventory features; it receives neither
factor labels nor bond marks. Quotes can reveal factor information. The original
daily-coupon vintage ledger and cash accounting remain shared across domains.

`diagnostic_policies.py`, `economic_screen.py`, `audit_economic.py` and
`economic_report.py` implement the development-only feasible-policy screen,
paired episode intervals, raw-record checks and report artifacts. The frozen
gate determines whether learned specialists are warranted. No benchmark success
or adaptive result should be inferred from infrastructure tests.

Numerical execution belongs on GPU workers. The campaign's `package_worker.py`
creates the single-file launcher payload with source/configuration hashes. The
payload preserves failed outcomes and waits for verified exports on either
success or failure. Once the launcher has written its run metadata, start the
supervisor verifier with:

```sh
uv run --no-project python ../campaigns/credit-curve-20261018/verify_export.py /absolute/path/to/launcher/run
```

The verifier sends the export acknowledgement only when all immutable files in
the worker's manifest match their local SHA256 hashes. The launcher then performs
its final sync and provider-confirmed shutdown. Keep the verifier and launcher
handles in the campaign progress record; a state file alone is not evidence that
a process is still running.

The current screen retains all daily cash/trade/position records and paths, full
vintage traces for eight representative episodes, and mean vintage surfaces for
all episodes. Its `mean_turnover` metric counts opening purchases/sales, excluding
final liquidation notional; `mean_cost` includes final liquidation costs.
