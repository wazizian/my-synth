# Learner/accounting/checkpoint review

Owned source audited against plan sections 5, 13 and 14. Public APIs unchanged.

- GRU uses only causal price history, public contract data and prior positions. Context lookup now accesses at most the current observation even for zero-filled inception features. Hidden state resets per episode; trading decisions precede subsequent increments.
- Accounting includes opening trade, each rebalance and liquidation exactly once. Public premium uses the base mixture mean long-run variance and is independent of hidden regime. Corrected its misleading comment (numerical convention unchanged).
- Existing carried Adam scan remains differentiable through normalized example weights and persistent moments. `eps_root=1e-12` protects second derivatives at zero moments.
- Outcome summaries now perform reductions on GPU rather than transferring full arrays for CPU aggregation; only compact summaries transfer to host. Raw arrays retain normalized signed shortfall/PnL, costs, turnover and positions. Added unit labels and diagnostic nonfinite counters.
- Atomic Equinox checkpoints preserve arbitrary state trees and typed keys with explicit metadata. Previous-pointer rotation now checks metadata integrity as well as state integrity. Caller must include generator optimizer, exact lookback bank, all keys/counters, cost/time/lineage state and immutable artifact hashes; serialization cannot invent omitted state.
- Added GPU-only resume test that compares the exact next optimizer step after restore, typed keys, snapshot bank and moments, plus previous-pointer recovery/corruption/hash checks and metric/raw-archive checks. Existing tests cover analytic accounting, no-future-action influence, reset, bounds and finite-difference carried-block derivatives.

Read official Equinox transformation and serialization documentation:
https://docs.kidger.site/equinox/api/transformations/
https://docs.kidger.site/equinox/api/serialisation/

Validation performed here: Python AST parsing only, all six owned Python files pass. No numerical imports, numerical workloads, dependency installations or rentals executed on the supervisor. Run `pytest tests/test_hedger.py tests/test_checkpoint.py` on the authorized GPU worker. Numerical correctness remains unverified until that succeeds.
