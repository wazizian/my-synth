# Shared optimizer migration before the eight-arm comparison

The unchanged credit learner uses clipped Adam with eps_root=0 and zero-initialized
policy heads. GPU audit on RTX4090 instance56621676 found nonfinite native DPG
derivatives through its eight-update, 160-path, 1260-day learner block. This is
not a claim that native differentiation through the unchanged optimizer works.

The same full-shape audit with Adam eps_root=1e-16 produced finite derivatives
and finite DPG, SOAR and selfplay teacher updates. Exact all-one DPG post-state
equivalence, weight-derivative sums and actual trainer checkpoint replay passed.
The warm DPG block took 8.065829seconds; peak memory was 13,703,975,424bytes.
All original production-domain terminal ledger comparisons passed at 1e-12.
The final exported source/config hashes, audit.json and artifact inventory are in
`audit/launcher/runs/20261018T213139Z-b6d9d259`.

The root controller approved applying eps_root=1e-16 identically to all eight new
comparison arms: baseline, easy_reference, hard_reference, dpg_hard, soar_hard,
selfplay, dpg_mixed and soar_mixed. The shared architecture, initial parameter
values, trading laws and gamma remain unchanged. This changes the Adam
denominator for every arm; old eps_root=0 specialist checkpoints are discovery
evidence only and are not treated as matched final comparison references.

The actual updated trainer passed full GPU execution/replay checks; exact hashes
and scope are recorded in `audit/APPROVAL.json`. No custom differentiation rule
has been introduced. Benchmark admission is a separate scientific gate.

The selfplay preconditioner uses this same eps_root in its actual-Adam
denominator. A followup current-source GPU audit explicitly checks that identity;
the earlier default-zero-preconditioner result is not used to validate the
migrated selfplay formula. Current credit and option formula checks passed with
errors3.33e-15 and6.66e-16, respectively.
