# Hetzner results — 19 October 2026

Downloaded 136 files (17,779,674 bytes), verified against remote SHA-256 hashes in REMOTE_SHA256.json. The selected snapshot contains reports, metrics, training logs, plots and terminal evaluation arrays for the comparison and volatility-shock campaigns. Full checkpoints, per-step raw records, duplicated source trees and external self-play history archives remain remote; this is not a complete recovery backup.

- [Combined report](campaigns/comparison-20261018/FINAL_REPORT.md)
- [Credit report](credit-evaluation/report.md)
- [Credit LaTeX source](credit-evaluation/report.tex)
- [All-trial training curves](credit-evaluation/all-trial-learning-curves.pdf)
- [Specialist gaps](credit-evaluation/specialist-gaps.pdf)
- [Daily errors](credit-evaluation/daily-errors.pdf)
- [Terminal comparison](credit-evaluation/terminal-comparison.pdf)
- [Option discovery report](campaigns/volatility-shocks-20261018/option-discovery-report.md)
- [Option results](campaigns/volatility-shocks-20261018/results.json)
- [Recovery and limitations](campaigns/comparison-20261018/RECOVERY_AND_SCOPE.md)

Credit: no adaptive method clearly beats the own-domain specialists. Option: all three joint specialist gates failed; hard specialists improve, easy specialists do not reliably improve; adaptive option stage skipped, final test stream unopened. Intervals are conditional on one shared initialization. All campaign GPU instances were reported shut down.
