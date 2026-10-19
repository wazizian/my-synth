# Remote source snapshot

Read-only snapshot of `ssh hetzner:/home/jump/deep-hedging`, preserving relative paths. Final remote re-inventory: **2026-10-19T12:31:29Z**.

- **5,319 files; 325,226,087 bytes**, all SHA-256 verified against the remote inventory.
- A second full remote inventory found **zero added, removed, or content-changed selected files** during transfer.
- Includes every Python source file, historical submitted worker and source snapshot, tests, shell scripts, dependency/lock files, configuration JSON/YAML/TOML, Markdown/Tex documentation and text/XML/SHA-256 records. Large generated Python workers are preserved exactly, including embedded checkpoint literals.
- Extensionless files were inspected for executable shebangs; none were additional source scripts.
- `control-plane/` additionally preserves project-associated `/home/jump/campaign-control` source/configuration/recovery records and `/home/jump/*.md` plans and AGENTS instructions; its manifest maps each local file to its absolute remote origin.

Excluded: Git metadata, virtual environments, caches, bytecode, numerical arrays, model/checkpoint binaries, training histories, logs, rendered figures/PDFs, raw CSV numerical exports, SSH credentials/known-host files and unrelated home directories. Four remote tar.gz archives were inspected: all contained only .pkl training-history snapshots, no source. Existing result downloads remain separately in `hetzner-results-20261019/`.

Original remote `README.md` is unchanged. Snapshot documentation and verification use `PULL_*` / `CURRENTNESS_*` filenames; `manifest.json` records the initial full file list and hashes. `CURRENTNESS_INVENTORY.json` records the final remote source inventory, including mtimes. `CURRENTNESS_DELTA.json` is empty. `PULL_VERIFICATION.json` records the local all-file hash audit. The additional control-plane manifest and verification are in its directory.

No remote files were changed. No numerical computation, training, or tests were launched. Transport used read-only file inspection, hashing and in-memory compression; no remote archive copy consumed disk.

