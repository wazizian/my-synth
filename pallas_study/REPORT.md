# Pallas training-speed study — 21–22 October 2026

Pallas can improve **option-hedger training updates**, especially at larger batches. The strongest completed default-precision analytic-GRU measurements were **1.23× at batch 128 and 1.54× at batch 1024**. Neural-SDE gains are conditional: the analytic MLP was slower for default-precision pathwise training; saved-backward kernels helped first-order training, particularly with highest matmul precision. **The tested credit workloads did not improve.**

The largest demonstrated SDE improvement needs no Pallas: vectorizing the fixed-path score objective over time reduced a full JAX update from **17.123 to 1.282 ms (13.36×)** at batch 128, default precision. With highest precision it reduced **29.547 to 1.344 ms (21.98×)**. The continuous-credit score already uses time vectorization.

## Recommendation

1. For option first-order hedger training, retain the analytic GRU as an opt-in candidate, with batch- and precision-specific checks. Do not infer a 54% whole-campaign improvement from a 1.54× update speed ratio: that ratio corresponds to about 35% less update time.
2. For fixed-path SDE likelihood fitting, prioritize ordinary-JAX time vectorization. Pallas adds only 1.09× over the already-vectorized highest-precision baseline here.
3. For ordinary first-order joint option training, saved-backward Pallas is promising (1.14× default; 1.65× highest). Keep these precision regimes separate.
4. Keep actual checkpointed pathwise meta-learning on ordinary JAX: the tested Pallas GRU fails there. Do not deploy saved custom-VJP variants where JVP is required.
5. Keep the tested credit implementation on JAX. Neither tested credit kernel variant provides a useful speed gain.

All changes are isolated in `pallas_study/`; production defaults and model code were not changed. These recommendations concern measured update speed, not convergence, financial performance, or validated deployment.

## Default-precision option results

RTX 4090; float32; hidden width 32; 126 steps. Speed ratio is paired JAX median / Pallas median; below 1 means slower. Ranges show min–max across repeated timing blocks, not confidence intervals.

| Workload | Batch | JAX ms/update | Pallas ms/update | Speed ratio | Pallas range ms |
|---|---:|---:|---:|---:|---:|
| Option hedger, analytic GRU | 128 | 8.387 | 6.805 | 1.23× | 6.748–7.396 |
| Option hedger, analytic GRU | 1024 | 13.849 | 9.004 | 1.54× | 8.966–9.061 |
| Share-bounded option hedger, analytic GRU | 128 | 8.958 | 7.281 | 1.23× | 7.114–7.864 |
| Share-bounded option hedger, analytic GRU | 1024 | 14.236 | 9.554 | 1.49× | 9.449–10.163 |
| Pathwise SDE, analytic MLP | 128 | 20.547 | 24.862 | 0.83× | 23.632–25.244 |
| Pathwise SDE, analytic MLP | 1024 | 25.186 | 26.599 | 0.95× | 26.367–26.773 |
| Joint SDE + hedger, analytic GRU only | 128 | 23.887 | 23.365 | 1.02× | 23.263–23.692 |
| Joint SDE + hedger, analytic GRU only | 1024 | 33.867 | 29.016 | 1.17× | 28.504–29.367 |
| Joint SDE + hedger, saved backward for both | 128 | 24.056 | 21.076 | 1.14× | 21.012–21.253 |

Sources: [final-default-b128.json](outputs/final-default-b128.json), [final-default-b1024.json](outputs/final-default-b1024.json), [saved-default-b128.json](outputs/saved-default-b128.json). “Hedger” uses the newer dollar-bounded convention; “historical” uses the share-bounded convention. At batch 128, the small 1.02× joint analytic-GRU ratio is insufficient evidence of a useful improvement. The default-precision SDE score Pallas variants failed the initial numerical gate and are excluded from speed claims.

The historical option policy was also tested in its float64 setting: **30.279 → 22.726 ms (1.33×)** for the analytic GRU at batch 128, with correctness and optimizer-trajectory gates passing. See [analytic-f64-b128.json](outputs/analytic-f64-b128.json). This is distinct from the float32 historical rows above.

## Highest-precision option results

Both members of each pair use `jax_default_matmul_precision='highest'`. These saved-activation custom-VJP kernels are first-order candidates; they do not supply the JVP required by some learning methods.

| Workload | Batch | JAX ms/update | Pallas ms/update | Speed ratio | Pallas range ms |
|---|---:|---:|---:|---:|---:|
| Option hedger | 128 | 12.258 | 8.655 | 1.42× | 8.560–9.074 |
| Scanned SDE score | 128 | 29.547 | 18.291 | 1.62× | 18.189–18.395 |
| Vectorized SDE score | 128 | 1.344 | 1.232 | 1.09× | 1.216–1.288 |
| Pathwise SDE | 128 | 40.218 | 25.828 | 1.56× | 25.649–25.886 |
| Joint SDE + hedger | 128 | 43.795 | 26.561 | 1.65× | 26.250–27.218 |

Source: [saved-high-b128.json](outputs/saved-high-b128.json). These speedups are relative to the slower highest-precision JAX baseline, not the default-precision baseline. Default XLA matmul precision and scalar Pallas FP32 arithmetic produced materially different score gradients in some tests. Tight highest-precision agreement supports the implementation but does not make rejected default comparisons valid.

For the ordinary-JAX vectorization comparison, scanned versus flattened score value/gradient maximum absolute difference was **4.77e-7 default**, **2.79e-9 highest**. The fixed-path per-transition score permits independent evaluation across time; sequential path simulation still has temporal dependence. This optimization therefore does not establish the same gain for path generation or joint training.

## Credit results

Batch 128, 1260 daily steps. The hedger includes the real 126-vintage ledger and float64 accounting; the continuous-credit score is float32 and already vectorizes time.

| Workload | Batch | JAX ms/update | Pallas ms/update | Speed ratio | Pallas range ms |
|---|---:|---:|---:|---:|---:|
| Credit hedger FP64, recomputed JVP | 128 | 500.664 | 681.792 | 0.73× | 680.990–682.000 |
| Credit hedger FP64, saved backward | 128 | 500.664 | 501.388 | 1.00× | 501.052–501.759 |
| Credit score FP32, recomputed JVP | 128 | 1.953 | 2.084 | 0.94× | 2.041–2.307 |
| Credit score FP32, saved backward | 128 | 1.953 | 3.308 | 0.59× | 3.086–3.464 |

Sources: [credit-hedger.json](outputs/credit-hedger.json), [credit-sde.json](outputs/credit-sde.json). Credit parameter gradients and optimizer trajectories passed. Saved backward essentially tied the ~501 ms credit hedger baseline; the other cases were slower. These results cover the tested recomputed-JVP and saved-backward implementations, not every possible future credit kernel.

## Differentiation and incomplete prototypes

- Isolated analytic GRU and MLP checks at widths 32 and 64 passed forward, parameter/input gradient, JVP, Hessian-vector, and finite-difference checks: [analytic-validation.json](outputs/analytic-validation.json).
- Actual pathwise meta-learning used two inner AdamW updates, batch 128, 126 days, and the production checkpointed learner block. JAX compiled, but the Pallas GRU failed with `ValueError: Too few leaves for PyTreeDef; expected 2, got 1`. Its selfplay reward numerical check passed (relative L2 error ~1.45e-6). The export stops before a completed MLP/meta comparison or timings. See [meta-v7.json](outputs/meta-v7.json), [repair.log](outputs/repair.log), and [meta.json](outputs/meta.json). Isolated HVP success does not establish compatibility with this full transformation stack.
- Whole-trajectory persistent fusion repeatedly hit Triton lowering errors (concatenation and scalar dtype/broadcast handling). The latest local correction has not passed GPU validation. There is **no valid persistent-kernel timing**; [persistent-f64-b128.json](outputs/persistent-f64-b128.json) is incomplete and excluded.
- The batched tensor-core MLP prototype has **no completed GPU result**. Do not treat its local source as validated or assign it a speedup.
- A short A100 confirmation rental was attempted, but direct SSH and the Vast proxy both timed out. It was deleted without running benchmarks. All numerical results in this report are RTX 4090 results.

## Measurement and numerical gates

Environment: Python 3.12.15, JAX/jaxlib 0.7.2, Equinox 0.13.2, Optax 0.2.6, Chex 0.1.91, Triton 3.4.0, CUDA 12 packages. Exact versions and device details are in [environment.txt](outputs/environment.txt) and [nvidia-smi.txt](outputs/nvidia-smi.txt).

Options use one fixed initialization/noise scheme shared across variants, 3 warmups, then 9 timing blocks of 10 clipped-AdamW updates, with alternating variant order and explicit synchronization. The timed update contains the objective, reverse gradient, clipping, and optimizer; joint/pathwise objectives also contain their required simulation. Credit uses 7 blocks of 3 updates. Compilation is recorded separately in raw JSON; the CSV includes compile-and-first-update time where available. Measurements do not cover an entire training campaign, input transfer, evaluation, checkpointing, replay management, or fresh random-number generation outside the objective. They use fixed inputs and do not establish convergence equivalence across long training runs.

Initial acceptance: absolute objective error <2e-5 and either gradient relative L2 <0.002 or maximum absolute error <2e-6. Option optimizer checks compare 20 updates (credit: 10): absolute reported-loss difference <2e-5 and either parameter relative L2 <0.001 or maximum absolute error <1e-4. The logged `loss_after_20_updates` is the objective returned by the twentieth update, evaluated before that update's parameter change. These tolerances permit small deviations; they are not bitwise equality. All Pallas rows in the tables pass both gates. Default-precision rejected SDE-score rows remain in raw data and the summary with `eligible=false`.

No multi-seed or multiple-device statistical study was performed. Small differences, especially around 1–9%, need fresh confirmation before adoption. The study explored multiple kernel revisions, so headline rows explicitly identify their files rather than averaging incompatible versions.

## Artifacts and reproduction

- [summary.csv](summary.csv) / [summary.json](summary.json): all completed timing rows, including rejected variants, gates, ranges, errors, precision, and compile times.
- [manifest.json](manifest.json): SHA-256 hashes of retained study artifacts.
- [audit.json](audit.json): export-integrity and source-hash audit.
- `versions/v1`–`versions/v7`: exported source snapshots. The final analytic default runs identify the v6 kernel hash; the float64 analytic run matches v4/v5. Early saved-backward results predate embedded per-result source hashes; `run_saved.sh` records the v2 snapshot workflow. Their provenance is less strong than later hash-tagged runs.
- Latest top-level prototype files include **untested later changes**. Reproduce published measurements from matching archived versions, not by assuming the latest files generated every table.

On a prepared matching CUDA environment, copy the chosen archived kernel/benchmark sources into an isolated checkout's `pallas_study` directory, retain the matching dependencies, and run for example:

```sh
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pallas_study.benchmark --batch 128 --steps 126 --analytic-jvp --precision default --repeats 9 --updates 10 --output outputs/reproduction.json
```

That command targets the v6 analytic run. The v2 first-order experiment instead uses `--saved-backward --precision highest` without `--analytic-jvp`. Inspect the archived CLI before running. Regenerate local tables with `python3 pallas_study/summarize.py`. The cloud orchestration scripts retain historical lifecycle state and should not be blindly rerun.

## Budget and cleanup

Starting account credit: **$12.266596**. After cleanup: **$8.300492** (provider snapshot 22 October 2026). Account decline: **$3.966104**. This is an account-level delta, not an itemized compute bill; storage or other account charges can contribute. The main RTX 4090 rental was capped at six hours at $0.53956/hour; the failed confirmation A100 was capped at one hour at $0.35789/hour and deleted early.

Both study instances (**57110659**, **57179650**) are absent from the final provider listing. The study watchdog was stopped; no study monitor remains running. Four pre-existing stopped instances were left untouched. Credentials remain outside the repository in the owner-only temporary access directory; no key or API secret is included in this report or evidence bundle. Remaining credit was preserved once additional testing was blocked and the central findings were supported.
