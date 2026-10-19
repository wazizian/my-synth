"""One-seed GPU-only hyperparameter screening and independent confirmation.

Run only on a provisioned worker. The external supervisor supplies the hard
runtime limit, incremental artifact export and provider deletion watchdog.
"""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .config import SDEConfig
from .dynamics import NeuralSDE, sample_noise, simulate
from .hedger import Contract, Hedger, terminal_losses
from .metrics import risk_metrics
from .training import atomic_checkpoint, load_checkpoint, restore_layout


def evaluate(params, static, states, contract):
    """Return losses [episodes] from learner arrays/static and fixed paths [episodes,time,state]."""
    from types import SimpleNamespace

    episodes, n_time, n_state = states.shape
    chex.assert_shape(states, (episodes, n_time, n_state))
    log_spot = states[..., 0]
    chex.assert_shape(log_spot, (episodes, n_time))
    losses = terminal_losses(eqx.combine(params, static), SimpleNamespace(log_spot=log_spot), contract)
    chex.assert_shape(losses, (episodes,))
    return losses


def paired_interval(losses, reference, key, gamma=10.):
    """Return paired bootstrap interval for entropic improvement reference minus candidate.

    Inputs losses/reference [episodes], independent bootstrap key and fixed gamma.
    Output point estimate and 95% percentile bounds; this is conditional on one
    trained initialization and does not measure training-seed uncertainty.
    """
    episodes = losses.shape[0]
    chex.assert_shape((losses, reference), (episodes,))
    indices = jax.random.randint(key, (256, episodes), 0, episodes)
    chex.assert_shape(indices, (256, episodes))
    candidate = losses[indices]
    baseline = reference[indices]
    chex.assert_shape((candidate, baseline), (256, episodes))
    def risk(values):
        """Return entropic risk over last episode axis, preserving bootstrap leading axes."""
        result = (jax.scipy.special.logsumexp(gamma * values, axis=-1) - jnp.log(episodes)) / gamma
        chex.assert_shape(result, values.shape[:-1])
        return result
    improvements = risk(baseline) - risk(candidate)
    chex.assert_shape(improvements, (256,))
    bounds = jnp.quantile(improvements, jnp.array([.025, .975]))
    chex.assert_shape(bounds, (2,))
    point = risk(reference) - risk(losses)
    chex.assert_shape(point, ())
    return {"improvement": float(point), "lower95": float(bounds[0]), "upper95": float(bounds[1])}


def save_json(path, value):
    """Atomically write JSON-safe value to Path; return nothing."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def recovery(folder, output, trial):
    """Archive a committed active checkpoint/history for off-worker recovery; return nothing."""
    if not (folder / "latest.pkl").exists():
        return
    target = output / "active-recovery.tar.gz"
    temporary = target.with_suffix(".tmp")
    # latest.pkl is atomically replaced. A newer file may coexist with an older
    # progress log; all earlier halfway histories are already committed first.
    with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
        for path in (folder / "latest.pkl", folder / "configuration.json"):
            archive.add(path, arcname=path.name)
        history = folder / "learner-history"
        if history.exists():
            for path in sorted(history.glob("*.pkl")):
                archive.add(path, arcname="learner-history/" + path.name)
    if temporary.stat().st_size > 300 * 1024**2:
        raise RuntimeError("Recovery export exceeds 300 MiB cap")
    temporary.replace(target)
    save_json(output / "active-recovery.json", trial)


def run_trial(spec, rounds, tag, output, evaluator, template, dev, contract):
    """Train one fixed-seed configuration, export recovery/metrics, return development evaluation."""
    folder = Path("/workspace/hyperparameter-trials") / tag
    folder.mkdir(parents=True, exist_ok=True)
    destination = output / "trials" / tag
    destination.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "neural_sde_hedging.training",
               "--method", spec["method"], "--output", str(folder),
               "--rounds", str(rounds), "--seconds", "5400", "--n-steps", "126",
               "--groups", "4", "--items", str(spec["items"]), "--inner-steps", "8",
               "--feedback-size", "512", "--payoff-cap", "1", "--seed", "20261017",
               "--student-lr", str(spec["student_lr"]), "--teacher-lr", str(spec["teacher_lr"]),
               "--warmup-steps", str(spec["warmup_steps"]), "--min-free-gb", "3"]
    save_json(destination / "command.json", command)
    save_json(output / "current-trial.json", {"tag": tag, "specification": spec, "rounds": rounds})
    started = time.monotonic()
    print("START", tag, spec, flush=True)
    with (destination / "train.log").open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            last_sync = 0.
            while process.poll() is None:
                if time.monotonic() - started > 5520:
                    raise TimeoutError("Trial hard deadline")
                if shutil.disk_usage(output).free < 3 * 1024**3:
                    raise RuntimeError("Worker disk reserve reached")
                if time.monotonic() - last_sync > 45:
                    recovery(folder, output, {"tag": tag, "specification": spec, "command": command})
                    if (folder / "progress.jsonl").exists():
                        shutil.copy2(folder / "progress.jsonl", destination)
                    last_sync = time.monotonic()
                time.sleep(2)
            if process.returncode:
                raise RuntimeError(f"{tag}: exit {process.returncode}; see train.log")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    for name in ("configuration.json", "progress.jsonl", "status.json"):
        shutil.copy2(folder / name, destination)
    status = json.loads((folder / "status.json").read_text())
    if status["rounds"] != rounds or status["status"] != "complete":
        raise RuntimeError(f"Incomplete calibration {tag}: {status}")
    state = load_checkpoint(folder / "latest.pkl")
    params = restore_layout(state["student"][0], template)
    losses = evaluator(params, dev, contract)
    chex.assert_shape(losses, (dev.shape[0],))
    metrics = {key: float(value) for key, value in risk_metrics(losses).items()}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("Nonfinite development evaluation")
    atomic_checkpoint(destination / "learner.pkl", params)
    np.save(destination / "development-losses.npy", np.asarray(losses))
    rows = [json.loads(line) for line in (folder / "progress.jsonl").read_text().splitlines()]
    result = {"tag": tag, "specification": spec, "rounds": rounds, "development": metrics,
              "wall_seconds": time.monotonic() - started, "first": rows[0], "last": rows[-1],
              "promotions": sum(row.get("promoted", 0) for row in rows),
              "max_teacher_gradient_norm": max(row["teacher_gradient_norm"] for row in rows),
              "max_reward_abs": max(row["reward_max_abs"] for row in rows),
              "min_feedback_entropic_ess": min(row["feedback_entropic_ess"] for row in rows)}
    save_json(destination / "result.json", result)
    print("END", tag, result["development"], "seconds", result["wall_seconds"], flush=True)
    # Recovery is a rotating full state; every trial's learner/config/curves stay
    # exported permanently. Active full checkpoints remain worker-local too.
    recovery(folder, output, {"tag": tag, "specification": spec, "command": command})
    del state
    return result


def plot_results(output, records):
    """Render GPU-worker learning/reward curves and export CSV/PNG/PDF; return nothing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for method, axis in zip(("baseline", "dpg", "soar", "selfplay"), axes.flat, strict=True):
        for record in records:
            if record["specification"]["method"] != method:
                continue
            path = output / "trials" / record["tag"] / "progress.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            axis.plot([row["round"] for row in rows], [row["target_risk"] * 1e4 for row in rows],
                      label=record["tag"], linewidth=1.)
        axis.set(title=method, xlabel="Teacher rounds", ylabel="Feedback entropic risk (bp)")
        axis.legend(fontsize=6)
        axis.grid(alpha=.2)
    fig.savefig(output / "learning-curves.png", dpi=150)
    fig.savefig(output / "learning-curves.pdf")
    plt.close(fig)
    with (output / "screening.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["tag", "method", "student_lr", "teacher_lr", "items", "warmup_steps",
                         "rounds", "dev_entropic", "dev_cvar95", "dev_terminal_std", "seconds"])
        for row in records:
            spec = row["specification"]
            writer.writerow([row["tag"], *[spec[key] for key in
                             ("method", "student_lr", "teacher_lr", "items", "warmup_steps")],
                             row["rounds"], *[row["development"][key] for key in
                             ("entropic", "cvar95", "terminal_std")], row["wall_seconds"]])


def main():
    """Screen small LR/batch/warmup grid then confirm selected/default configurations independently."""
    if jax.default_backend() != "gpu":
        raise RuntimeError("Hyperparameter validation requires provisioned GPU")
    output = Path(os.environ["JOB_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    config = SDEConfig()
    prior = NeuralSDE(jax.random.key(20261018), config)
    initial, static = eqx.partition(Hedger(jax.random.key(20261017)), eqx.is_inexact_array)
    contract = Contract(payoff_cap=1.)
    simulator = eqx.filter_jit(simulate)
    dev = simulator(prior, sample_noise(jax.random.key(20261121), 2048, config), config).states
    confirmation = simulator(prior, sample_noise(jax.random.key(20261122), 8192, config), config).states
    chex.assert_shape(dev, (2048, 127, 3))
    chex.assert_shape(confirmation, (8192, 127, 3))
    save_json(output / "evaluation-protocol.json", {
        "seed": 20261017, "feedback_seed": 20261020, "development_seed": 20261121,
        "confirmation_seed": 20261122, "feedback_episodes": 512, "development_episodes": 2048,
        "confirmation_episodes": 8192, "screen_rounds": 64, "confirmation_rounds": 256,
        "law": "fixed initial neural-SDE prior P; demonstration target",
        "development_sha256": hashlib.sha256(np.asarray(dev).tobytes()).hexdigest(),
        "confirmation_sha256": hashlib.sha256(np.asarray(confirmation).tobytes()).hexdigest(),
        "selection": "lowest independent development entropic risk after 64 rounds",
        "confirmation": "selected and original default rerun from identical seed for 256 rounds; no reselection",
        "uncertainty": "paired path bootstrap only, not training-seed uncertainty",
        "daily_marked_error_std": "unavailable; not fabricated"})
    evaluator = eqx.filter_jit(lambda params, states, contract: evaluate(params, static, states, contract))
    initial_losses = evaluator(initial, confirmation, contract)
    np.save(output / "initial-confirmation-losses.npy", np.asarray(initial_losses))
    save_json(output / "initial-metrics.json", {k: float(v) for k, v in risk_metrics(initial_losses).items()})
    records = []
    defaults = {"student_lr": 3e-4, "teacher_lr": 1e-3, "items": 32, "warmup_steps": 32}
    selected = {}
    for rate in (3e-4, 1e-3):
        for items in (32, 64):
            spec = defaults | {"method": "baseline", "student_lr": rate, "items": items}
            tag = f"screen-baseline-lr{rate:g}-n{items}"
            records.append(run_trial(spec, 64, tag, output, evaluator, initial, dev, contract))
            save_json(output / "results.json", records)
    selected["baseline"] = min(records, key=lambda row: row["development"]["entropic"])["specification"]
    for method in ("dpg", "soar", "selfplay"):
        candidates = []
        for teacher_rate in (1e-4, 3e-4, 1e-3):
            for warmup in ((8, 32) if method == "dpg" else (32,)):
                spec = selected["baseline"] | {"method": method, "teacher_lr": teacher_rate,
                                               "warmup_steps": warmup}
                tag = f"screen-{method}-tlr{teacher_rate:g}-w{warmup}"
                row = run_trial(spec, 64, tag, output, evaluator, initial, dev, contract)
                records.append(row)
                candidates.append(row)
                save_json(output / "results.json", records)
        selected[method] = min(candidates, key=lambda row: row["development"]["entropic"])["specification"]
    save_json(output / "selected-configurations.json", selected)
    # Freeze all choices before observing any confirmation metrics.
    confirmations = {}
    for method in ("baseline", "dpg", "soar", "selfplay"):
        entries = [("selected", selected[method])]
        original = defaults | {"method": method}
        if original != selected[method]:
            entries.insert(0, ("default", original))
        for label, spec in entries:
            tag = f"confirm-{method}-{label}"
            row = run_trial(spec, 256, tag, output, evaluator, initial, dev, contract)
            records.append(row)
            params = restore_layout(load_checkpoint(output / "trials" / tag / "learner.pkl"), initial)
            losses = evaluator(params, confirmation, contract)
            np.save(output / "trials" / tag / "confirmation-losses.npy", np.asarray(losses))
            confirmations[tag] = {"specification": spec,
                "metrics": {k: float(v) for k, v in risk_metrics(losses).items()},
                "vs_initial": paired_interval(losses, initial_losses, jax.random.key(20261200))}
            save_json(output / "confirmation.json", confirmations)
            save_json(output / "results.json", records)
            # The final rotating archive is the selected self-play state. Avoid
            # duplicating that largest bank on the small control-plane disk.
            if label == "selected" and method != "selfplay":
                shutil.copy2(output / "active-recovery.tar.gz", output / (method + "-selected-recovery.tar.gz"))
    for method in ("baseline", "dpg", "soar", "selfplay"):
        chosen = f"confirm-{method}-selected"
        original = f"confirm-{method}-default"
        if original in confirmations:
            losses = jnp.asarray(np.load(output / "trials" / chosen / "confirmation-losses.npy"))
            reference = jnp.asarray(np.load(output / "trials" / original / "confirmation-losses.npy"))
            confirmations[chosen]["vs_default"] = paired_interval(losses, reference, jax.random.key(20261201))
    save_json(output / "confirmation.json", confirmations)
    plot_results(output, records)
    save_json(output / "summary.json", {"complete": True, "trial_count": len(records),
              "selected": selected, "confirmation": confirmations,
              "scope": "one-seed prior-target hyperparameter pilot; not final easy/base/hard benchmark"})
    print("HYPERPARAMETER VALIDATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
