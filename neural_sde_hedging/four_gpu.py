"""Detached, bounded four-GPU pilot orchestration; uses only stdlib on control path.

Run on the rented GPU node, in the already installed GPU Python environment.
See FOUR_GPU.md. No provisioning, dependency installation, or work at import.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


METHODS = ("baseline", "dpg", "soar", "selfplay")


def write_json(path, value):
    """Atomically write JSON value to Path; return nothing."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def export(config):
    """Copy committed artifacts to configured rsync destination; return success."""
    try:
        result = subprocess.run(
            ["rsync", "-az", "--partial", "--timeout=30", "--exclude=*.tmp",
             "--exclude=__pycache__", "--exclude=.campaign.lock",
             config["output"] + "/", config["export_to"]],
            timeout=60, check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def cleanup(config):
    """Invoke idempotent provider deletion argv; mark successful return on disk."""
    root = Path(config["output"])
    if (root / "cleanup.done").exists():
        return True
    with (root / "cleanup.log").open("a") as stream:
        try:
            result = subprocess.run(config["cleanup_argv"], stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=60, check=False)
            if result.returncode == 0:
                (root / "cleanup.done").touch()
                return True
        except (OSError, subprocess.TimeoutExpired) as error:
            stream.write(str(error) + "\n")
    return False


def watchdog(config):
    """Enforce absolute cleanup deadline independently of supervisor/SSH session."""
    root = Path(config["output"])
    while time.time() < config["deadline"] + 180:
        if (root / "cleanup.done").exists():
            return
        time.sleep(5)
    for _ in range(5):
        if cleanup(config):
            return
        time.sleep(10)


def stop(process):
    """Terminate a job's entire new process group, then force-kill after ten seconds."""
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except ProcessLookupError:
            pass


def disk_bytes(root):
    """Return apparent bytes of campaign files without following directory symlinks."""
    return sum(path.stat().st_size for path in root.rglob("*")
               if path.is_file() and not path.is_symlink())


def run_jobs(config, jobs, deadline):
    """Run (name, GPU UUID, argv) jobs concurrently; export and enforce resource limits."""
    root = Path(config["output"])
    children = []
    last_export = 0.
    export_failures = 0
    try:
        for name, gpu, argv in jobs:
            env = os.environ | {"CUDA_VISIBLE_DEVICES": gpu,
                                "JAX_PLATFORMS": "cuda",
                                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                                "PYTHONUNBUFFERED": "1"}
            with (root / (name + ".log")).open("a") as log:
                process = subprocess.Popen(argv, cwd=root / "code", env=env,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            children.append((name, process))
        while True:
            failures = {name: process.returncode for name, process in children
                        if process.poll() not in (None, 0)}
            if failures:
                raise RuntimeError(f"GPU job failed: {failures}; inspect logs")
            if all(process.poll() == 0 for _, process in children):
                return
            if time.time() >= min(deadline, config["deadline"]):
                raise TimeoutError("Hard phase/campaign deadline reached")
            if shutil.disk_usage(root).free < config["min_free_gb"] * 1024**3:
                raise RuntimeError("Free disk reserve reached")
            if time.time() - last_export >= config["export_interval"]:
                telemetry = subprocess.run(
                    ["nvidia-smi", "--query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True,
                    timeout=10, check=True)
                with (root / "gpu-telemetry.csv").open("a") as stream:
                    stream.write(telemetry.stdout)
                if disk_bytes(root) > config["max_artifact_gb"] * 1024**3:
                    raise RuntimeError("Campaign artifact cap reached")
                ok = export(config)
                export_failures = 0 if ok else export_failures + 1
                last_export = time.time()
                if export_failures >= 3:
                    raise RuntimeError("Three consecutive artifact exports failed")
            time.sleep(2)
    finally:
        for _, process in children:
            stop(process)


def training_jobs(config, phase):
    """Build four equal-seed training commands; calibration and pilot states are separate."""
    root = Path(config["output"])
    seconds = config["calibration_seconds"] if phase == "calibration" else config["pilot_seconds"]
    jobs = []
    for method, gpu in zip(METHODS, config["gpus"], strict=True):
        profile = config.get("hyperparameters", {}).get(method, {})
        argv = [config["python"], "-m", "neural_sde_hedging.training",
                "--method", method, "--output", str(root / phase / method),
                "--seconds", str(seconds), "--rounds",
                str(config["calibration_rounds"] if phase == "calibration" else 1000000),
                "--seed", "20261017", "--groups", "4", "--items", str(profile.get("items", 32)),
                "--inner-steps", "8", "--n-steps", "126", "--gamma", "10",
                "--student-lr", str(profile.get("student_lr", .0003)),
                "--teacher-lr", str(profile.get("teacher_lr", .001)),
                "--warmup-steps", str(profile.get("warmup_steps", 32)), "--weight-decay", "0.01",
                "--feedback-size", "512", "--payoff-cap", "1",
                "--min-free-gb", str(config["min_free_gb"])]
        if config["feedback_paths"]:
            argv += ["--feedback-paths", config["feedback_paths"]]
        jobs.append((phase + "-" + method, gpu, argv))
    write_json(root / (phase + "-commands.json"), jobs)
    return jobs


def inspect_phase(config, phase):
    """Validate finite progress/checkpoints and summarize risk/reward/round throughput."""
    report = {}
    for method in METHODS:
        folder = Path(config["output"]) / phase / method
        status = json.loads((folder / "status.json").read_text())
        rows = [json.loads(line) for line in (folder / "progress.jsonl").read_text().splitlines()]
        if not rows or not (folder / "latest.pkl").is_file():
            raise RuntimeError(f"{phase}/{method}: missing committed training output")
        if any(not math.isfinite(value) for row in rows for value in row.values()):
            raise RuntimeError(f"{phase}/{method}: nonfinite metric")
        if phase == "calibration" and (
            status["status"] != "complete" or len(rows) < config["calibration_rounds"]
        ):
            raise RuntimeError(f"{method}: calibration did not finish requested rounds")
        differences = [b["seconds"] - a["seconds"] for a, b in zip(rows, rows[1:])]
        report[method] = {"status": status,
                          "first": rows[0], "last": rows[-1],
                          "mean_postcompile_round_seconds":
                              sum(differences) / len(differences) if differences else None}
    write_json(Path(config["output"]) / (phase + "-summary.json"), report)
    return report


def supervise(config):
    """Run GPU tests, calibration gate, then bounded pilot; export before cleanup."""
    root = Path(config["output"])
    with (root / ".campaign.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = {"status": "running", "started": time.time()}
        try:
            if not export(config):
                raise RuntimeError("Initial artifact export failed")
            # Tests opt in explicitly and run on a GPU, never on the control host.
            test_argv = [config["python"], "-m", "pytest", "-q",
                         "tests/test_neural_sde_gpu.py", "--junitxml=" + str(root / "pytest.xml")]
            os.environ["NEURAL_SDE_GPU_TESTS"] = "1"
            if not (root / "gpu-tests-passed.json").exists():
                run_jobs(config, [("gpu-tests", config["gpus"][0], test_argv)],
                         min(config["deadline"], time.time() + config["test_seconds"]))
                write_json(root / "gpu-tests-passed.json", {"passed": True})
            if not (root / "calibration-gate.json").exists():
                run_jobs(config, training_jobs(config, "calibration"),
                         time.time() + config["calibration_seconds"] + 120)
                inspect_phase(config, "calibration")
                write_json(root / "calibration-gate.json",
                           {"passed": True, "meaning": "finite execution and throughput only; not convergence"})
            remaining = []
            for method in METHODS:
                progress = root / "pilot" / method / "progress.jsonl"
                # A checkpoint may be one round ahead of progress after interruption;
                # using the log is conservative for the hard-deadline estimate.
                lines = progress.read_text().splitlines() if progress.exists() else []
                elapsed = json.loads(lines[-1])["seconds"] if lines else 0.
                remaining.append(max(0., config["pilot_seconds"] - elapsed))
            if time.time() + max(remaining) + 120 > config["deadline"]:
                raise RuntimeError("Insufficient remaining hard budget for pilot")
            run_jobs(config, training_jobs(config, "pilot"),
                     time.time() + max(remaining) + 120)
            inspect_phase(config, "pilot")
            result["status"] = "pilot_finished"
        except Exception as error:
            result |= {"status": "failed", "error": repr(error)}
            raise
        finally:
            result["finished"] = time.time()
            write_json(root / "campaign-status.json", result)
            exported = export(config)
            write_json(root / "final-export.json", {"success": exported})
            export(config)
            # Independent watchdog retries if provider deletion fails here.
            cleanup(config)


def launch(args):
    """Validate allocation/budget, snapshot code, and detach supervisor plus watchdog."""
    root = Path(args.output).resolve()
    if root.exists():
        raise ValueError("Use a new output directory for each campaign")
    cleanup_argv = json.loads(args.cleanup_argv)
    if not isinstance(cleanup_argv, list) or not cleanup_argv or not all(
        isinstance(value, str) and value for value in cleanup_argv
    ):
        raise ValueError("--cleanup-argv must be a nonempty JSON list of command arguments")
    positive = (args.pilot_hours, args.calibration_seconds, args.test_seconds,
                args.hourly_rate, args.max_cost, args.min_free_gb,
                args.max_artifact_gb, args.export_interval)
    if not all(math.isfinite(value) and value > 0 for value in positive):
        raise ValueError("Budgets, rate and resource limits must be positive finite numbers")
    if args.calibration_rounds < 8:
        raise ValueError("At least eight calibration rounds are required")
    profiles = {}
    if args.hyperparameter_profile:
        profiles = json.loads(Path(args.hyperparameter_profile).read_text())
        if not isinstance(profiles, dict) or set(profiles) != set(METHODS):
            raise ValueError("Hyperparameter profile must contain all four methods")
        for method, profile in profiles.items():
            if not isinstance(profile, dict) or set(profile) != {
                "method", "student_lr", "teacher_lr", "items", "warmup_steps"
            } or profile["method"] != method:
                raise ValueError("Invalid method hyperparameter profile")
            for name in ("student_lr", "teacher_lr"):
                value = profile[name]
                if (not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not math.isfinite(value) or value <= 0):
                    raise ValueError("Learning rates must be finite and positive")
            if (not isinstance(profile["items"], int) or isinstance(profile["items"], bool)
                    or profile["items"] < 2):
                raise ValueError("Items must be an integer >=2")
            if (not isinstance(profile["warmup_steps"], int) or isinstance(profile["warmup_steps"], bool)
                    or profile["warmup_steps"] < 0):
                raise ValueError("Warmup must be a nonnegative integer")
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"], text=True)
    rows = [row.split(",", 1) for row in inventory.strip().splitlines()]
    if len(rows) != 4 or any("H100" not in row[1] for row in rows):
        raise RuntimeError("Expected exactly four H100 GPUs")
    if not shutil.which("rsync"):
        raise RuntimeError("Install rsync before launching")
    duration = args.test_seconds + args.calibration_seconds + args.pilot_hours * 3600 + 600
    # Include watchdog cleanup grace and five deletion retries in the spending envelope.
    upper_cost = (duration + 180 + 350) / 3600 * args.hourly_rate
    if upper_cost > args.max_cost:
        raise ValueError(f"Conservative compute cost ${upper_cost:.2f} exceeds cap ${args.max_cost:.2f}")
    if shutil.disk_usage(root.parent).free < args.min_free_gb * 1024**3:
        raise RuntimeError("Insufficient free disk")
    root.mkdir()
    code = root / "code"
    code.mkdir()
    source = Path(__file__).resolve().parent
    shutil.copytree(source, code / source.name, ignore=shutil.ignore_patterns("__pycache__"))
    (code / "tests").mkdir()
    shutil.copy2(source.parent / "tests/test_neural_sde_gpu.py", code / "tests")
    feedback = None
    if args.feedback_paths:
        feedback = root / "feedback.npz"
        shutil.copy2(Path(args.feedback_paths).resolve(), feedback)
    config = vars(args) | {"output": str(root), "python": sys.executable,
                           "cleanup_argv": cleanup_argv, "gpus": [row[0].strip() for row in rows],
                           "pilot_seconds": args.pilot_hours * 3600,
                           "hyperparameters": profiles,
                           "feedback_paths": str(feedback) if feedback else None,
                           "deadline": time.time() + duration,
                           "upper_compute_cost_usd": upper_cost}
    config_path = root / "campaign.json"
    write_json(config_path, config)
    write_json(root / "source-sha256.json", {
        str(path.relative_to(code)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in code.rglob("*.py")})
    (root / "gpu-inventory.txt").write_text(inventory)
    with (root / "packages.txt").open("w") as stream:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=stream, check=True)
    # Both detached processes use immutable snapshot code, not the working tree.
    for mode in ("watchdog", "supervise"):
        with (root / (mode + ".log")).open("a") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "neural_sde_hedging.four_gpu", mode,
                 "--config", str(config_path)], cwd=code, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (root / (mode + ".pid")).write_text(str(process.pid) + "\n")
    print(f"Detached campaign: {root}\nConservative compute envelope: ${upper_cost:.2f}")


def main():
    """Parse launch or internal daemon command; dispatch without importing JAX."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    command = modes.add_parser("launch")
    command.add_argument("--output", required=True)
    command.add_argument("--export-to", required=True, help="rsync destination outside this rented node")
    command.add_argument("--cleanup-argv", required=True, help="JSON argv for idempotent provider deletion")
    command.add_argument("--hourly-rate", type=float, required=True, help="Total node USD/hour")
    command.add_argument("--max-cost", type=float, required=True, help="Compute envelope; excludes storage/network")
    command.add_argument("--pilot-hours", type=float, default=2.)
    command.add_argument("--calibration-seconds", type=float, default=900.)
    command.add_argument("--calibration-rounds", type=int, default=16)
    command.add_argument("--test-seconds", type=float, default=600.)
    command.add_argument("--min-free-gb", type=float, default=10.)
    command.add_argument("--max-artifact-gb", type=float, default=10.)
    command.add_argument("--export-interval", type=float, default=60.)
    command.add_argument("--feedback-paths")
    command.add_argument("--hyperparameter-profile", help="Per-method JSON from hyperparameter validation")
    for mode in ("supervise", "watchdog"):
        command = modes.add_parser(mode)
        command.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.mode == "launch":
        launch(args)
    else:
        config = json.loads(Path(args.config).read_text())
        (supervise if args.mode == "supervise" else watchdog)(config)


if __name__ == "__main__":
    main()
