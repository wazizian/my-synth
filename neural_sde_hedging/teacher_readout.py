"""GPU-only matched-budget readout of frozen DPG generators; no teacher updates."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import SDEConfig
from .dynamics import NeuralSDE, sample_noise, simulate
from .hedger import Contract, Hedger, terminal_losses
from .hyperparameter_validation import paired_interval, save_json
from .metrics import risk_metrics
from .training import atomic_checkpoint, learner_block, load_checkpoint, restore_layout


from .runtime import require_enabled


def main():
    """Compare prior/default/selected frozen teachers using same 96/512-step learner budgets.

    Inputs CLI trusted DPG full checkpoints and output folder. Outputs curves,
    learner partitions, losses and paired path-bootstrap intervals on a new
    8192-path prior-target set, never used to choose generator hyperparameters.
    """
    require_enabled()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-checkpoint", required=True)
    parser.add_argument("--selected-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError("Teacher readout requires a provisioned GPU worker")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = SDEConfig()
    contract = Contract(payoff_cap=1.)
    prior = NeuralSDE(jax.random.key(20261018), config)
    teacher_template, teacher_static = eqx.partition(prior, eqx.is_inexact_array)
    teachers = {"prior": prior}
    for name, checkpoint in (("default", args.default_checkpoint), ("selected", args.selected_checkpoint)):
        state = load_checkpoint(Path(checkpoint))
        if state["round_index"] != 256:
            raise RuntimeError("Expected frozen generator after 256 teacher rounds")
        params = restore_layout(state["teacher"], teacher_template)
        teachers[name] = eqx.combine(params, teacher_static)
    initial, learner_static = eqx.partition(Hedger(jax.random.key(20261017)), eqx.is_inexact_array)

    def learning_rate(count):
        """Return common learner LR .001 with 32-update warmup from optimizer count."""
        result = .001 * jnp.minimum((count + 1.) / 32., 1.)
        chex.assert_shape(result, ())
        return result

    optimizer = optax.chain(optax.clip_by_global_norm(1.),
                           optax.adamw(learning_rate, eps_root=1e-16, weight_decay=.01))

    def train(student, states, key):
        """Return student arrays/moments after eight bootstrap updates on states [256,127,3]."""
        chex.assert_shape(states, (256, 127, 3))
        weights = jnp.ones((256,))
        chex.assert_shape(weights, (256,))
        return learner_block(student, states, weights, learner_static, optimizer,
                             contract, 10., 8, key)

    def evaluate(params, states):
        """Return raw held-out losses [8192] for learner params and states [8192,127,3]."""
        chex.assert_shape(states, (8192, 127, 3))
        log_spot = states[..., 0]
        chex.assert_shape(log_spot, (8192, 127))
        losses = terminal_losses(eqx.combine(params, learner_static),
                                 SimpleNamespace(log_spot=log_spot), contract)
        chex.assert_shape(losses, (8192,))
        return losses

    compiled_simulate = eqx.filter_jit(simulate)
    compiled_train = eqx.filter_jit(train)
    compiled_evaluate = eqx.filter_jit(evaluate)
    target = compiled_simulate(prior, sample_noise(jax.random.key(20261222), 8192, config), config).states
    chex.assert_shape(target, (8192, 127, 3))
    initial_losses = compiled_evaluate(initial, target)
    results, evaluated_losses = {}, {}
    save_json(output / "protocol.json", {"seed": 20261017, "training_seed": 20261223,
              "evaluation_seed": 20261222, "evaluation_episodes": 8192,
              "learner_lr": .001, "warmup_steps": 32, "batch": 256,
              "learner_updates": [96, 512], "teacher_updates": 0,
              "scope": "frozen-generator diagnostic; settings fixed before confirmation; no reselection"})
    for name, teacher in teachers.items():
        student = (initial, optimizer.init(initial))
        key = jax.random.key(20261223)
        curve = []
        for block in range(64):
            key, noise_key, train_key = jax.random.split(key, 3)
            paths = compiled_simulate(teacher, sample_noise(noise_key, 256, config), config)
            student = compiled_train(student, paths.states, train_key)
            if block + 1 in (1, 4, 8, 12, 16, 32, 48, 64):
                losses = compiled_evaluate(student[0], target)
                if not bool(jnp.all(jnp.isfinite(losses))):
                    raise RuntimeError("Nonfinite readout loss")
                metrics = {k: float(v) for k, v in risk_metrics(losses).items()}
                curve.append({"updates": (block + 1) * 8, **metrics})
                save_json(output / (name + "-curve.json"), curve)
            if block + 1 in (12, 64):
                tag = f"{name}-{(block + 1) * 8}"
                evaluated_losses[tag] = losses
                np.save(output / (tag + "-losses.npy"), np.asarray(losses))
                atomic_checkpoint(output / (tag + "-learner.pkl"), student[0])
                results[tag] = {"metrics": metrics,
                                "vs_initial": paired_interval(losses, initial_losses, jax.random.key(20261224))}
                save_json(output / "results.json", results)
        print(name, curve[-1], flush=True)
    for name in ("default", "selected"):
        for steps in (96, 512):
            tag = f"{name}-{steps}"
            results[tag]["vs_prior"] = paired_interval(evaluated_losses[tag],
                                                         evaluated_losses[f"prior-{steps}"],
                                                         jax.random.key(20261225))
    save_json(output / "results.json", results)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for name in teachers:
        curve = json.loads((output / (name + "-curve.json")).read_text())
        axis.plot([row["updates"] for row in curve], [row["entropic"] * 1e4 for row in curve],
                  marker="o", label=name)
    axis.set(xlabel="Learner updates", ylabel="Held-out entropic risk (bp)",
             title="DPG frozen-generator readout: equal learner budgets")
    axis.legend()
    axis.grid(alpha=.2)
    fig.savefig(output / "readout.png", dpi=160)
    fig.savefig(output / "readout.pdf")
    plt.close(fig)
    save_json(output / "summary.json", {"complete": True, "results": results})


if __name__ == "__main__":
    main()
