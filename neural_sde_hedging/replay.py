"""Self-play fresh/mutation/replay pools, adapted from programs to SDEs.

A bank entry is a frozen SDE program, not just a price path. Replay executes
that program with fresh Brownian noise. Its behavior density is recomputed on
the newly executed joint path; an old trajectory's density is never reused.
This file deliberately performs host-side archive management outside JIT.
"""

from dataclasses import dataclass, field
import math
import random

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from .dynamics import path_log_prob, sample_noise, simulate


@dataclass
class BankEntry:
    """Frozen generating SDE, positive reward, niche, and creation round."""

    teacher: object
    reward: float
    niche: tuple[int, int]
    round_index: int


@dataclass
class ProgramBank:
    """Quality-diversity mutation archive plus uniform non-replay history."""

    archive: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    decay: float = 0.97
    per_niche: int = 8

    def __post_init__(self):
        """Validate decay and per-niche archive capacity; return nothing or raise ValueError."""
        if not math.isfinite(self.decay) or not 0 < self.decay <= 1:
            raise ValueError("Archive decay must lie in (0,1]")
        if not isinstance(self.per_niche, int) or self.per_niche < 1:
            raise ValueError("per_niche must be a positive integer")

    def age(self):
        """Decay mutation-archive scores once per round; return nothing."""
        for entries in self.archive.values():
            for entry in entries:
                entry.reward *= self.decay

    def add(self, teacher, reward, niche, round_index, source):
        """Save a non-replay program; positive rewards also enter elites."""
        if source not in ("fresh", "mutation", "replay"):
            raise ValueError("Unknown program source")
        if source == "replay":
            return
        if not math.isfinite(reward):
            raise ValueError("Nonfinite bank reward")
        entry = BankEntry(teacher, reward, niche, round_index)
        self.history.append(entry)
        if reward > 0:
            # Separate score objects: aging the mutation archive must not age
            # or otherwise alter the uniform history bank.
            elite = BankEntry(teacher, reward, niche, round_index)
            entries = self.archive.setdefault(niche, [])
            entries.append(elite)
            entries.sort(key=lambda x: x.reward, reverse=True)
            del entries[self.per_niche:]

    def replay(self, count, round_index, rng):
        """Draw earlier non-replay entries uniformly without replacement."""
        eligible = [e for e in self.history if e.round_index < round_index]
        return rng.sample(eligible, min(count, len(eligible)))

    def parents(self, count, rng):
        """Choose niches uniformly, then a parent uniformly within each."""
        occupied = list(self.archive.values())
        return [rng.choice(rng.choice(occupied)) for _ in range(count)] if occupied else []


def niche_of_path(states, dt=1 / 252):
    """Map joint path [time+1,state] and dt to 9 persistence × 4 annual-RV niches."""
    n_time, n_state = states.shape
    chex.assert_shape(states, (n_time, n_state))
    if n_time < 2 or n_state < 1 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Niches need a nonempty path and finite positive dt")
    increments = jnp.diff(states[:, 0])
    chex.assert_shape(increments, (n_time - 1,))
    squared = increments**2
    chex.assert_shape(squared, (n_time - 1,))
    centered = squared - jnp.mean(squared)
    chex.assert_shape(centered, (n_time - 1,))
    persistence = jnp.sum(centered[1:] * centered[:-1]) / (jnp.sum(centered**2) + 1e-12)
    chex.assert_shape(persistence, ())
    annual_rv = jnp.sqrt(jnp.mean(squared) / dt)
    chex.assert_shape(annual_rv, ())
    # Fixed descriptors, never fitted to target/final evaluation outcomes.
    pbin = int(jax.device_get(jnp.clip(jnp.floor((persistence + 1) * 4.5), 0, 8)))
    vbin = sum(float(jax.device_get(annual_rv)) > edge for edge in (0.15, 0.3, 0.6))
    return pbin, vbin


def mutate(teacher, key, scale=0.01):
    """Perturb SDE-program parameters locally; return an admissible SDE."""
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("Mutation scale must be finite and nonnegative")
    params, static = eqx.partition(teacher, eqx.is_inexact_array)
    leaves, structure = jax.tree.flatten(params)
    keys = jax.random.split(key, len(leaves))
    changed = []
    for value, subkey in zip(leaves, keys):
        noise = jax.random.normal(subkey, value.shape, value.dtype)
        chex.assert_shape(noise, value.shape)
        new_value = value + scale * noise
        chex.assert_shape(new_value, value.shape)
        changed.append(new_value)
    return eqx.combine(jax.tree.unflatten(structure, changed), static)


@eqx.filter_jit
def execute_program(program, noise, config, prior):
    """Execute program/noise/config and return path plus behavior/prior densities on that new path."""
    path = simulate(program, noise, config)
    behavior = path_log_prob(program, path, config)
    reference = path_log_prob(prior, path, config)
    batch = noise.shape[0]
    chex.assert_shape((behavior, reference), (batch,))
    return path, behavior, reference


def build_pool(teacher, prior, bank, key, config, batch_size, round_index,
               mutation_fraction=0.25, replay_fraction=0.25, mutation_scale=0.01,
               host_seed=20261017, execute_fn=execute_program, noise_fn=sample_noise):
    """Return joint paths, behavior/prior densities, sources, and SDE programs.

    Fixed total pool size. Missing archive entries fall back to fresh samples.
    Each replay is a newly executed frozen program. Every row's behavior
    density therefore corresponds to its actual producing SDE. Mutation rows
    have densities for auditing but must be excluded from policy gradients.
    """
    if (not isinstance(batch_size, int) or batch_size < 1
            or not all(math.isfinite(x) for x in (mutation_fraction, replay_fraction))
            or min(mutation_fraction, replay_fraction) < 0
            or mutation_fraction + replay_fraction > 1):
        raise ValueError("Invalid pool size/fractions")
    rng = random.Random(host_seed + round_index)
    replay = bank.replay(int(batch_size * replay_fraction), round_index, rng)
    parents = bank.parents(int(batch_size * mutation_fraction), rng)
    keys = jax.random.split(key, batch_size * 2)
    programs = [e.teacher for e in replay]
    sources = ["replay"] * len(programs)
    for index, parent in enumerate(parents):
        programs.append(mutate(parent.teacher, keys[index], mutation_scale))
        sources.append("mutation")
    while len(programs) < batch_size:
        programs.append(teacher)
        sources.append("fresh")
    paths, behavior, reference = [], [], []
    for index, program in enumerate(programs):
        noise = noise_fn(keys[batch_size + index], 1, config)
        path, behavior_density, reference_density = execute_fn(program, noise, config, prior)
        paths.append(path)
        behavior.append(behavior_density)
        reference.append(reference_density)
    merged = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *[
        (p.states, p.innovations, p.log_density) for p in paths
    ])
    # Keep the deterministic common time axis once, rather than concatenate it.
    pool = type(paths[0])(merged[0], merged[1], merged[2], paths[0].times)
    chex.assert_shape(pool.states, (batch_size, config.n_steps + 1, config.state_dim))
    chex.assert_shape(pool.innovations, (batch_size, config.n_steps, config.state_dim))
    chex.assert_shape(pool.log_density, (batch_size,))
    chex.assert_shape(pool.times, (config.n_steps + 1,))
    behavior = jnp.concatenate(behavior)
    chex.assert_shape(behavior, (batch_size,))
    reference = jnp.concatenate(reference)
    chex.assert_shape(reference, (batch_size,))
    return pool, jax.lax.stop_gradient(behavior), jax.lax.stop_gradient(reference), sources, programs
