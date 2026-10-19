"""Feasible observable-only economic pre-screen rules."""
import equinox as eqx
import jax.numpy as jnp


class FixedProtection(eqx.Module):
    target: float = eqx.field(static=True)
    replace_daily: bool = eqx.field(static=True)
    n_vintages: int = eqx.field(static=True)
    max_protection: float = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True, default=1)

    def __init__(self, target, replace_daily=False, n_vintages=126, max_protection=2.):
        if not 0 <= target <= max_protection:
            raise ValueError('Target must obey the common notional bound')
        self.target = target
        self.replace_daily = replace_daily
        self.n_vintages = n_vintages
        self.max_protection = max_protection

    def __call__(self, features, hidden):
        total = features[3]*self.max_protection
        buy = jnp.asarray(self.target) if self.replace_daily else jnp.maximum(self.target-total, 0)
        retain = jnp.full(self.n_vintages, 0. if self.replace_daily else 1.)
        return hidden, dict(buy=buy, retain=retain)


class ObservableSensitivity(eqx.Module):
    """Ordinary-model quote-implied delta proxy, shared across every domain.

    Map today's new par spread to an ordinary-model state; approximate the
    financed bond hedge using a newly issued CDS sensitivity there. This is a
    deliberately misspecified estimator in the two-factor domain. It does not
    receive realized factors or claim to hedge every old-vintage sensitivity.
    """
    spreads: object
    targets: object
    n_vintages: int = eqx.field(static=True)
    max_protection: float = eqx.field(static=True)
    bond_days: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True, default=1)

    def __init__(self, ordinary_tables, config):
        self.n_vintages = config.cds_days
        self.max_protection = config.max_protection
        self.bond_days = config.bond_days
        self.spreads = ordinary_tables['par_spread']
        grid = ordinary_tables['x_grid']
        bond_delta = jnp.gradient(ordinary_tables['bond'], grid, axis=1)
        protection_delta = jnp.gradient(ordinary_tables['protection'][config.cds_days], grid)
        annuity_delta = jnp.gradient(ordinary_tables['annuity'][config.cds_days], grid)
        cds_delta = protection_delta-self.spreads*annuity_delta
        # Endpoint/low-sensitivity regularization is fixed before outcomes.
        self.targets = jnp.clip(-bond_delta/jnp.maximum(cds_delta, 1e-8), 0, self.max_protection)

    def __call__(self, features, hidden):
        remaining = jnp.clip(jnp.rint(features[2]*self.bond_days).astype(jnp.int32), 0, self.bond_days)
        target = jnp.interp(features[0]/100, self.spreads, self.targets[remaining])
        total = features[3]*self.max_protection
        retain_fraction = jnp.minimum(1., target/jnp.maximum(total, 1e-12))
        buy = jnp.maximum(target-total, 0)
        return hidden, dict(buy=buy, retain=jnp.full(self.n_vintages, retain_fraction))
