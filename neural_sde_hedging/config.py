"""Static, explicit integration grid and admissible neural SDE coefficient bounds."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SDEConfig:
    """Configure conditional path simulation; all fields are static inputs, not learned."""

    n_steps: int = 126
    dt: float = 1 / 252
    n_latent: int = 2
    spot0: float = 1.0
    diagonal_min: float = 0.05
    diagonal_max: float = 0.60
    off_diagonal_max: float = 0.25
    latent_drift_max: float = 2.0
    risk_price_max: float = 2.0
    feature_scale: float = 2.0

    def __post_init__(self):
        """Validate static dimensions and coefficient bounds; return nothing or raise ValueError."""
        if self.n_steps < 1 or self.n_latent < 1:
            raise ValueError("Require a positive time grid and at least one latent factor")
        positive = (self.dt, self.spot0, self.diagonal_min, self.diagonal_max,
                    self.latent_drift_max, self.risk_price_max, self.feature_scale)
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("SDE scales must be finite and strictly positive")
        if self.diagonal_max < self.diagonal_min:
            raise ValueError("diagonal_max must be at least diagonal_min")
        if not math.isfinite(self.off_diagonal_max) or self.off_diagonal_max < 0:
            raise ValueError("off_diagonal_max must be finite and nonnegative")

    @property
    def state_dim(self):
        """Return joint log spot and latent state dimension from static configuration."""
        return self.n_latent + 1
