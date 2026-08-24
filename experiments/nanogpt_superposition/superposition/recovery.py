"""FLOP/source-byte based coarse, mixed, and ordinary recovery schedules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecoverySchedule:
    coarse_fraction: float
    mixed_fraction: float
    ordinary_fraction: float
    mixed_superposition_probability: float = 0.5

    @classmethod
    def parse(cls, value: str) -> RecoverySchedule:
        aliases = {
            "50/25/25": (0.50, 0.25, 0.25),
            "50/0/50": (0.50, 0.0, 0.50),
            "75/15/10": (0.75, 0.15, 0.10),
            "30/30/40": (0.30, 0.30, 0.40),
        }
        if value not in aliases:
            raise ValueError(f"unsupported recovery schedule: {value}")
        return cls(*aliases[value])

    def __post_init__(self) -> None:
        total = self.coarse_fraction + self.mixed_fraction + self.ordinary_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError("recovery fractions must sum to one")
        if not 0.0 <= self.mixed_superposition_probability <= 1.0:
            raise ValueError("mixed probability must be in [0, 1]")

    def phase(self, consumed_budget: float, total_budget: float) -> str:
        if total_budget <= 0:
            raise ValueError("total budget must be positive")
        fraction = min(max(consumed_budget / total_budget, 0.0), 1.0)
        if fraction < self.coarse_fraction:
            return "coarse"
        if fraction < self.coarse_fraction + self.mixed_fraction:
            return "mixed"
        return "ordinary"

    def use_superposition(
        self,
        consumed_budget: float,
        total_budget: float,
        random_value: float,
    ) -> bool:
        phase = self.phase(consumed_budget, total_budget)
        if phase == "coarse":
            return True
        if phase == "ordinary":
            return False
        return random_value < self.mixed_superposition_probability

    def learning_rate_multiplier(
        self,
        consumed_budget: float,
        total_budget: float,
        coarse_multiplier: float,
        ordinary_multiplier: float = 1.0,
    ) -> float:
        """Return the phase multiplier, interpolating through mixed recovery."""
        if total_budget <= 0:
            raise ValueError("total budget must be positive")
        if coarse_multiplier <= 0:
            raise ValueError("coarse multiplier must be positive")
        if ordinary_multiplier <= 0:
            raise ValueError("ordinary multiplier must be positive")
        fraction = min(max(consumed_budget / total_budget, 0.0), 1.0)
        mixed_start = self.coarse_fraction
        mixed_end = mixed_start + self.mixed_fraction
        if fraction < mixed_start:
            return coarse_multiplier
        if fraction >= mixed_end or self.mixed_fraction == 0:
            return ordinary_multiplier
        mixed_progress = (fraction - mixed_start) / self.mixed_fraction
        return (
            coarse_multiplier
            + (ordinary_multiplier - coarse_multiplier) * mixed_progress
        )
