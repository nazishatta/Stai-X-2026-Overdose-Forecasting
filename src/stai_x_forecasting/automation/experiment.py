"""Experiment metadata primitives."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentRecord:
    """Metadata captured for a single experiment run."""

    experiment_id: str
    track: str
    hypothesis: str
    status: str = "planned"
