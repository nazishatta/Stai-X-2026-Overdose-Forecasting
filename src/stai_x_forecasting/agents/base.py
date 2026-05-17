"""Base contracts for statistical agents."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StatisticalAgent:
    """A narrow, auditable statistical review agent."""

    name: str
    mission: str
    outputs: tuple[str, ...]
