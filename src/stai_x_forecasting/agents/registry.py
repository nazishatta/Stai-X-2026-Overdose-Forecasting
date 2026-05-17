"""Build configured statistical agents."""

from __future__ import annotations

from typing import Any

from stai_x_forecasting.agents.base import StatisticalAgent


def build_agent_registry(config: dict[str, Any]) -> dict[str, StatisticalAgent]:
    """Create statistical-agent definitions from config."""
    agents = {}
    for name, values in config.get("agents", {}).items():
        agents[name] = StatisticalAgent(
            name=name,
            mission=values["mission"],
            outputs=tuple(values.get("outputs", ())),
        )
    return agents
