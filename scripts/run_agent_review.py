"""Run statistical-agent review scaffolds for diagnostics and model critique."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stai_x_forecasting.agents.registry import build_agent_registry
from stai_x_forecasting.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/agents.yaml", help="Agent config path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(Path(args.config))
    registry = build_agent_registry(config)
    for name, agent in registry.items():
        print(f"{name}: {agent.mission}")


if __name__ == "__main__":
    main()
