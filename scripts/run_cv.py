"""Run local cross-validation for the configured forecasting experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stai_x_forecasting.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/validation.yaml", help="Validation config path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(Path(args.config))
    print(f"Loaded validation config: {config['name']}")
    print("Cross-validation runner scaffold is ready for competition data.")


if __name__ == "__main__":
    main()
