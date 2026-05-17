"""Create a Kaggle submission file from a trained model artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stai_x_forecasting.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/submission.yaml", help="Submission config path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(Path(args.config))
    output_file = config["output_file"]
    print(f"Loaded submission config. Target output: {output_file}")
    print("Submission builder scaffold is ready for trained artifacts.")


if __name__ == "__main__":
    main()
