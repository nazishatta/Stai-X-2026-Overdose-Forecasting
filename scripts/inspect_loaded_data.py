"""Inspect required competition CSVs loaded directly from the Kaggle zip."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stai_x_forecasting.data.loaders import load_competition_data


def main() -> None:
    dataframes = load_competition_data()

    for name, frame in dataframes.items():
        print("\n" + "=" * 80)
        print(name)
        print("=" * 80)
        print(f"Shape: {frame.shape}")
        print("\nColumns:")
        print(frame.columns.tolist())
        print("\nMissing values:")
        print(frame.isna().sum())


if __name__ == "__main__":
    main()
