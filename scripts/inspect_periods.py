"""Inspect period_id structure in zipped competition files."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stai_x_forecasting.data.loaders import load_competition_data


def ordered_unique_periods(frame) -> list[str]:
    return frame["period_id"].drop_duplicates().tolist()


def print_period_report(name: str, frame, print_counts: bool = True) -> None:
    periods = ordered_unique_periods(frame)

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(f"Unique period_id count: {len(periods)}")
    print("\nperiod_id values in first-seen order:")
    print(periods)
    print("\nFirst 10 period_id values by file order:")
    print(periods[:10])
    print("\nLast 10 period_id values by file order:")
    print(periods[-10:])

    if print_counts:
        print("\nCounts by period_id:")
        print(frame["period_id"].value_counts(sort=False))


def main() -> None:
    dataframes = load_competition_data()

    print_period_report("train/dose_sys_train.csv", dataframes["train/dose_sys_train.csv"])
    print_period_report("train/covariates.csv", dataframes["train/covariates.csv"])
    print_period_report("val/covariates.csv", dataframes["val/covariates.csv"])
    print_period_report("sample_submission.csv", dataframes["sample_submission.csv"], print_counts=False)


if __name__ == "__main__":
    main()
