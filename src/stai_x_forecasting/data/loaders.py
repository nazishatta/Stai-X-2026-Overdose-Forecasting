"""Load competition CSV files directly from the local Kaggle zip."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

REQUIRED_CSV_FILES = (
    "sample_submission.csv",
    "train/covariates.csv",
    "train/dose_sys_train.csv",
    "val/covariates.csv",
)


def _read_env_value(env_path: Path, key: str) -> str:
    if not env_path.exists():
        raise FileNotFoundError(
            f"Missing .env file at {env_path}. Add {key}=path/to/stai-x.zip to .env."
        )

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        if name.strip() == key:
            value = value.strip().strip('"').strip("'")
            if value:
                return value
            raise ValueError(f"{key} is present in {env_path}, but it is empty.")

    raise KeyError(f"{key} is missing from {env_path}.")


def get_stai_zip_path(env_path: Path | str = ".env") -> Path:
    """Return the STAI Kaggle zip path configured in .env."""
    zip_path = Path(_read_env_value(Path(env_path), "STAI_ZIP_FILE")).expanduser()
    if not zip_path.exists():
        raise FileNotFoundError(f"STAI_ZIP_FILE points to a missing zip file: {zip_path}")
    if not zip_path.is_file():
        raise FileNotFoundError(f"STAI_ZIP_FILE is not a file: {zip_path}")
    return zip_path


def load_competition_data(env_path: Path | str = ".env") -> dict[str, pd.DataFrame]:
    """Read required competition CSVs from the zip without extracting it."""
    zip_path = get_stai_zip_path(env_path)

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            available_files = set(archive.namelist())
            missing_files = [file for file in REQUIRED_CSV_FILES if file not in available_files]
            if missing_files:
                joined = ", ".join(missing_files)
                raise FileNotFoundError(f"Missing required CSV file(s) in {zip_path}: {joined}")

            loaded = {}
            for csv_file in REQUIRED_CSV_FILES:
                with archive.open(csv_file) as file:
                    loaded[csv_file] = pd.read_csv(file)
            return loaded
    except zipfile.BadZipFile as error:
        raise zipfile.BadZipFile(f"STAI_ZIP_FILE is not a valid zip file: {zip_path}") from error
