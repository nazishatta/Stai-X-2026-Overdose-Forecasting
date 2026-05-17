"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return an empty dict for blank files."""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}
