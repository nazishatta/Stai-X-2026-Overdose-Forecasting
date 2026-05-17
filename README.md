# STAI-X 2026 Overdose Forecasting

Solo Kaggle competition workspace for forecasting state-level overdose emergency
department visit rates in the STAI-X Challenge 2026.

The repository is organized around all three award tracks:

- **Leaderboard forecasting:** reproducible feature engineering, validation,
  temporal models, ensembling, and submission generation.
- **AI automation:** automated experiment orchestration, report generation, and
  repeatable Kaggle workflows.
- **Statistical agents:** auditable agents for diagnostics, model critique,
  uncertainty review, and hypothesis-driven analysis.

## Project Layout

```text
.
├── configs/                 # Dataset, validation, model, and agent settings
├── data/                    # Local-only competition data and derived artifacts
├── docs/                    # Competition notes, award strategy, and reports
├── experiments/             # Human-readable experiment registry
├── notebooks/               # Numbered exploration and modeling notebooks
├── reports/                 # Figures, tables, and generated narratives
├── scripts/                 # CLI wrappers for common workflows
├── src/stai_x_forecasting/  # Reusable project package
├── submissions/             # Kaggle submission files and manifests
└── tests/                   # Unit tests for reusable logic
```

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Run the starter validation check:

```powershell
python scripts/run_cv.py --config configs/validation.yaml
```

Generate a submission once competition files are available locally:

```powershell
python scripts/make_submission.py --config configs/submission.yaml
```

## Data Policy

Competition data, intermediate feature stores, model artifacts, and submissions
can be large or restricted by Kaggle rules. This repo tracks directory
structure and metadata, but ignores the actual data and generated outputs by
default. Place raw Kaggle files under `data/raw/`.
