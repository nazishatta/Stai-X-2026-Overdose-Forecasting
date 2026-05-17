.PHONY: install test lint format cv submission

install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts

cv:
	python scripts/run_cv.py --config configs/validation.yaml

submission:
	python scripts/make_submission.py --config configs/submission.yaml
