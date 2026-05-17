# Award Strategy

## Leaderboard Forecasting

- Build robust rolling-origin validation before optimizing models.
- Start with transparent baselines, then add tree ensembles and time-series
  models only when they beat baselines consistently.
- Track every candidate submission in `experiments/registry.csv`.

## AI Automation

- Keep repeatable scripts for data checks, cross-validation, training,
  submission generation, and report creation.
- Save configuration, metrics, and artifacts for each experiment.
- Use agent logs to document automated decisions and rejected hypotheses.

## Statistical Agents

- Implement narrow, auditable agents rather than opaque general chat flows.
- Give each agent a clear input contract, output file, and review checklist.
- Prioritize leakage detection, residual analysis, uncertainty calibration, and
  model critique.
