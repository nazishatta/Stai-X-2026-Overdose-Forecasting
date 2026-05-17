# Competition Brief

## Objective

Forecast state-level overdose emergency department visit rates for the STAI-X
Challenge 2026 Kaggle competition.

## Working Assumptions

- The prediction target is a rate, so negative predictions should be clipped or
  transformed back from a non-negative modeling scale.
- Validation must respect time ordering and state-level grouping.
- Public leaderboard feedback may be noisy; model selection should prioritize
  local validation stability and error diagnostics.

## Key Risks

- Temporal leakage from future covariates, target-derived rolling features, or
  improperly aligned state aggregates.
- State-level distribution shifts caused by reporting changes or rare events.
- Overfitting to the public leaderboard with too many submissions.
