# Experiment Protocol

1. Record the hypothesis before running the experiment.
2. Use the same validation split unless the experiment is explicitly testing
   validation design.
3. Save metrics, notes, and artifact paths in `experiments/registry.csv`.
4. Promote a model only when it improves both aggregate score and state-level
   robustness.
5. Mark any public leaderboard submission with the exact local experiment ID.
