# Experiment Layout

All experiments should follow this layout:

```text
scripts/
  exp001/
    train.py
    feature_engineering.py
    outputs/
      validation_summary.json
      metrics.json
      fold_scores.csv
      oof_predictions.csv
      feature_importance.csv
      submission/
        submission.csv
        test_predictions.csv
```

Rules:

- `scripts/expXXX/` contains the code for one experiment
- `scripts/expXXX/outputs/` contains the outputs for that experiment
- save validation outputs inside that experiment's `outputs/` directory
- save per-fold validation scores in `outputs/fold_scores.csv`
- save the OOF dataframe in `outputs/oof_predictions.csv`
- save inference artifacts in `outputs/submission/`
- build `outputs/submission/submission.csv` against `data/hello_world_submission-1773575379610.csv`
- when an experiment is logged to `wandb`, point to files from that experiment's `outputs/` directory
