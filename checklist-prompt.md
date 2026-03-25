Read `AGENTS.md`, `data-description.md`, `CHECKLIST.md`, and the actual files under `data/` before you start.

Start every new research pass with a schema-first check:

1. inspect the raw training table
2. inspect the dictionary file
3. confirm the target columns and submission format
4. map raw column names to semantic concepts before feature engineering

For each experiment, create or reuse a dedicated folder with the pattern `scripts/expXXX/`.

Use this layout:

- code lives in `scripts/expXXX/`
- outputs live in `scripts/expXXX/outputs/`
- fold metrics live in `scripts/expXXX/outputs/fold_scores.csv`
- the OOF dataframe lives in `scripts/expXXX/outputs/oof_predictions.csv`
- requested inference artifacts live in `scripts/expXXX/outputs/submission/`

Read `CHECKLIST.md` to see which feature families are still pending.

For every pending feature family:

- test a meaningful batch of features from that family
- if the family is broad, test at least 10 genuinely distinct new features
- keep only the features that improve the current best model under the rules in `AGENTS.md`

You can test as many features as needed for each family. It is better to test enough features and reject the family than to under-test it.

Be exhaustive within each family, but stay clinically coherent. Do not generate random feature spam.

Complete one item or family at a time. Work autonomously and only stop when you have finished the list or hit a real blocker.

After each completed experiment, log outputs, metrics, and scores to `wandb` before moving on.

At minimum, each `wandb` run must contain:

- experiment name
- feature-family or hypothesis label
- model/config summary
- split strategy
- validation metric name
- combined score
- per-target scores
- fold-level scores
- acceptance or rejection decision
- paths or uploaded copies of any validation-summary outputs that were produced
- any `oof_predictions.csv`, `fold_scores.csv`, or submission artifacts produced for that run

Update `CHECKLIST.md` and `JOURNAL.md` as each family is completed, after the `wandb` run has been recorded and before moving on to the next one.

In `JOURNAL.md`, always record the corresponding `scripts/expXXX/` directory.

Treat partial progress as non-terminal. Never end the turn just to report status.

If the local raw data uses repeated visit-style suffixes such as `_v1`, `_v2`, and so on, treat them as wide tabular inputs and derive within-row summary features unless the user explicitly asks for sequence modeling.

If `wandb` is not configured in the environment, state that clearly instead of silently skipping experiment logging.

Read `AGENTS.md` carefully and follow the rules of this repo.
