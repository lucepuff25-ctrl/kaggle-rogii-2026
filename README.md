# ROGII Wellbore Geology Prediction

Private, reproducible working repository for the Kaggle competition `rogii-wellbore-geology-prediction`.

## Current state

- Stage 2 is complete as of 2026-07-16.
- Current promoted solution: PF/ANCC geosteering + Beam Search + learned GBM/CatBoost track + Ridge stack + robust trajectory projection.
- Honest grouped OOF: MSE `108.570148`, RMSE `10.4197`.
- Official Kaggle submission `54709570`: Public Score `7.761`.
- Previous best was `13.531`; improvement is `5.770`.
- Current fold map SHA-256: `e524dfa3dd43f1a4d2105971f16a5b1b0e0135bdc4d20ea476388310146df9e1`.

The public title `7.091` of the upstream notebook is not treated as a reproducibility guarantee. Our verified score for the frozen private rebuild is `7.761`.

## Validation boundary

- 773 raw training wells; 770 effective wells.
- The three visible public-sample wells `000d7d20`, `00bbac68`, and `00e12e8b` are quarantined from fitting and validation.
- Five deterministic folds group identical typewell numeric fingerprints together.
- Hidden labels, manual labels, train/test-name lookup, and row-level random CV are forbidden.
- Competition data, credentials, model weights, submissions, and runtime logs are excluded from Git.

## Reproduce the foundation checks

```bash
conda activate kaggle-rogii
pytest -q
python scripts/validate_foundation.py
sha256sum data/processed/rogii_well_folds.csv
```

Expected result: `96 passed`, foundation status `ok`, 770 effective wells, 749 typewell groups, and the fold SHA above.

## Key records

- `docs/phase_2_completion.md`: completed-stage decision record.
- `reports/phase_2_completion_acceptance.json`: final acceptance evidence.
- `reports/fast_competitive_pf_geosteering_rebuild_v1.json`: promoted 7.761 result.
- `reports/selector_cv_diagnostic_v1.json`: cancelled diagnostic; evidence of selector headroom only.
- `docs/phase_3_plan.md`: authorized next-stage boundary.

## Next stage

Stage 3 will freeze the existing PF/Beam candidate generator and build strictly out-of-fold candidates. The primary experiment is a strongly regularized direct-MSE soft combiner, with risk regression as the comparator. No learned candidate may score a well that was visible during that candidate model's training.
