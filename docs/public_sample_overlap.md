# ROGII public sample overlap quarantine

## Finding

The downloaded public `test/` directory contains three wells that also exist in
`train/`:

- `000d7d20`
- `00bbac68`
- `00e12e8b`

Across these wells, all 19,221 rows match on `MD,X,Y,Z,GR,TVT_input`. The train
copies also contain `TVT` for all 14,151 public-test prediction rows.

## Interpretation

This is recorded as a **public sample overlap**, not as confirmed leaderboard
label leakage. ROGII is notebook-only, so Kaggle may replace the public sample
with hidden wells during scoring. The public overlap must not be used as evidence
that hidden labels are available.

## Policy

The three wells are quarantined from:

- model fitting;
- cross-validation;
- target statistics and target-derived features;
- feature selection and preprocessing fitting;
- experiment comparison;
- copying train `TVT` values into a formal submission.

They may only be used to validate file reading, inference plumbing, ID alignment,
and submission schema. Code must call the helpers in `src/rogii/quarantine.py`
and fail closed before honest training or validation.

The finding is retained for reproducibility and organizer clarification. It is
not an approved competition strategy.
