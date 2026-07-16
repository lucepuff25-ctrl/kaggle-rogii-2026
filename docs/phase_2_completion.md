# ROGII Stage 2 Completion

Date: 2026-07-16

## Outcome

Stage 2 is complete. The project moved from a row-wise LightGBM baseline to a competition-mode geosteering ensemble and established a new verified recovery point.

Promoted solution:

- PF/ANCC multi-scale geosteering and Beam Search candidates;
- a learned LightGBM/CatBoost residual track stacked by Ridge;
- fixed physical/learned blending and robust low-order trajectory projection;
- private, Internet-off Kaggle Notebook execution with explicit upstream attribution.

Verified metrics:

- OOF MSE: `108.570148` (derived from rounded RMSE);
- OOF RMSE: `10.4197`;
- fold RMSE: `9.4973`, `10.8118`, `9.3836`, `11.4277`, `10.8220`;
- Kaggle submission: `54709570`, status `COMPLETE`, Public Score `7.761`;
- improvement over prior Public best `13.531`: `5.770`.

Decision: promoted as the current competitive baseline. It passes the rapid-score objective but does not meet the gold-medal objective.

## Experiments closed

- Baseline B: Public `13.623`.
- Last-known slope: Public `14.048`, rejected.
- Typewell local GR slope: Public `13.917`, rejected.
- Baseline/typewell 60/40 blend: Public `13.531`, superseded.
- PF geosteering rebuild: Public `7.761`, promoted.
- Selector CV diagnostic: Kaggle status `CANCEL_ACKNOWLEDGED`. Its 250-well selector report reached RMSE `9.274922` (MSE `86.024182`), but the full-stack ablation stopped at `175/250` wells after the runtime limit. It created no competition submission and is not an accepted model result.

## Important finding for the next stage

The selector diagnostic proves that candidate selection has headroom, but it does not prove that a learned selector generalizes. Before using the diagnostic for model selection, learned-track candidates must be regenerated strictly out of fold. A frozen full-data model can leak validation-well targets through its weights even when inference-time feature hashes are unchanged.

## Frozen safety boundary

- Raw train wells: 773; effective wells: 770.
- Quarantined public-sample wells: `000d7d20`, `00bbac68`, `00e12e8b`.
- Typewell groups: 749; folds: 5.
- Fold mapping SHA-256: `e524dfa3dd43f1a4d2105971f16a5b1b0e0135bdc4d20ea476388310146df9e1`.
- Original data remains read-only and untracked.
- No secrets are stored in the repository or operation log.

## Final acceptance

- `96` tests passed.
- Foundation validation status `ok`.
- Git object integrity check passed.
- Git diff whitespace check passed.
- Raw-data Git status is empty.
- GPU compute-process list is empty.
- Promoted Kaggle Notebook and submission are complete.
- Stage documents and structured acceptance evidence are committed before push.

Stage 2 is therefore closed. Further model changes belong to Stage 3.
