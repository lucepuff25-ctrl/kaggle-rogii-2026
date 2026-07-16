# ROGII Stage 3 Plan

## Objective

Improve the verified Public Score `7.761` by fixing candidate selection without rebuilding the expensive PF/Beam generator first.

## Single primary experiment

Build strictly out-of-fold PF, Beam, and learned-track candidates, cache them once, then train a strongly regularized soft combiner that outputs candidate weights and directly minimizes row-weighted TVT MSE.

Risk regression on `log(RMSE + 0.25)` is the fixed comparator, not a second open-ended research branch.

## Mandatory controls

1. Learned candidates for an outer validation well must come from models that never trained on that well or its typewell fingerprint group.
2. Use deterministic pseudo-cut coverage around `30/45/60/75/85%`, with fixed small perturbations and minimum prefix/suffix sizes.
3. Candidate and combiner features may use all legally visible MD/X/Y/Z/GR and typewell inputs, but never hidden TVT or derivatives from it.
4. Changing suffix TVT must not change candidate or selector-feature hashes.
5. Report candidate residual correlations, per-well winner counts, oracle RMSE, oracle headroom capture, and fallback coverage before training a complex gate.
6. Candidate-order permutation must preserve predictions apart from explicitly declared candidate-type features.
7. All thresholds are selected inside grouped nested CV, never from Public LB.

## Acceptance gate

- Full 770-well grouped OOF RMSE `<= 10.20`.
- At least four of five folds improve over the frozen `10.4197` baseline.
- No fold degrades by more than `0.10` RMSE.
- Candidate oracle provides at least 15% MSE headroom.
- The selected combiner captures at least 25% of available oracle headroom.
- NaN/Inf and leakage intersections are zero.
- Only the winning validated method receives one private Notebook submission.

## Explicit non-goals

No diffusion model, PINN, Mamba, broad hyperparameter search, blind notebook blending, manual labeling, or leaderboard probing in this stage.
