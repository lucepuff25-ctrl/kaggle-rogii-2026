from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rogii.baseline_b import (
    SelectionLimits,
    build_bounded_fold_split,
    load_baseline_b_artifact,
    predict_baseline_b,
    reconstruct_tvt,
    residual_target,
    save_baseline_b_artifact,
    train_baseline_b,
    validate_baseline_b_parameters,
)
from rogii.features import BASELINE_B_FEATURE_COLUMNS
from rogii.io import FOLD_COLUMNS
from rogii.quarantine import PUBLIC_SAMPLE_OVERLAP_WELLS


def parameters() -> dict:
    return {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 7,
        "min_data_in_leaf": 1,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "device_type": "cpu",
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": 1,
        "seed": 20260713,
        "verbosity": -1,
    }


def feature_frame(rows: int = 64) -> pd.DataFrame:
    offset = np.arange(1, rows + 1, dtype=np.float64)
    return pd.DataFrame(
        {
            "anchor_tvt_input": np.full(rows, 100.0),
            "row_offset": offset,
            "prediction_fraction": offset / rows,
            "md_delta_anchor": offset * 0.5,
            "x_delta_anchor": offset * 0.1,
            "y_delta_anchor": offset * -0.2,
            "z_delta_anchor": offset * -0.4,
            "gr": 50.0 + offset,
            "gr_delta_anchor": offset,
            "gr_missing": np.zeros(rows),
            "anchor_gr_missing": np.zeros(rows),
        },
        columns=BASELINE_B_FEATURE_COLUMNS,
    )


def fold_mapping() -> pd.DataFrame:
    rows = []
    for fold in range(5):
        rows.append(
            {
                "well_id": f"honest{fold}",
                "typewell_group": f"group{fold}",
                "fold": fold,
                "prediction_rows": 5,
            }
        )
    return pd.DataFrame(rows, columns=FOLD_COLUMNS)


def test_residual_target_and_anchor_reconstruction_are_exact() -> None:
    features = feature_frame(3)
    truth = np.array([101.0, 103.0, 106.0])

    residuals = residual_target(truth, features)
    np.testing.assert_array_equal(residuals, [1.0, 3.0, 6.0])
    np.testing.assert_array_equal(reconstruct_tvt(residuals, features), truth)


def test_parameters_fail_closed_on_gpu_or_parameter_drift() -> None:
    gpu = parameters()
    gpu["device_type"] = "gpu"
    with pytest.raises(ValueError, match="must be cpu"):
        validate_baseline_b_parameters(gpu)

    extra = parameters()
    extra["max_depth"] = 4
    with pytest.raises(ValueError, match="keys mismatch"):
        validate_baseline_b_parameters(extra)


def test_lightgbm_handles_gr_nan_without_fitted_imputation() -> None:
    features = feature_frame()
    features.loc[::3, "gr"] = np.nan
    features.loc[::3, "gr_delta_anchor"] = np.nan
    residuals = 0.1 * features["row_offset"].to_numpy()
    booster = train_baseline_b(
        features,
        residuals,
        parameters=parameters(),
        num_boost_round=4,
    )

    assert np.isfinite(predict_baseline_b(booster, features)).all()


def test_bounded_split_uses_training_folds_only_and_keeps_groups_whole() -> None:
    train, validation = build_bounded_fold_split(
        fold_mapping(),
        validation_fold=0,
        train_limits=SelectionLimits(4, 4, 20),
        validation_limits=SelectionLimits(1, 1, 5),
        seed=7,
    )

    assert set(train["fold"]) == {1, 2, 3, 4}
    assert set(validation["fold"]) == {0}
    assert set(train["well_id"]).isdisjoint(validation["well_id"])
    assert set(train["typewell_group"]).isdisjoint(validation["typewell_group"])
    assert train["prediction_rows"].sum() == 20
    assert validation["prediction_rows"].sum() == 5


def test_bounded_split_rejects_quarantine_before_training() -> None:
    mapping = fold_mapping()
    mapping.loc[1, "well_id"] = sorted(PUBLIC_SAMPLE_OVERLAP_WELLS)[0]
    with pytest.raises(ValueError, match="quarantined"):
        build_bounded_fold_split(
            mapping,
            validation_fold=0,
            train_limits=SelectionLimits(4, 4, 20),
            validation_limits=SelectionLimits(1, 1, 5),
            seed=7,
        )


def test_bounded_split_rejects_typewell_group_crossing_fold_boundary() -> None:
    mapping = fold_mapping()
    mapping.loc[1, "typewell_group"] = mapping.loc[0, "typewell_group"]
    with pytest.raises(ValueError, match="crosses folds"):
        build_bounded_fold_split(
            mapping,
            validation_fold=0,
            train_limits=SelectionLimits(4, 4, 20),
            validation_limits=SelectionLimits(1, 1, 5),
            seed=7,
        )


def _save_test_artifact(tmp_path: Path):
    features = feature_frame()
    residuals = 0.2 * features["row_offset"].to_numpy()
    booster = train_baseline_b(
        features,
        residuals,
        parameters=parameters(),
        num_boost_round=8,
    )
    mapping = pd.DataFrame(
        [
            {
                "well_id": "honest_training_well",
                "typewell_group": "honest_training_group",
                "fold": 1,
                "prediction_rows": len(features),
            }
        ],
        columns=FOLD_COLUMNS,
    )
    model_path, manifest_path = save_baseline_b_artifact(
        booster,
        tmp_path,
        parameters=parameters(),
        num_boost_round=8,
        fold_mapping_sha256="a" * 64,
        validation_fold=0,
        training_mapping=mapping,
        training_rows=len(features),
    )
    return features, booster, model_path, manifest_path


def test_model_save_reload_predictions_are_bitwise_equal(tmp_path: Path) -> None:
    features, booster, _, manifest_path = _save_test_artifact(tmp_path)
    reloaded, manifest = load_baseline_b_artifact(
        tmp_path,
        expected_fold_mapping_sha256="a" * 64,
        expected_parameters=parameters(),
        expected_validation_fold=0,
    )

    np.testing.assert_array_equal(
        predict_baseline_b(booster, features),
        predict_baseline_b(reloaded, features),
    )
    text = manifest_path.read_text(encoding="utf-8")
    assert "honest_training_well" not in text
    assert "honest_training_group" not in text
    assert manifest["training_wells"] == 1


def test_model_or_feature_manifest_tampering_is_rejected(tmp_path: Path) -> None:
    _, _, model_path, manifest_path = _save_test_artifact(tmp_path)
    model_path.write_bytes(model_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="model SHA256 mismatch"):
        load_baseline_b_artifact(
            tmp_path,
            expected_fold_mapping_sha256="a" * 64,
            expected_parameters=parameters(),
            expected_validation_fold=0,
        )

    _, _, _, manifest_path = _save_test_artifact(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_columns"] = list(reversed(manifest["feature_columns"]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="feature columns mismatch"):
        load_baseline_b_artifact(
            tmp_path,
            expected_fold_mapping_sha256="a" * 64,
            expected_parameters=parameters(),
            expected_validation_fold=0,
        )
