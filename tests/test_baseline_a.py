import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rogii.baseline import (
    BASELINE_A_ARTIFACT_TYPE,
    BASELINE_A_METHOD,
    BASELINE_A_USED_FIELDS,
    BaselineAAlgorithm,
    load_baseline_a_algorithm,
    predict_baseline_a,
    save_baseline_a_algorithm,
)
from rogii.io import sha256_file
from rogii.quarantine import QUARANTINE_POLICY_VERSION
from run_baseline_a_cv import evaluate_cv


def test_last_known_value_is_extended_deterministically() -> None:
    frame = pd.DataFrame({"TVT_input": [10.0, 11.5, np.nan, np.nan, np.nan]})
    first = predict_baseline_a(frame)
    second = predict_baseline_a(frame.copy())
    np.testing.assert_array_equal(first, np.array([11.5, 11.5, 11.5]))
    np.testing.assert_array_equal(first, second)


def test_targets_and_forbidden_fields_cannot_change_predictions() -> None:
    base = pd.DataFrame(
        {
            "TVT_input": [100.0, 101.0, np.nan, np.nan],
            "TVT": [100.0, 101.0, 102.0, 103.0],
            "ANCC": [1.0, 2.0, 3.0, 4.0],
            "ASTNU": [5.0, 6.0, 7.0, 8.0],
            "ASTNL": [9.0, 10.0, 11.0, 12.0],
            "EGFDU": [13.0, 14.0, 15.0, 16.0],
            "EGFDL": [17.0, 18.0, 19.0, 20.0],
            "BUDA": [21.0, 22.0, 23.0, 24.0],
            "Geology": ["a", "b", "c", "d"],
        }
    )
    changed = base.copy()
    changed.loc[:, ["TVT", "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]] = (
        -999999.0
    )
    changed["Geology"] = "changed"
    np.testing.assert_array_equal(predict_baseline_a(base), predict_baseline_a(changed))


def test_missing_known_prefix_and_internal_gap_are_rejected() -> None:
    with pytest.raises(ValueError, match="known"):
        predict_baseline_a(pd.DataFrame({"TVT_input": [np.nan, np.nan]}))
    with pytest.raises(ValueError, match="contiguous"):
        predict_baseline_a(pd.DataFrame({"TVT_input": [1.0, np.nan, 2.0]}))


def test_algorithm_artifact_round_trip_is_minimal_and_bitwise_equal(
    tmp_path: Path,
) -> None:
    fold_sha = "a" * 64
    artifact = BaselineAAlgorithm.create(fold_mapping_sha256=fold_sha)
    path = save_baseline_a_algorithm(artifact, tmp_path / "algorithm.json")
    loaded = load_baseline_a_algorithm(path, expected_fold_mapping_sha256=fold_sha)
    frame = pd.DataFrame({"TVT_input": [10.0, 11.0, np.nan, np.nan]})
    np.testing.assert_array_equal(artifact.predict(frame), loaded.predict(frame))
    second_path = save_baseline_a_algorithm(loaded, tmp_path / "algorithm-second.json")
    assert sha256_file(path) == sha256_file(second_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "artifact_type": BASELINE_A_ARTIFACT_TYPE,
        "schema_version": 1,
        "method": BASELINE_A_METHOD,
        "used_fields": list(BASELINE_A_USED_FIELDS),
        "fold_mapping_sha256": fold_sha,
        "quarantine_policy_version": QUARANTINE_POLICY_VERSION,
    }
    serialized = path.read_text(encoding="utf-8")
    assert "well_id" not in serialized
    assert "fallback" not in serialized
    assert "target" not in serialized


def test_algorithm_artifact_rejects_fold_sha_or_extra_state(tmp_path: Path) -> None:
    path = tmp_path / "algorithm.json"
    artifact = BaselineAAlgorithm.create(fold_mapping_sha256="b" * 64)
    save_baseline_a_algorithm(artifact, path)
    with pytest.raises(ValueError, match="fold SHA256 mismatch"):
        load_baseline_a_algorithm(path, expected_fold_mapping_sha256="c" * 64)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["target_mean"] = 123.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="keys mismatch"):
        load_baseline_a_algorithm(path, expected_fold_mapping_sha256="b" * 64)


def _write_cv_well(path: Path, *, prediction_rows: int, error: float) -> None:
    anchor = 100.0
    rows = 2 + prediction_rows
    tvt_input = [99.0, anchor] + [np.nan] * prediction_rows
    target = [99.0, anchor] + [anchor + error] * prediction_rows
    pd.DataFrame(
        {
            "MD": np.arange(rows, dtype=float),
            "X": np.ones(rows),
            "Y": np.ones(rows) * 2,
            "Z": -np.arange(rows, dtype=float),
            "GR": np.ones(rows) * 30,
            "TVT_input": tvt_input,
            "TVT": target,
        }
    ).to_csv(path, index=False)


def test_full_oof_score_is_row_weighted_not_mean_of_fold_scores(tmp_path: Path) -> None:
    prediction_rows = [1, 1, 1, 1, 6]
    errors = [0.0, 1.0, 2.0, 3.0, 10.0]
    records = []
    for fold, (rows, error) in enumerate(zip(prediction_rows, errors, strict=True)):
        well_id = f"honest{fold}"
        _write_cv_well(
            tmp_path / f"{well_id}__horizontal_well.csv",
            prediction_rows=rows,
            error=error,
        )
        records.append(
            {
                "well_id": well_id,
                "typewell_group": f"group{fold}",
                "fold": fold,
                "prediction_rows": rows,
            }
        )
    mapping = pd.DataFrame.from_records(records)
    result = evaluate_cv(mapping, tmp_path, n_splits=5)
    assert result["oof_prediction_rows"] == 10
    assert result["cv_mse"] == pytest.approx(61.4)
    assert result["cv_rmse"] == pytest.approx(np.sqrt(61.4))
    assert result["cv_mse"] != pytest.approx(
        np.mean([item["mse"] for item in result["fold_metrics"]])
    )
    assert result["nan_count"] == 0
    assert result["inf_count"] == 0
