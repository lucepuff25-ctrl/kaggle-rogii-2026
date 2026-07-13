from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import rogii.baseline_b_runtime as runtime
from rogii.baseline_b import (
    load_baseline_b_artifact,
    save_baseline_b_artifact,
    train_baseline_b,
)
from rogii.baseline_b_full import (
    BASELINE_B_FIXED_NUM_BOOST_ROUND,
    BASELINE_B_FIXED_PARAMETERS,
    BASELINE_B_FIXED_SEED,
    validate_full_run_contract,
)
from rogii.baseline_b_runtime import ResourceLimits, guard_resources
from rogii.features import BASELINE_B_FEATURE_COLUMNS
from rogii.io import FOLD_COLUMNS
from run_baseline_b_cv import estimate_resources


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


def test_formal_contract_rejects_any_smoke_parameter_drift() -> None:
    validate_full_run_contract(
        parameters=BASELINE_B_FIXED_PARAMETERS,
        num_boost_round=BASELINE_B_FIXED_NUM_BOOST_ROUND,
        seed=BASELINE_B_FIXED_SEED,
    )

    changed = dict(BASELINE_B_FIXED_PARAMETERS)
    changed["learning_rate"] = 0.1
    with pytest.raises(ValueError, match="frozen smoke contract"):
        validate_full_run_contract(
            parameters=changed,
            num_boost_round=BASELINE_B_FIXED_NUM_BOOST_ROUND,
            seed=BASELINE_B_FIXED_SEED,
        )
    with pytest.raises(ValueError, match="remain 64"):
        validate_full_run_contract(
            parameters=BASELINE_B_FIXED_PARAMETERS,
            num_boost_round=63,
            seed=BASELINE_B_FIXED_SEED,
        )
    with pytest.raises(ValueError, match="remain 20260713"):
        validate_full_run_contract(
            parameters=BASELINE_B_FIXED_PARAMETERS,
            num_boost_round=BASELINE_B_FIXED_NUM_BOOST_ROUND,
            seed=7,
        )


def test_resource_estimate_scales_from_bounded_smoke() -> None:
    mapping = pd.DataFrame(
        {
            "fold": [0, 1, 2, 3, 4],
            "prediction_rows": [100, 110, 120, 130, 140],
        }
    )
    smoke = {
        "run_id": "bounded",
        "training": {"prediction_rows": 200},
        "validation": {"prediction_rows": 50},
        "train_seconds": 2.0,
        "data_prepare_seconds": 1.0,
        "inference_seconds": 0.25,
        "peak_rss_estimate_mib": 256.0,
        "parameters": {"num_threads": 4},
    }

    estimate = estimate_resources(mapping, smoke)

    assert estimate["max_formal_training_rows"] == 500
    assert estimate["max_formal_validation_rows"] == 140
    assert estimate["estimated_per_fold_wall_seconds"] > 0
    assert estimate["estimated_total_cpu_seconds_upper"] == pytest.approx(
        estimate["estimated_total_wall_seconds"] * 4
    )


def test_resource_guard_fails_closed_on_memory_or_system_load(monkeypatch) -> None:
    limits = ResourceLimits(100.0, 900.0, 32.0, 2.0)
    monkeypatch.setattr(
        runtime,
        "resource_snapshot",
        lambda: {
            "peak_rss_mib": 101.0,
            "available_ram_gib": 100.0,
            "load_average_1m": 1.0,
            "logical_cpus": 4,
        },
    )
    with pytest.raises(RuntimeError, match="peak RSS"):
        guard_resources(limits, context="test")

    monkeypatch.setattr(
        runtime,
        "resource_snapshot",
        lambda: {
            "peak_rss_mib": 50.0,
            "available_ram_gib": 100.0,
            "load_average_1m": 9.0,
            "logical_cpus": 4,
        },
    )
    with pytest.raises(RuntimeError, match="load per CPU"):
        guard_resources(limits, context="test")


def test_final_artifact_has_explicit_all_effective_well_scope(
    tmp_path: Path,
) -> None:
    features = feature_frame()
    residuals = 0.2 * features["row_offset"].to_numpy()
    parameters = dict(BASELINE_B_FIXED_PARAMETERS)
    parameters["min_data_in_leaf"] = 1
    parameters["num_threads"] = 1
    booster = train_baseline_b(
        features,
        residuals,
        parameters=parameters,
        num_boost_round=4,
    )
    mapping = pd.DataFrame(
        [
            {
                "well_id": "honest_final_well",
                "typewell_group": "honest_final_group",
                "fold": 0,
                "prediction_rows": len(features),
            }
        ],
        columns=FOLD_COLUMNS,
    )
    save_baseline_b_artifact(
        booster,
        tmp_path,
        parameters=parameters,
        num_boost_round=4,
        fold_mapping_sha256="a" * 64,
        validation_fold=None,
        training_mapping=mapping,
        training_rows=len(features),
    )
    _, manifest = load_baseline_b_artifact(
        tmp_path,
        expected_fold_mapping_sha256="a" * 64,
        expected_parameters=parameters,
        expected_validation_fold=None,
    )

    assert manifest["training_scope"] == "all_effective_wells"
    assert manifest["validation_fold"] is None
