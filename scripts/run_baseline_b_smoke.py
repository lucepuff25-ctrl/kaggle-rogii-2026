#!/usr/bin/env python3
"""Run a bounded, single-fold CPU smoke test for ROGII Baseline B."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii.baseline import predict_baseline_a
from rogii.baseline_b import (
    BASELINE_B_METHOD,
    BASELINE_B_TARGET,
    SelectionLimits,
    build_bounded_fold_split,
    identifier_sha256,
    load_baseline_b_artifact,
    predict_baseline_b,
    residual_target,
    save_baseline_b_artifact,
    train_baseline_b,
    validate_baseline_b_parameters,
)
from rogii.features import BASELINE_B_FEATURE_COLUMNS, build_baseline_b_features
from rogii.io import (
    INFERENCE_COLUMNS,
    display_path,
    load_fold_mapping,
    prediction_ids,
    prediction_mask,
    read_horizontal_well,
    sha256_file,
)
from rogii.metric import mean_squared_error, root_mean_squared_error
from rogii.quarantine import (
    PUBLIC_SAMPLE_OVERLAP_WELLS,
    QUARANTINE_POLICY_VERSION,
    assert_no_public_sample_overlap,
)

EXCLUDED_FIELDS = (
    "TVT",
    "ANCC",
    "ASTNU",
    "ASTNL",
    "EGFDU",
    "EGFDL",
    "BUDA",
    "Geology",
    "well_id",
    "typewell_group",
    "fold",
)
MAX_TRAIN_LIMITS = SelectionLimits(
    max_groups=64, max_wells=96, max_prediction_rows=300_000
)
MAX_VALIDATION_LIMITS = SelectionLimits(
    max_groups=24, max_wells=32, max_prediction_rows=120_000
)


@dataclass(frozen=True)
class LoadedRows:
    features: pd.DataFrame
    truth: np.ndarray
    baseline_a_predictions: np.ndarray | None
    baseline_a_seconds: float
    ordered_row_ids_sha256: str


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _limits(payload: dict, *, maximum: SelectionLimits) -> SelectionLimits:
    if not isinstance(payload, dict) or set(payload) != {
        "max_groups",
        "max_wells",
        "max_prediction_rows",
    }:
        raise ValueError("smoke selection limit keys mismatch")
    limits = SelectionLimits(
        max_groups=payload["max_groups"],
        max_wells=payload["max_wells"],
        max_prediction_rows=payload["max_prediction_rows"],
    )
    limits.validate()
    if (
        limits.max_groups > maximum.max_groups
        or limits.max_wells > maximum.max_wells
        or limits.max_prediction_rows > maximum.max_prediction_rows
    ):
        raise ValueError("requested selection exceeds the hard smoke boundary")
    return limits


def _load_rows(
    mapping: pd.DataFrame,
    train_dir: Path,
    *,
    include_baseline_a: bool,
) -> LoadedRows:
    assert_no_public_sample_overlap(
        mapping["well_id"], context="Baseline B file loading"
    )
    feature_parts: list[pd.DataFrame] = []
    truth_parts: list[np.ndarray] = []
    baseline_parts: list[np.ndarray] = []
    row_ids: list[str] = []
    baseline_seconds = 0.0
    for row in mapping.itertuples(index=False):
        source = train_dir / f"{row.well_id}__horizontal_well.csv"
        frame = read_horizontal_well(source, include_target=True)
        mask = prediction_mask(frame)
        actual_rows = int(mask.sum())
        if actual_rows != int(row.prediction_rows):
            raise ValueError(
                f"prediction rows for {row.well_id}: actual={actual_rows}, "
                f"mapping={row.prediction_rows}"
            )
        inference_frame = frame.loc[:, list(INFERENCE_COLUMNS)]
        features = build_baseline_b_features(inference_frame)
        truth = frame.loc[mask, "TVT"].to_numpy(dtype=np.float64)
        if len(features) != len(truth):
            raise ValueError(f"feature/target row mismatch for {row.well_id}")
        feature_parts.append(features)
        truth_parts.append(truth)
        row_ids.extend(prediction_ids(row.well_id, inference_frame).tolist())
        if include_baseline_a:
            started = time.perf_counter()
            baseline_parts.append(predict_baseline_a(inference_frame))
            baseline_seconds += time.perf_counter() - started

    features = pd.concat(feature_parts, ignore_index=True)
    truth = np.concatenate(truth_parts)
    baseline = np.concatenate(baseline_parts) if include_baseline_a else None
    return LoadedRows(
        features=features,
        truth=truth,
        baseline_a_predictions=baseline,
        baseline_a_seconds=baseline_seconds,
        ordered_row_ids_sha256=hashlib.sha256(
            ("\n".join(row_ids) + "\n").encode("utf-8")
        ).hexdigest(),
    )


def _selection_summary(mapping: pd.DataFrame) -> dict:
    return {
        "folds": sorted(int(value) for value in mapping["fold"].unique()),
        "groups": int(mapping["typewell_group"].nunique()),
        "wells": int(len(mapping)),
        "prediction_rows": int(mapping["prediction_rows"].sum()),
        "well_ids_sha256": identifier_sha256(mapping["well_id"]),
        "group_ids_sha256": identifier_sha256(mapping["typewell_group"]),
    }


def _peak_rss_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()

    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if config["smoke_only"] is not True:
        raise ValueError("Baseline B runner is smoke-only")
    if config["method"] != BASELINE_B_METHOD:
        raise ValueError(f"unsupported Baseline B method: {config['method']}")
    if config["quarantine_policy_version"] != QUARANTINE_POLICY_VERSION:
        raise ValueError("quarantine policy version mismatch")
    if args.profile not in config["profiles"]:
        raise ValueError(f"unknown smoke profile: {args.profile}")
    profile = config["profiles"][args.profile]
    if set(profile) != {
        "validation_fold",
        "train_limits",
        "validation_limits",
        "artifact_dir",
        "report_path",
    }:
        raise ValueError("Baseline B smoke profile keys mismatch")

    parameters = validate_baseline_b_parameters(config["parameters"])
    num_boost_round = config["num_boost_round"]
    if not isinstance(num_boost_round, int) or num_boost_round <= 0:
        raise ValueError("num_boost_round must be a positive integer")
    validation_fold = profile["validation_fold"]
    if not isinstance(validation_fold, int) or isinstance(validation_fold, bool):
        raise ValueError("validation_fold must be an integer")
    train_limits = _limits(profile["train_limits"], maximum=MAX_TRAIN_LIMITS)
    validation_limits = _limits(
        profile["validation_limits"], maximum=MAX_VALIDATION_LIMITS
    )

    mapping = load_fold_mapping(
        _path(config["fold_mapping"]),
        expected_sha256=config["fold_mapping_sha256"],
        n_splits=int(config["folds"]),
        expected_wells=int(config["effective_wells"]),
        expected_groups=int(config["typewell_groups"]),
    )
    train_mapping, validation_mapping = build_bounded_fold_split(
        mapping,
        validation_fold=validation_fold,
        train_limits=train_limits,
        validation_limits=validation_limits,
        seed=int(config["seed"]),
    )
    train_summary = _selection_summary(train_mapping)
    validation_summary = _selection_summary(validation_mapping)

    training_load_started = time.perf_counter()
    training = _load_rows(
        train_mapping, _path(config["train_dir"]), include_baseline_a=False
    )
    training_load_seconds = time.perf_counter() - training_load_started
    if len(training.features) != train_summary["prediction_rows"]:
        raise ValueError("training row total differs from bounded mapping")

    training_residuals = residual_target(training.truth, training.features)
    train_started = time.perf_counter()
    booster = train_baseline_b(
        training.features,
        training_residuals,
        parameters=parameters,
        num_boost_round=num_boost_round,
    )
    train_seconds = time.perf_counter() - train_started

    artifact_dir = _path(profile["artifact_dir"])
    model_path, manifest_path = save_baseline_b_artifact(
        booster,
        artifact_dir,
        parameters=parameters,
        num_boost_round=num_boost_round,
        fold_mapping_sha256=config["fold_mapping_sha256"],
        validation_fold=validation_fold,
        training_mapping=train_mapping,
        training_rows=len(training.features),
    )
    reloaded, manifest = load_baseline_b_artifact(
        artifact_dir,
        expected_fold_mapping_sha256=config["fold_mapping_sha256"],
        expected_parameters=parameters,
        expected_validation_fold=validation_fold,
    )

    validation_load_started = time.perf_counter()
    validation = _load_rows(
        validation_mapping, _path(config["train_dir"]), include_baseline_a=True
    )
    validation_load_seconds = time.perf_counter() - validation_load_started
    if len(validation.features) != validation_summary["prediction_rows"]:
        raise ValueError("validation row total differs from bounded mapping")

    predictions_before_reload = predict_baseline_b(booster, validation.features)
    inference_started = time.perf_counter()
    predictions = predict_baseline_b(reloaded, validation.features)
    inference_seconds = time.perf_counter() - inference_started
    reload_equal = np.array_equal(predictions_before_reload, predictions)
    if not reload_equal:
        raise ValueError("Baseline B artifact reload changed predictions")

    baseline_a_predictions = validation.baseline_a_predictions
    if baseline_a_predictions is None:
        raise ValueError("Baseline A comparison predictions are missing")
    baseline_a_mse = mean_squared_error(validation.truth, baseline_a_predictions)
    baseline_a_rmse = root_mean_squared_error(
        validation.truth, baseline_a_predictions
    )
    baseline_b_mse = mean_squared_error(validation.truth, predictions)
    baseline_b_rmse = root_mean_squared_error(validation.truth, predictions)
    mse_delta = baseline_b_mse - baseline_a_mse
    rmse_delta = baseline_b_rmse - baseline_a_rmse
    training_wells = set(train_mapping["well_id"])
    validation_wells = set(validation_mapping["well_id"])
    training_groups = set(train_mapping["typewell_group"])
    validation_groups = set(validation_mapping["typewell_group"])

    now = datetime.now(timezone.utc)
    report = {
        "run_id": f"baseline_b_{args.profile}_"
        + now.strftime("%Y%m%dT%H%M%SZ"),
        "timestamp": now.isoformat(),
        "status": "smoke_ok",
        "full_cv_run": False,
        "folds_configured": int(config["folds"]),
        "folds_run": 1,
        "trained_model": True,
        "kaggle_submitted": False,
        "profile": args.profile,
        "git_commit": _git_commit(),
        "data_hash": config["data_hash"],
        "method": BASELINE_B_METHOD,
        "target_definition": BASELINE_B_TARGET,
        "used_source_fields": list(INFERENCE_COLUMNS),
        "feature_columns": list(BASELINE_B_FEATURE_COLUMNS),
        "excluded_fields": list(EXCLUDED_FIELDS),
        "parameters": parameters,
        "num_boost_round": num_boost_round,
        "lightgbm_version": lgb.__version__,
        "cv_scheme": config["cv_scheme"],
        "validation_fold": validation_fold,
        "seed": int(config["seed"]),
        "selection_rule": "seeded_sha256_group_order_round_robin_by_fold",
        "train_limits": profile["train_limits"],
        "validation_limits": profile["validation_limits"],
        "training": train_summary,
        "validation": validation_summary,
        "boundary_audit": {
            "training_validation_well_overlap": len(
                training_wells & validation_wells
            ),
            "training_validation_group_overlap": len(
                training_groups & validation_groups
            ),
            "quarantined_training_wells": len(
                training_wells & PUBLIC_SAMPLE_OVERLAP_WELLS
            ),
            "quarantined_validation_wells": len(
                validation_wells & PUBLIC_SAMPLE_OVERLAP_WELLS
            ),
        },
        "comparison": {
            "prediction_rows": len(validation.truth),
            "ordered_row_ids_sha256": validation.ordered_row_ids_sha256,
            "same_validation_rows": True,
        },
        "baseline_a": {
            "mse": baseline_a_mse,
            "rmse": baseline_a_rmse,
            "inference_seconds": validation.baseline_a_seconds,
            "prediction_rows": len(baseline_a_predictions),
            "nan_count": int(np.isnan(baseline_a_predictions).sum()),
            "inf_count": int(np.isinf(baseline_a_predictions).sum()),
        },
        "baseline_b": {
            "mse": baseline_b_mse,
            "rmse": baseline_b_rmse,
            "mse_delta_vs_baseline_a": mse_delta,
            "rmse_delta_vs_baseline_a": rmse_delta,
            "mse_relative_change_percent": 100.0 * mse_delta / baseline_a_mse,
            "inference_seconds": inference_seconds,
            "prediction_rows": len(predictions),
            "nan_count": int(np.isnan(predictions).sum()),
            "inf_count": int(np.isinf(predictions).sum()),
        },
        "data_prepare_seconds": training_load_seconds + validation_load_seconds,
        "training_prepare_seconds": training_load_seconds,
        "validation_prepare_seconds": validation_load_seconds,
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "peak_rss_estimate_mib": _peak_rss_mib(),
        "gpu": "none",
        "artifact_model_path": display_path(model_path, relative_to=ROOT),
        "artifact_model_sha256": sha256_file(model_path),
        "artifact_manifest_path": display_path(manifest_path, relative_to=ROOT),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "artifact_reload_predictions_equal": reload_equal,
        "artifact_training_well_ids_sha256": manifest[
            "training_well_ids_sha256"
        ],
        "fold_mapping_sha256": config["fold_mapping_sha256"],
        "quarantine_policy_version": QUARANTINE_POLICY_VERSION,
    }
    output = _path(profile["report_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
