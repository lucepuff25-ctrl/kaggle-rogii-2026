#!/usr/bin/env python3
"""Train the final all-well Baseline B model and build a local submission."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii.baseline_b import (
    BASELINE_B_METHOD,
    BASELINE_B_TARGET,
    identifier_sha256,
    load_baseline_b_artifact,
    predict_baseline_b,
    residual_target,
    save_baseline_b_artifact,
    train_baseline_b,
)
from rogii.baseline_b_full import validate_full_run_contract
from rogii.baseline_b_runtime import (
    ResourceLimits,
    guard_resources,
    load_mapping_rows,
    mapping_summary,
    peak_rss_mib,
)
from rogii.features import (
    BASELINE_B_FEATURE_COLUMNS,
    LAST_KNOWN_SLOPE_FEATURE_COLUMNS,
    TYPEWELL_GR_SLOPE_FEATURE_COLUMNS,
    TYPEWELL_PRIOR_FEATURE_COLUMNS,
    build_baseline_b_features,
    build_last_known_slope_features,
    build_typewell_gr_slope_features,
    build_typewell_prior_features,
)
from rogii.io import (
    INFERENCE_COLUMNS,
    discover_horizontal_wells,
    display_path,
    load_fold_mapping,
    read_horizontal_well,
    sha256_file,
)
from rogii.quarantine import QUARANTINE_POLICY_VERSION, assert_no_public_sample_overlap
from rogii.submission import (
    align_predictions_to_sample,
    validate_submission_file,
    well_prediction_frame,
    write_submission,
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


def _limits(config: dict) -> ResourceLimits:
    limits = ResourceLimits(**config["resource_limits"])
    limits.validate()
    return limits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if config["method"] != BASELINE_B_METHOD:
        raise ValueError("formal Baseline B method mismatch")
    if config["quarantine_policy_version"] != QUARANTINE_POLICY_VERSION:
        raise ValueError("quarantine policy version mismatch")
    validate_full_run_contract(
        parameters=config["parameters"],
        num_boost_round=config["num_boost_round"],
        seed=config["seed"],
    )
    use_slope = config.get("use_last_known_slope", False)
    use_typewell = config.get("use_typewell_tvt_prior", False)
    use_typewell_gr_slope = config.get("use_typewell_gr_slope", False)
    if not all(
        isinstance(value, bool)
        for value in (use_slope, use_typewell, use_typewell_gr_slope)
    ):
        raise ValueError("single-variable feature flags must be boolean")
    slope_window = config.get("last_known_slope_window", 2)
    if (
        not isinstance(slope_window, int)
        or isinstance(slope_window, bool)
        or slope_window < 2
    ):
        raise ValueError("last_known_slope_window must be an integer of at least two")
    if sum((use_slope, use_typewell, use_typewell_gr_slope)) > 1:
        raise ValueError("single-variable features are mutually exclusive")
    feature_columns = (
        LAST_KNOWN_SLOPE_FEATURE_COLUMNS
        if use_slope
        else (
            TYPEWELL_PRIOR_FEATURE_COLUMNS
            if use_typewell
            else (
                TYPEWELL_GR_SLOPE_FEATURE_COLUMNS
                if use_typewell_gr_slope
                else BASELINE_B_FEATURE_COLUMNS
            )
        )
    )
    limits = _limits(config)
    mapping = load_fold_mapping(
        _path(config["fold_mapping"]),
        expected_sha256=config["fold_mapping_sha256"],
        n_splits=int(config["folds"]),
        expected_wells=int(config["effective_wells"]),
        expected_groups=int(config["typewell_groups"]),
    ).sort_values("well_id")
    assert_no_public_sample_overlap(
        mapping["well_id"], context="Baseline B final all-well training"
    )
    training_summary = mapping_summary(mapping)
    if training_summary != {
        "folds": [0, 1, 2, 3, 4],
        "groups": int(config["typewell_groups"]),
        "wells": int(config["effective_wells"]),
        "prediction_rows": int(config["expected_prediction_rows"]),
    }:
        raise ValueError(f"final training mapping summary mismatch: {training_summary}")

    full_started = time.perf_counter()
    cpu_started = time.process_time()
    resource_before = guard_resources(
        limits, context="Baseline B final training start", started=full_started
    )
    prepare_started = time.perf_counter()
    training = load_mapping_rows(
        mapping,
        _path(config["train_dir"]),
        include_baseline_a=False,
        hash_row_ids=False,
        limits=limits,
        stage_started=full_started,
        context="Baseline B final training load",
        use_typewell_tvt_prior=use_typewell,
        use_typewell_gr_slope=use_typewell_gr_slope,
        use_last_known_slope=use_slope,
        last_known_slope_window=slope_window,
    )
    prepare_seconds = time.perf_counter() - prepare_started
    if len(training.features) != int(config["expected_prediction_rows"]):
        raise ValueError("final training row total mismatch")
    training_residuals = residual_target(training.truth, training.features)

    train_started = time.perf_counter()
    booster = train_baseline_b(
        training.features,
        training_residuals,
        parameters=config["parameters"],
        num_boost_round=config["num_boost_round"],
    )
    train_seconds = time.perf_counter() - train_started
    guard_resources(
        limits, context="Baseline B final model trained", started=full_started
    )

    artifact_dir = _path(config["artifact_dir"])
    model_path, manifest_path = save_baseline_b_artifact(
        booster,
        artifact_dir,
        parameters=config["parameters"],
        num_boost_round=config["num_boost_round"],
        fold_mapping_sha256=config["fold_mapping_sha256"],
        validation_fold=None,
        training_mapping=mapping,
        training_rows=len(training.features),
    )
    reloaded, manifest = load_baseline_b_artifact(
        artifact_dir,
        expected_fold_mapping_sha256=config["fold_mapping_sha256"],
        expected_parameters=config["parameters"],
        expected_validation_fold=None,
        expected_feature_columns=feature_columns,
    )
    if manifest["training_scope"] != "all_effective_wells":
        raise ValueError("final artifact training scope mismatch")
    roundtrip_features = training.features.iloc[:4096].copy()
    roundtrip_equal = np.array_equal(
        predict_baseline_b(booster, roundtrip_features),
        predict_baseline_b(reloaded, roundtrip_features),
    )
    if not roundtrip_equal:
        raise ValueError("final artifact reload changed predictions")
    del training, training_residuals, roundtrip_features, booster
    gc.collect()

    submission_started = time.perf_counter()
    model_inference_seconds = 0.0
    prediction_frames: list[pd.DataFrame] = []
    test_wells = discover_horizontal_wells(_path(config["test_dir"]))
    for well in test_wells:
        horizontal = read_horizontal_well(well.path, include_target=False)
        inference_frame = horizontal.loc[:, list(INFERENCE_COLUMNS)]
        if use_slope:
            features = build_last_known_slope_features(
                inference_frame, known_window=slope_window
            )
        elif use_typewell:
            typewell_path = Path(config["test_dir"]) / f"{well.well_id}__typewell.csv"
            features = build_typewell_prior_features(
                inference_frame, pd.read_csv(_path(typewell_path), usecols=["TVT"])
            )
        elif use_typewell_gr_slope:
            typewell_path = Path(config["test_dir"]) / f"{well.well_id}__typewell.csv"
            features = build_typewell_gr_slope_features(
                inference_frame,
                pd.read_csv(_path(typewell_path), usecols=["TVT", "GR"]),
            )
        else:
            features = build_baseline_b_features(inference_frame)
        started = time.perf_counter()
        predictions = predict_baseline_b(reloaded, features)
        model_inference_seconds += time.perf_counter() - started
        prediction_frames.append(
            well_prediction_frame(well.well_id, horizontal, predictions)
        )
        guard_resources(
            limits,
            context=f"Baseline B test inference {well.well_id}",
            started=full_started,
        )
    raw_predictions = pd.concat(prediction_frames, ignore_index=True)
    sample_path = _path(config["sample_submission"])
    sample = pd.read_csv(sample_path, dtype={"id": "string"})
    submission = align_predictions_to_sample(raw_predictions, sample)
    submission_path = write_submission(
        submission, _path(config["submission_path"])
    )
    submission_pipeline_seconds = time.perf_counter() - submission_started
    submission_validation = validate_submission_file(submission_path, sample_path)
    resource_after = guard_resources(
        limits, context="Baseline B final inference complete", started=full_started
    )

    now = datetime.now(timezone.utc)
    values = submission["tvt"].to_numpy(dtype=np.float64)
    report = {
        "run_id": "baseline_b_inference_" + now.strftime("%Y%m%dT%H%M%SZ"),
        "timestamp": now.isoformat(),
        "status": "ok",
        "git_commit": _git_commit(),
        "data_hash": config["data_hash"],
        "method": BASELINE_B_METHOD,
        "target_definition": BASELINE_B_TARGET,
        "used_source_fields": list(INFERENCE_COLUMNS)
        + (["typewell.TVT"] if use_typewell else [])
        + (["typewell.TVT", "typewell.GR"] if use_typewell_gr_slope else []),
        "feature_columns": list(feature_columns),
        "excluded_fields": list(EXCLUDED_FIELDS),
        "parameters": config["parameters"],
        "num_boost_round": config["num_boost_round"],
        "last_known_slope_window": slope_window,
        "lightgbm_version": lgb.__version__,
        "seed": int(config["seed"]),
        "cv_scheme": config["cv_scheme"],
        "fold_mapping_sha256": config["fold_mapping_sha256"],
        "quarantine_policy_version": QUARANTINE_POLICY_VERSION,
        "training_scope": manifest["training_scope"],
        "training": {
            **training_summary,
            "well_ids_sha256": identifier_sha256(mapping["well_id"]),
            "group_ids_sha256": identifier_sha256(mapping["typewell_group"]),
        },
        "training_prepare_seconds": prepare_seconds,
        "train_seconds": train_seconds,
        "model_inference_seconds": model_inference_seconds,
        "submission_pipeline_seconds": submission_pipeline_seconds,
        "wall_seconds": time.perf_counter() - full_started,
        "cpu_seconds": time.process_time() - cpu_started,
        "peak_rss_mib": peak_rss_mib(),
        "resource_limits": config["resource_limits"],
        "resource_before": resource_before,
        "resource_after": resource_after,
        "resource_limits_respected": True,
        "gpu": "none",
        "trained_model": True,
        "artifact_model_path": display_path(model_path, relative_to=ROOT),
        "artifact_model_sha256": sha256_file(model_path),
        "artifact_manifest_path": display_path(manifest_path, relative_to=ROOT),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "artifact_reload_predictions_equal": roundtrip_equal,
        "artifact_validated": True,
        "test_wells": len(test_wells),
        "prediction_rows": len(submission),
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
        "submission_path": display_path(submission_path, relative_to=ROOT),
        "submission_sha256": sha256_file(submission_path),
        "submission_validation": submission_validation,
        "kaggle_submitted": False,
    }
    output = _path(config["inference_report_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
