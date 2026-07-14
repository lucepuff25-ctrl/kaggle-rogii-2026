#!/usr/bin/env python3
"""Run formal, resource-bounded five-fold OOF evaluation for Baseline B."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
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
    predict_baseline_b,
    residual_target,
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
)
from rogii.io import INFERENCE_COLUMNS, load_fold_mapping
from rogii.metric import mean_squared_error, root_mean_squared_error
from rogii.quarantine import (
    PUBLIC_SAMPLE_OVERLAP_WELLS,
    QUARANTINE_POLICY_VERSION,
    assert_no_public_sample_overlap,
)

EXPECTED_BASELINE_A_FOLDS = {
    0: (271.5395445577022, 16.47845698351949),
    1: (188.15693844522872, 13.717030963194212),
    2: (248.837679035405, 15.774589662980302),
    3: (278.1934269529421, 16.679131480773876),
    4: (281.1553735985231, 16.767688379693936),
}
EXPECTED_BASELINE_A_OVERALL = (253.5737461861229, 15.923999063869694)
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
    payload = config["resource_limits"]
    expected = {
        "max_peak_rss_mib",
        "max_stage_seconds",
        "min_available_ram_gib",
        "max_load_per_cpu",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("formal resource limit keys mismatch")
    limits = ResourceLimits(**payload)
    limits.validate()
    return limits


def _validate_config(config: dict) -> None:
    if config["method"] != BASELINE_B_METHOD:
        raise ValueError("formal Baseline B method mismatch")
    if config["quarantine_policy_version"] != QUARANTINE_POLICY_VERSION:
        raise ValueError("quarantine policy version mismatch")
    if int(config["folds"]) != 5:
        raise ValueError("formal Baseline B must run exactly five folds")
    validate_full_run_contract(
        parameters=config["parameters"],
        num_boost_round=config["num_boost_round"],
        seed=config["seed"],
    )
    if not isinstance(config.get("use_typewell_tvt_prior", False), bool):
        raise ValueError("use_typewell_tvt_prior must be boolean")
    if not isinstance(config.get("use_last_known_slope", False), bool):
        raise ValueError("use_last_known_slope must be boolean")
    if not isinstance(config.get("use_typewell_gr_slope", False), bool):
        raise ValueError("use_typewell_gr_slope must be boolean")
    slope_window = config.get("last_known_slope_window", 2)
    if (
        not isinstance(slope_window, int)
        or isinstance(slope_window, bool)
        or slope_window < 2
    ):
        raise ValueError("last_known_slope_window must be an integer of at least two")
    enabled = sum(
        bool(config.get(key, False))
        for key in (
            "use_typewell_tvt_prior",
            "use_typewell_gr_slope",
            "use_last_known_slope",
        )
    )
    if enabled > 1:
        raise ValueError("single-variable features are mutually exclusive")
    _limits(config)


def estimate_resources(mapping: pd.DataFrame, smoke_report: dict) -> dict:
    total_rows = int(mapping["prediction_rows"].sum())
    fold_rows = [
        int(mapping.loc[mapping["fold"] == fold, "prediction_rows"].sum())
        for fold in sorted(mapping["fold"].unique())
    ]
    bounded_train_rows = int(smoke_report["training"]["prediction_rows"])
    bounded_validation_rows = int(smoke_report["validation"]["prediction_rows"])
    bounded_total_rows = bounded_train_rows + bounded_validation_rows
    max_training_rows = max(total_rows - rows for rows in fold_rows)
    max_validation_rows = max(fold_rows)
    prepare_linear = smoke_report["data_prepare_seconds"] * (
        total_rows / bounded_total_rows
    )
    train_linear = smoke_report["train_seconds"] * (
        max_training_rows / bounded_train_rows
    )
    inference_linear = smoke_report["inference_seconds"] * (
        max_validation_rows / bounded_validation_rows
    )
    safety_factor = 3.0
    per_fold_wall = safety_factor * (
        prepare_linear + train_linear + inference_linear
    )
    final_wall = safety_factor * (
        prepare_linear
        + smoke_report["train_seconds"] * (total_rows / bounded_train_rows)
    )
    total_wall = per_fold_wall * len(fold_rows) + final_wall
    peak_linear_mib = smoke_report["peak_rss_estimate_mib"] * (
        total_rows / bounded_total_rows
    )
    peak_conservative_mib = 1.5 * peak_linear_mib
    threads = int(smoke_report["parameters"]["num_threads"])
    return {
        "basis_report": smoke_report["run_id"],
        "basis_training_rows": bounded_train_rows,
        "basis_validation_rows": bounded_validation_rows,
        "basis_train_seconds": smoke_report["train_seconds"],
        "basis_data_prepare_seconds": smoke_report["data_prepare_seconds"],
        "basis_inference_seconds": smoke_report["inference_seconds"],
        "basis_peak_rss_mib": smoke_report["peak_rss_estimate_mib"],
        "max_formal_training_rows": max_training_rows,
        "max_formal_validation_rows": max_validation_rows,
        "linear_per_fold_wall_seconds": prepare_linear
        + train_linear
        + inference_linear,
        "safety_factor": safety_factor,
        "estimated_per_fold_wall_seconds": per_fold_wall,
        "estimated_cv_wall_seconds": per_fold_wall * len(fold_rows),
        "estimated_final_train_wall_seconds": final_wall,
        "estimated_total_wall_seconds": total_wall,
        "estimated_total_cpu_seconds_upper": total_wall * threads,
        "linear_peak_rss_mib": peak_linear_mib,
        "estimated_peak_rss_mib": peak_conservative_mib,
        "method": "bounded-smoke row-linear extrapolation with 3x time and 1.5x RSS margins",
    }


def _load_mapping(config: dict) -> pd.DataFrame:
    return load_fold_mapping(
        _path(config["fold_mapping"]),
        expected_sha256=config["fold_mapping_sha256"],
        n_splits=int(config["folds"]),
        expected_wells=int(config["effective_wells"]),
        expected_groups=int(config["typewell_groups"]),
    )


def _boundary_audit(
    training: pd.DataFrame, validation: pd.DataFrame
) -> dict[str, int]:
    training_wells = set(training["well_id"])
    validation_wells = set(validation["well_id"])
    training_groups = set(training["typewell_group"])
    validation_groups = set(validation["typewell_group"])
    audit = {
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
    }
    if any(audit.values()):
        raise ValueError(f"Baseline B fold boundary violation: {audit}")
    return audit


def _run_fold(config: dict, fold: int, output_prefix: Path) -> dict:
    limits = _limits(config)
    mapping = _load_mapping(config)
    training_mapping = mapping.loc[mapping["fold"] != fold].sort_values(
        "well_id"
    )
    validation_mapping = mapping.loc[mapping["fold"] == fold].sort_values(
        "well_id"
    )
    assert_no_public_sample_overlap(
        training_mapping["well_id"], context=f"Baseline B fold {fold} training"
    )
    assert_no_public_sample_overlap(
        validation_mapping["well_id"], context=f"Baseline B fold {fold} validation"
    )
    boundary = _boundary_audit(training_mapping, validation_mapping)

    fold_started = time.perf_counter()
    cpu_started = time.process_time()
    resource_before = guard_resources(
        limits, context=f"Baseline B fold {fold} start", started=fold_started
    )
    training_prepare_started = time.perf_counter()
    training = load_mapping_rows(
        training_mapping,
        _path(config["train_dir"]),
        include_baseline_a=False,
        hash_row_ids=False,
        limits=limits,
        stage_started=fold_started,
        context=f"Baseline B fold {fold} training load",
        use_typewell_tvt_prior=config.get("use_typewell_tvt_prior", False),
        use_typewell_gr_slope=config.get("use_typewell_gr_slope", False),
        use_last_known_slope=config.get("use_last_known_slope", False),
        last_known_slope_window=config.get("last_known_slope_window", 2),
    )
    training_prepare_seconds = time.perf_counter() - training_prepare_started
    training_summary = mapping_summary(training_mapping)
    if len(training.features) != training_summary["prediction_rows"]:
        raise ValueError(f"fold {fold} training row total mismatch")
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
        limits, context=f"Baseline B fold {fold} trained", started=fold_started
    )
    del training, training_residuals
    gc.collect()

    validation_prepare_started = time.perf_counter()
    validation = load_mapping_rows(
        validation_mapping,
        _path(config["train_dir"]),
        include_baseline_a=True,
        hash_row_ids=True,
        limits=limits,
        stage_started=fold_started,
        context=f"Baseline B fold {fold} validation load",
        use_typewell_tvt_prior=config.get("use_typewell_tvt_prior", False),
        use_typewell_gr_slope=config.get("use_typewell_gr_slope", False),
        use_last_known_slope=config.get("use_last_known_slope", False),
        last_known_slope_window=config.get("last_known_slope_window", 2),
    )
    validation_prepare_seconds = time.perf_counter() - validation_prepare_started
    validation_summary = mapping_summary(validation_mapping)
    if len(validation.features) != validation_summary["prediction_rows"]:
        raise ValueError(f"fold {fold} validation row total mismatch")

    inference_started = time.perf_counter()
    predictions = predict_baseline_b(booster, validation.features)
    inference_seconds = time.perf_counter() - inference_started
    baseline_a_predictions = validation.baseline_a_predictions
    if baseline_a_predictions is None:
        raise ValueError("Baseline A comparison predictions are missing")
    if not (
        len(validation.truth)
        == len(predictions)
        == len(baseline_a_predictions)
    ):
        raise ValueError("A/B validation row alignment mismatch")

    baseline_a_mse = mean_squared_error(validation.truth, baseline_a_predictions)
    baseline_a_rmse = root_mean_squared_error(
        validation.truth, baseline_a_predictions
    )
    expected_mse, expected_rmse = EXPECTED_BASELINE_A_FOLDS[fold]
    if not np.isclose(baseline_a_mse, expected_mse, rtol=0.0, atol=1e-12):
        raise ValueError(f"fold {fold} Baseline A MSE does not reproduce")
    if not np.isclose(baseline_a_rmse, expected_rmse, rtol=0.0, atol=1e-12):
        raise ValueError(f"fold {fold} Baseline A RMSE does not reproduce")
    baseline_b_mse = mean_squared_error(validation.truth, predictions)
    baseline_b_rmse = root_mean_squared_error(validation.truth, predictions)
    fold_seconds = time.perf_counter() - fold_started
    resource_after = guard_resources(
        limits, context=f"Baseline B fold {fold} complete", started=fold_started
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(output_prefix) + ".npz",
        truth=validation.truth,
        baseline_a=baseline_a_predictions,
        baseline_b=predictions,
    )
    result = {
        "fold": fold,
        "training": {
            **training_summary,
            "well_ids_sha256": identifier_sha256(training_mapping["well_id"]),
            "group_ids_sha256": identifier_sha256(
                training_mapping["typewell_group"]
            ),
        },
        "validation": {
            **validation_summary,
            "well_ids_sha256": identifier_sha256(validation_mapping["well_id"]),
            "group_ids_sha256": identifier_sha256(
                validation_mapping["typewell_group"]
            ),
        },
        "boundary_audit": boundary,
        "comparison": {
            "prediction_rows": len(predictions),
            "ordered_row_ids_sha256": validation.ordered_row_ids_sha256,
            "same_validation_rows": True,
        },
        "baseline_a": {
            "mse": baseline_a_mse,
            "rmse": baseline_a_rmse,
            "inference_seconds": validation.baseline_a_seconds,
            "nan_count": int(np.isnan(baseline_a_predictions).sum()),
            "inf_count": int(np.isinf(baseline_a_predictions).sum()),
        },
        "baseline_b": {
            "mse": baseline_b_mse,
            "rmse": baseline_b_rmse,
            "mse_delta_vs_baseline_a": baseline_b_mse - baseline_a_mse,
            "rmse_delta_vs_baseline_a": baseline_b_rmse - baseline_a_rmse,
            "mse_relative_change_percent": 100.0
            * (baseline_b_mse - baseline_a_mse)
            / baseline_a_mse,
            "nan_count": int(np.isnan(predictions).sum()),
            "inf_count": int(np.isinf(predictions).sum()),
        },
        "training_prepare_seconds": training_prepare_seconds,
        "validation_prepare_seconds": validation_prepare_seconds,
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "wall_seconds": fold_seconds,
        "cpu_seconds": time.process_time() - cpu_started,
        "peak_rss_mib": peak_rss_mib(),
        "resource_before": resource_before,
        "resource_after": resource_after,
        "lightgbm_dataset_scope": "training_fold_only",
        "fitted_preprocessing": "none",
    }
    Path(str(output_prefix) + ".json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _run_parent(config: dict, config_path: Path) -> dict:
    mapping = _load_mapping(config)
    assert_no_public_sample_overlap(mapping["well_id"], context="Baseline B OOF")
    estimate = estimate_resources(
        mapping,
        json.loads(_path(config["resource_estimate_basis"]).read_text(encoding="utf-8")),
    )
    limits = _limits(config)
    if estimate["estimated_peak_rss_mib"] > limits.max_peak_rss_mib:
        raise RuntimeError("estimated peak RSS exceeds the formal limit")
    if estimate["estimated_per_fold_wall_seconds"] > limits.max_stage_seconds:
        raise RuntimeError("estimated per-fold time exceeds the formal limit")

    git_commit = _git_commit()
    work_dir = ROOT / "models/baseline_b/cv_work" / git_commit
    fold_metrics: list[dict] = []
    truth_parts: list[np.ndarray] = []
    baseline_a_parts: list[np.ndarray] = []
    baseline_b_parts: list[np.ndarray] = []
    parent_started = time.perf_counter()
    for fold in range(int(config["folds"])):
        prefix = work_dir / f"fold_{fold}"
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["OMP_NUM_THREADS"] = str(config["parameters"]["num_threads"])
        environment["OPENBLAS_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(config_path),
            "--fold-worker",
            str(fold),
            "--worker-output-prefix",
            str(prefix),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=limits.max_stage_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"fold {fold} worker failed with code {completed.returncode}: "
                f"{completed.stderr[-2000:]}"
            )
        metric = json.loads(Path(str(prefix) + ".json").read_text(encoding="utf-8"))
        arrays = np.load(str(prefix) + ".npz")
        fold_metrics.append(metric)
        truth_parts.append(arrays["truth"].copy())
        baseline_a_parts.append(arrays["baseline_a"].copy())
        baseline_b_parts.append(arrays["baseline_b"].copy())
        arrays.close()

    truth = np.concatenate(truth_parts)
    baseline_a = np.concatenate(baseline_a_parts)
    baseline_b = np.concatenate(baseline_b_parts)
    expected_rows = int(config["expected_prediction_rows"])
    if len(truth) != expected_rows:
        raise ValueError(f"OOF rows={len(truth)}, expected={expected_rows}")
    if sum(item["validation"]["wells"] for item in fold_metrics) != int(
        config["effective_wells"]
    ):
        raise ValueError("OOF validation well total mismatch")
    if sum(item["validation"]["groups"] for item in fold_metrics) != int(
        config["typewell_groups"]
    ):
        raise ValueError("OOF validation group total mismatch")

    baseline_a_mse = mean_squared_error(truth, baseline_a)
    baseline_a_rmse = root_mean_squared_error(truth, baseline_a)
    if not np.isclose(
        baseline_a_mse, EXPECTED_BASELINE_A_OVERALL[0], rtol=0.0, atol=1e-12
    ):
        raise ValueError("overall Baseline A MSE does not reproduce")
    if not np.isclose(
        baseline_a_rmse, EXPECTED_BASELINE_A_OVERALL[1], rtol=0.0, atol=1e-12
    ):
        raise ValueError("overall Baseline A RMSE does not reproduce")
    baseline_b_mse = mean_squared_error(truth, baseline_b)
    baseline_b_rmse = root_mean_squared_error(truth, baseline_b)
    row_hashes = "\n".join(
        item["comparison"]["ordered_row_ids_sha256"] for item in fold_metrics
    )
    now = datetime.now(timezone.utc)
    report = {
        "run_id": "baseline_b_cv_" + now.strftime("%Y%m%dT%H%M%SZ"),
        "timestamp": now.isoformat(),
        "status": "ok",
        "full_cv_run": True,
        "git_commit": git_commit,
        "data_hash": config["data_hash"],
        "method": BASELINE_B_METHOD,
        "target_definition": BASELINE_B_TARGET,
        "used_source_fields": list(INFERENCE_COLUMNS)
        + (["typewell.TVT"] if config.get("use_typewell_tvt_prior", False) else [])
        + (
            ["typewell.TVT", "typewell.GR"]
            if config.get("use_typewell_gr_slope", False)
            else []
        ),
        "feature_columns": list(
            LAST_KNOWN_SLOPE_FEATURE_COLUMNS
            if config.get("use_last_known_slope", False)
            else (
                TYPEWELL_PRIOR_FEATURE_COLUMNS
                if config.get("use_typewell_tvt_prior", False)
                else (
                    TYPEWELL_GR_SLOPE_FEATURE_COLUMNS
                    if config.get("use_typewell_gr_slope", False)
                    else BASELINE_B_FEATURE_COLUMNS
                )
            )
        ),
        "excluded_fields": list(EXCLUDED_FIELDS),
        "parameters": config["parameters"],
        "num_boost_round": config["num_boost_round"],
        "last_known_slope_window": config.get("last_known_slope_window", 2),
        "lightgbm_version": lgb.__version__,
        "cv_scheme": config["cv_scheme"],
        "folds": int(config["folds"]),
        "seed": int(config["seed"]),
        "fold_mapping_sha256": config["fold_mapping_sha256"],
        "quarantine_policy_version": QUARANTINE_POLICY_VERSION,
        "resource_limits": config["resource_limits"],
        "resource_estimate": estimate,
        "fold_metrics": fold_metrics,
        "oof_prediction_rows": len(truth),
        "oof_wells": int(config["effective_wells"]),
        "oof_groups": int(config["typewell_groups"]),
        "oof_ordered_fold_hashes_sha256": hashlib.sha256(
            (row_hashes + "\n").encode("utf-8")
        ).hexdigest(),
        "same_oof_rows": True,
        "baseline_a": {
            "mse": baseline_a_mse,
            "rmse": baseline_a_rmse,
            "nan_count": int(np.isnan(baseline_a).sum()),
            "inf_count": int(np.isinf(baseline_a).sum()),
        },
        "baseline_b": {
            "mse": baseline_b_mse,
            "rmse": baseline_b_rmse,
            "mse_delta_vs_baseline_a": baseline_b_mse - baseline_a_mse,
            "rmse_delta_vs_baseline_a": baseline_b_rmse - baseline_a_rmse,
            "mse_relative_change_percent": 100.0
            * (baseline_b_mse - baseline_a_mse)
            / baseline_a_mse,
            "nan_count": int(np.isnan(baseline_b).sum()),
            "inf_count": int(np.isinf(baseline_b).sum()),
        },
        "cv_mse": baseline_b_mse,
        "cv_rmse": baseline_b_rmse,
        "train_seconds": float(sum(item["train_seconds"] for item in fold_metrics)),
        "inference_seconds": float(
            sum(item["inference_seconds"] for item in fold_metrics)
        ),
        "cpu_seconds": float(sum(item["cpu_seconds"] for item in fold_metrics)),
        "wall_seconds": time.perf_counter() - parent_started,
        "peak_rss_mib": float(max(item["peak_rss_mib"] for item in fold_metrics)),
        "gpu": "none",
        "fitted_preprocessing": "none",
        "lightgbm_dataset_scope": "training_fold_only",
    }
    output = _path(config["cv_report_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--fold-worker", type=int)
    parser.add_argument("--worker-output-prefix", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    if args.estimate_only:
        mapping = _load_mapping(config)
        estimate = estimate_resources(
            mapping,
            json.loads(
                _path(config["resource_estimate_basis"]).read_text(encoding="utf-8")
            ),
        )
        limits = _limits(config)
        estimate["within_peak_rss_limit"] = (
            estimate["estimated_peak_rss_mib"] <= limits.max_peak_rss_mib
        )
        estimate["within_per_fold_time_limit"] = (
            estimate["estimated_per_fold_wall_seconds"]
            <= limits.max_stage_seconds
        )
        print(json.dumps(estimate, indent=2, sort_keys=True))
        if not (
            estimate["within_peak_rss_limit"]
            and estimate["within_per_fold_time_limit"]
        ):
            raise RuntimeError("formal Baseline B estimate exceeds a safety limit")
        return 0
    if args.fold_worker is not None:
        if args.worker_output_prefix is None:
            raise ValueError("fold worker requires --worker-output-prefix")
        _run_fold(config, args.fold_worker, args.worker_output_prefix)
        return 0
    if args.worker_output_prefix is not None:
        raise ValueError("--worker-output-prefix is internal to fold workers")
    _run_parent(config, config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
