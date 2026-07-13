#!/usr/bin/env python3
"""Run deterministic ROGII Baseline A smoke tests or full five-fold CV."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii.baseline import (
    BASELINE_A_METHOD,
    BASELINE_A_USED_FIELDS,
    BaselineAAlgorithm,
    load_baseline_a_algorithm,
    predict_baseline_a,
    save_baseline_a_algorithm,
)
from rogii.io import (
    INFERENCE_COLUMNS,
    load_fold_mapping,
    prediction_mask,
    read_horizontal_well,
    sha256_file,
)
from rogii.metric import mean_squared_error, root_mean_squared_error
from rogii.quarantine import QUARANTINE_POLICY_VERSION, assert_no_public_sample_overlap

EXCLUDED_FIELDS = (
    "TVT",
    "ANCC",
    "ASTNU",
    "ASTNL",
    "EGFDU",
    "EGFDL",
    "BUDA",
    "Geology",
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


def evaluate_cv(
    mapping: pd.DataFrame,
    train_dir: Path,
    *,
    n_splits: int,
    smoke_wells_per_fold: int = 0,
) -> dict:
    """Score each validation row; this train-free baseline fits no fold statistics."""
    if smoke_wells_per_fold < 0:
        raise ValueError("smoke_wells_per_fold must be non-negative")
    assert_no_public_sample_overlap(mapping["well_id"], context="Baseline A CV mapping")

    all_truth: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    fold_metrics: list[dict] = []
    for fold in range(n_splits):
        train_ids = mapping.loc[mapping["fold"] != fold, "well_id"]
        validation = mapping.loc[mapping["fold"] == fold].sort_values("well_id")
        assert_no_public_sample_overlap(
            train_ids, context=f"Baseline A fold {fold} training"
        )
        assert_no_public_sample_overlap(
            validation["well_id"], context=f"Baseline A fold {fold} validation"
        )
        if smoke_wells_per_fold:
            validation = validation.head(smoke_wells_per_fold)
        if validation.empty:
            raise ValueError(f"fold {fold} has no validation wells")

        fold_truth: list[np.ndarray] = []
        fold_predictions: list[np.ndarray] = []
        started = time.perf_counter()
        for row in validation.itertuples(index=False):
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
            predictions = predict_baseline_a(inference_frame)
            truth = frame.loc[mask, "TVT"].to_numpy(dtype=np.float64)
            fold_truth.append(truth)
            fold_predictions.append(predictions)
        elapsed = time.perf_counter() - started

        truth_values = np.concatenate(fold_truth)
        prediction_values = np.concatenate(fold_predictions)
        nan_count = int(np.isnan(prediction_values).sum())
        inf_count = int(np.isinf(prediction_values).sum())
        fold_result = {
            "fold": fold,
            "wells": len(validation),
            "prediction_rows": len(prediction_values),
            "mse": mean_squared_error(truth_values, prediction_values),
            "rmse": root_mean_squared_error(truth_values, prediction_values),
            "inference_seconds": elapsed,
            "nan_count": nan_count,
            "inf_count": inf_count,
        }
        fold_metrics.append(fold_result)
        all_truth.append(truth_values)
        all_predictions.append(prediction_values)

    oof_truth = np.concatenate(all_truth)
    oof_predictions = np.concatenate(all_predictions)
    return {
        "fold_metrics": fold_metrics,
        "oof_prediction_rows": len(oof_predictions),
        "cv_mse": mean_squared_error(oof_truth, oof_predictions),
        "cv_rmse": root_mean_squared_error(oof_truth, oof_predictions),
        "train_seconds": 0.0,
        "inference_seconds": float(
            sum(item["inference_seconds"] for item in fold_metrics)
        ),
        "nan_count": int(np.isnan(oof_predictions).sum()),
        "inf_count": int(np.isinf(oof_predictions).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-wells-per-fold", type=int, default=0)
    args = parser.parse_args()

    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if config["method"] != BASELINE_A_METHOD:
        raise ValueError(f"unsupported Baseline A method: {config['method']}")
    if config["quarantine_policy_version"] != QUARANTINE_POLICY_VERSION:
        raise ValueError("quarantine policy version mismatch")
    n_splits = int(config["folds"])
    mapping = load_fold_mapping(
        _path(config["fold_mapping"]),
        expected_sha256=config["fold_mapping_sha256"],
        n_splits=n_splits,
        expected_wells=int(config["effective_wells"]),
        expected_groups=int(config["typewell_groups"]),
    )
    result = evaluate_cv(
        mapping,
        _path(config["train_dir"]),
        n_splits=n_splits,
        smoke_wells_per_fold=args.smoke_wells_per_fold,
    )
    artifact = BaselineAAlgorithm.create(
        fold_mapping_sha256=config["fold_mapping_sha256"]
    )
    artifact_path = save_baseline_a_algorithm(artifact, _path(config["artifact_path"]))
    reloaded_artifact = load_baseline_a_algorithm(
        artifact_path,
        expected_fold_mapping_sha256=config["fold_mapping_sha256"],
    )
    roundtrip_smoke = pd.DataFrame({"TVT_input": [100.0, 101.5, np.nan, np.nan]})
    roundtrip_equal = np.array_equal(
        artifact.predict(roundtrip_smoke),
        reloaded_artifact.predict(roundtrip_smoke),
    )
    if not roundtrip_equal:
        raise ValueError("Baseline A artifact reload changed smoke predictions")
    is_smoke = args.smoke_wells_per_fold > 0
    report = {
        "run_id": "baseline_a_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "status": "smoke_ok" if is_smoke else "ok",
        "git_commit": _git_commit(),
        "method": BASELINE_A_METHOD,
        "parameters": {"strategy": "extend_final_known_tvt_input"},
        "used_fields": list(BASELINE_A_USED_FIELDS),
        "excluded_fields": list(EXCLUDED_FIELDS),
        "cv_scheme": config["cv_scheme"],
        "fold_count": n_splits,
        "seed": int(config["seed"]),
        "fold_mapping_sha256": config["fold_mapping_sha256"],
        "quarantine_policy_version": QUARANTINE_POLICY_VERSION,
        "gpu": "none",
        "trained_model": False,
        "artifact_path": artifact_path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_reload_verified": True,
        "artifact_roundtrip_predictions_equal": roundtrip_equal,
        "smoke_wells_per_fold": args.smoke_wells_per_fold,
        **result,
    }
    if not is_smoke:
        output = _path(config["report_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
