#!/usr/bin/env python3
"""Validate the generated ROGII fold mapping against immutable raw data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_folds import HORIZONTAL_SUFFIX, Moments, _sha256, scan_prediction_zone
from rogii.quarantine import (
    PUBLIC_SAMPLE_OVERLAP_WELLS,
    assert_no_public_sample_overlap,
    partition_public_sample_overlap,
)
from rogii.typewell import typewell_numeric_fingerprint
from rogii.typewell import TYPEWELL_FINGERPRINT_ALGORITHM
from rogii.quarantine import QUARANTINE_POLICY_VERSION


EXPECTED_TRAIN_WELLS = 773
EXPECTED_QUARANTINED_WELLS = 3
EXPECTED_EFFECTIVE_WELLS = 770
EXPECTED_FOLDS = 5


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f"{name} mismatch: actual={actual}, expected={expected}")


def validate_foundation(train_dir: Path, mapping_path: Path, summary_path: Path, manifest_path: Path) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mapping = pd.read_csv(mapping_path, dtype={"well_id": "string", "typewell_group": "string"})
    required = {"well_id", "typewell_group", "fold", "prediction_rows"}
    _require(required.issubset(mapping.columns), f"mapping missing columns: {sorted(required - set(mapping.columns))}")
    _require(not mapping[list(required)].isna().any().any(), "mapping contains missing values")
    _require(not mapping["well_id"].duplicated().any(), "mapping contains duplicate wells")
    numeric_folds = pd.to_numeric(mapping["fold"], errors="raise")
    _require((numeric_folds == numeric_folds.astype(int)).all(), "fold labels must be integers")
    mapping["fold"] = numeric_folds.astype(int)
    _require(set(mapping["fold"]) == set(range(EXPECTED_FOLDS)), "fold labels must be 0..4")
    _require(len(mapping) == EXPECTED_EFFECTIVE_WELLS, f"effective well count is {len(mapping)}, expected 770")
    _require(
        not mapping.groupby("typewell_group")["fold"].nunique().gt(1).any(),
        "a typewell group crosses folds",
    )
    assert_no_public_sample_overlap(mapping["well_id"], context="persisted fold mapping")

    raw_wells = sorted(path.name[: -len(HORIZONTAL_SUFFIX)].lower() for path in train_dir.glob(f"*{HORIZONTAL_SUFFIX}"))
    _require(len(raw_wells) == EXPECTED_TRAIN_WELLS, f"raw train wells={len(raw_wells)}, expected 773")
    raw_frame = pd.DataFrame({"well_id": raw_wells})
    expected_development, quarantined = partition_public_sample_overlap(raw_frame)
    _require(len(quarantined) == EXPECTED_QUARANTINED_WELLS, f"quarantined wells={len(quarantined)}, expected 3")
    _require(
        set(quarantined["well_id"]) == set(PUBLIC_SAMPLE_OVERLAP_WELLS),
        f"unexpected quarantine set: {sorted(quarantined['well_id'])}",
    )
    _require(
        set(mapping["well_id"]) == set(expected_development["well_id"]),
        "mapping does not exactly cover the 770 non-quarantined wells",
    )

    fold_moments = {fold: Moments() for fold in range(EXPECTED_FOLDS)}
    total_prediction_rows = 0
    for row in mapping.sort_values("well_id").itertuples(index=False):
        horizontal_path = train_dir / f"{row.well_id}{HORIZONTAL_SUFFIX}"
        typewell_path = train_dir / f"{row.well_id}__typewell.csv"
        moments = scan_prediction_zone(horizontal_path)
        _require(
            moments.count == int(row.prediction_rows),
            f"prediction row mismatch for {row.well_id}: {moments.count} != {row.prediction_rows}",
        )
        fingerprint = typewell_numeric_fingerprint(typewell_path)
        _require(fingerprint == row.typewell_group, f"typewell fingerprint mismatch for {row.well_id}")
        fold_moments[int(row.fold)].merge(moments)
        total_prediction_rows += moments.count

    _require(summary["original_train_wells"] == EXPECTED_TRAIN_WELLS, "summary raw well count mismatch")
    _require(summary["quarantined_wells"] == EXPECTED_QUARANTINED_WELLS, "summary quarantine count mismatch")
    _require(summary["effective_wells"] == EXPECTED_EFFECTIVE_WELLS, "summary effective count mismatch")
    _require(summary["fold_count"] == EXPECTED_FOLDS, "summary fold count mismatch")
    _require(summary["fingerprint_algorithm"] == TYPEWELL_FINGERPRINT_ALGORITHM, "fingerprint algorithm mismatch")
    _require(summary["quarantine_policy_version"] == QUARANTINE_POLICY_VERSION, "quarantine policy mismatch")
    _require(summary["typewell_groups"] == int(mapping["typewell_group"].nunique()), "summary group count mismatch")
    _require(summary["total_prediction_rows"] == total_prediction_rows, "summary prediction row total mismatch")
    _require(summary["group_cross_fold"] is False, "summary reports group leakage")
    _require(summary["quarantine_in_fold"] is False, "summary reports quarantined wells in folds")
    _require(summary["fold_mapping_sha256"] == _sha256(mapping_path), "fold mapping SHA256 mismatch")
    _require(summary["manifest_sha256"] == _sha256(manifest_path), "manifest SHA256 mismatch")

    summary_folds = {int(item["fold"]): item for item in summary["folds"]}
    _require(set(summary_folds) == set(range(EXPECTED_FOLDS)), "summary fold entries are incomplete")
    for fold, moments in fold_moments.items():
        expected = summary_folds[fold]
        fold_rows = mapping.loc[mapping["fold"] == fold]
        _require(expected["wells"] == len(fold_rows), f"fold {fold} well count mismatch")
        _require(
            expected["typewell_groups"] == int(fold_rows["typewell_group"].nunique()),
            f"fold {fold} group count mismatch",
        )
        _require(expected["prediction_rows"] == moments.count, f"fold {fold} prediction rows mismatch")
        actual = moments.as_summary()
        for key in ("mean", "std", "min", "max"):
            _close(float(actual[key]), float(expected["tvt"][key]), f"fold {fold} TVT {key}")

    return {
        "status": "ok",
        "raw_train_wells": len(raw_wells),
        "quarantined_wells": len(quarantined),
        "effective_wells": len(mapping),
        "typewell_groups": int(mapping["typewell_group"].nunique()),
        "folds": EXPECTED_FOLDS,
        "prediction_rows": total_prediction_rows,
        "fold_mapping_sha256": summary["fold_mapping_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=ROOT / "data/raw/train")
    parser.add_argument("--mapping", type=Path, default=ROOT / "data/processed/rogii_well_folds.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "reports/rogii_fold_summary.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.json")
    args = parser.parse_args()
    result = validate_foundation(
        args.train_dir.resolve(),
        args.mapping.resolve(),
        args.summary.resolve(),
        args.manifest.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
