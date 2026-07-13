#!/usr/bin/env python3
"""Build the deterministic, quarantined ROGII well-level fold mapping."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii.cv import DEFAULT_SEED, assign_group_folds
from rogii.quarantine import (
    PUBLIC_SAMPLE_OVERLAP_WELLS,
    QUARANTINE_POLICY_VERSION,
    assert_no_public_sample_overlap,
    partition_public_sample_overlap,
)
from rogii.typewell import TYPEWELL_FINGERPRINT_ALGORITHM, typewell_numeric_fingerprint


HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"
MISSING_TOKENS = {"", "nan", "na", "null", "none"}


@dataclass
class Moments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def merge(self, other: "Moments") -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.count, self.mean, self.m2 = other.count, other.mean, other.m2
            self.minimum, self.maximum = other.minimum, other.maximum
            return
        combined = self.count + other.count
        delta = other.mean - self.mean
        self.m2 += other.m2 + delta * delta * self.count * other.count / combined
        self.mean += delta * other.count / combined
        self.count = combined
        self.minimum = min(self.minimum, other.minimum)
        self.maximum = max(self.maximum, other.maximum)

    def as_summary(self) -> dict[str, int | float | None]:
        return {
            "count": self.count,
            "mean": self.mean if self.count else None,
            "std": math.sqrt(self.m2 / self.count) if self.count else None,
            "min": self.minimum if self.count else None,
            "max": self.maximum if self.count else None,
        }


def _well_id(path: Path, suffix: str) -> str:
    if not path.name.endswith(suffix):
        raise ValueError(f"unexpected filename: {path}")
    identifier = path.name[: -len(suffix)].lower()
    if not identifier:
        raise ValueError(f"missing well ID in filename: {path}")
    return identifier


def scan_prediction_zone(path: Path) -> Moments:
    moments = Moments()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = [column for column in ("TVT", "TVT_input") if column not in fields]
        if missing:
            raise ValueError(f"missing required horizontal columns {missing} in {path}")
        for row_number, row in enumerate(reader, start=2):
            if row["TVT_input"].strip().lower() not in MISSING_TOKENS:
                continue
            try:
                target = float(row["TVT"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"non-numeric TVT at row {row_number} in {path}") from exc
            if not math.isfinite(target):
                raise ValueError(f"non-finite TVT at row {row_number} in {path}")
            moments.add(target)
    if moments.count == 0:
        raise ValueError(f"no prediction-zone rows in {path}")
    return moments


def collect_well_metadata(train_dir: Path) -> pd.DataFrame:
    horizontal = {_well_id(path, HORIZONTAL_SUFFIX): path for path in train_dir.glob(f"*{HORIZONTAL_SUFFIX}")}
    typewells = {_well_id(path, TYPEWELL_SUFFIX): path for path in train_dir.glob(f"*{TYPEWELL_SUFFIX}")}
    if set(horizontal) != set(typewells):
        missing_horizontal = sorted(set(typewells) - set(horizontal))
        missing_typewell = sorted(set(horizontal) - set(typewells))
        raise ValueError(
            f"unpaired wells: missing_horizontal={missing_horizontal}, missing_typewell={missing_typewell}"
        )
    records = []
    for identifier in sorted(horizontal):
        moments = scan_prediction_zone(horizontal[identifier])
        records.append(
            {
                "well_id": identifier,
                "typewell_group": typewell_numeric_fingerprint(typewells[identifier]),
                "prediction_rows": moments.count,
                "target_mean": moments.mean,
                "target_m2": moments.m2,
                "target_min": moments.minimum,
                "target_max": moments.maximum,
            }
        )
    return pd.DataFrame.from_records(records)


def _moments_from_row(row: object) -> Moments:
    return Moments(
        count=int(row.prediction_rows),
        mean=float(row.target_mean),
        m2=float(row.target_m2),
        minimum=float(row.target_min),
        maximum=float(row.target_max),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_identifier(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_folds(
    train_dir: Path,
    output_path: Path,
    summary_path: Path,
    manifest_path: Path,
    *,
    n_splits: int,
    seed: int,
) -> dict:
    metadata = collect_well_metadata(train_dir)
    development, quarantined = partition_public_sample_overlap(metadata, well_col="well_id")
    assert_no_public_sample_overlap(development["well_id"], context="fold generation")
    quarantine_ids = set(quarantined["well_id"].str.lower())
    if quarantine_ids != set(PUBLIC_SAMPLE_OVERLAP_WELLS):
        raise ValueError(
            f"quarantine mismatch: expected={sorted(PUBLIC_SAMPLE_OVERLAP_WELLS)}, actual={sorted(quarantine_ids)}"
        )

    development = development.sort_values("well_id").reset_index(drop=True)
    development["fold"] = assign_group_folds(
        development,
        "typewell_group",
        n_splits=n_splits,
        seed=seed,
        sample_weight=development["prediction_rows"].to_numpy(),
    ).to_numpy()
    mapping = development[["well_id", "typewell_group", "fold", "prediction_rows"]].copy()
    mapping = mapping.sort_values("well_id").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(output_path, index=False, lineterminator="\n")
    mapping_sha256 = _sha256(output_path)

    fold_summaries = []
    for fold in range(n_splits):
        rows = development.loc[development["fold"] == fold].sort_values("well_id")
        moments = Moments()
        for row in rows.itertuples(index=False):
            moments.merge(_moments_from_row(row))
        fold_summaries.append(
            {
                "fold": fold,
                "wells": int(len(rows)),
                "typewell_groups": int(rows["typewell_group"].nunique()),
                "prediction_rows": moments.count,
                "tvt": moments.as_summary(),
            }
        )

    group_cross_fold = bool(development.groupby("typewell_group")["fold"].nunique().gt(1).any())
    quarantine_in_fold = bool(set(mapping["well_id"]) & set(PUBLIC_SAMPLE_OVERLAP_WELLS))
    summary = {
        "original_train_wells": int(len(metadata)),
        "quarantined_wells": int(len(quarantined)),
        "effective_wells": int(len(development)),
        "typewell_groups": int(development["typewell_group"].nunique()),
        "fold_count": n_splits,
        "seed": seed,
        "fingerprint_algorithm": TYPEWELL_FINGERPRINT_ALGORITHM,
        "quarantine_policy_version": QUARANTINE_POLICY_VERSION,
        "folds": fold_summaries,
        "total_prediction_rows": int(development["prediction_rows"].sum()),
        "group_cross_fold": group_cross_fold,
        "quarantine_in_fold": quarantine_in_fold,
        "fold_mapping_sha256": mapping_sha256,
        "manifest_path": _manifest_identifier(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=ROOT / "data/raw/train")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/rogii_well_folds.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "reports/rogii_fold_summary.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    summary = build_folds(
        args.train_dir.resolve(),
        args.output.resolve(),
        args.summary.resolve(),
        args.manifest.resolve(),
        n_splits=args.folds,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
