#!/usr/bin/env python3
"""Read-only, streaming audit for the official ROGII competition data."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import zip_longest
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii.typewell import typewell_numeric_fingerprint


HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"
MISSING_TOKENS = {"", "nan", "na", "null", "none"}
SAMPLE_ID_RE = re.compile(r"^(?P<well>[0-9a-f]{8})_(?P<row>\d+)$")


def well_id(path: Path) -> str:
    return path.name.split("__", 1)[0]


@dataclass
class RunningStats:
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

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "mean": self.mean if self.count else None,
            "std": math.sqrt(self.m2 / self.count) if self.count else None,
            "min": self.minimum if self.count else None,
            "max": self.maximum if self.count else None,
        }


class Reservoir:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.values: list[float] = []
        self.seen = 0
        self.rng = random.Random(seed)

    def add(self, value: float) -> None:
        self.seen += 1
        if len(self.values) < self.size:
            self.values.append(value)
            return
        index = self.rng.randrange(self.seen)
        if index < self.size:
            self.values[index] = value

    def quantiles(self) -> dict[str, float | None]:
        if not self.values:
            return {"p01": None, "p25": None, "p50": None, "p75": None, "p99": None}
        values = sorted(self.values)
        result = {}
        for label, q in (("p01", 0.01), ("p25", 0.25), ("p50", 0.5), ("p75", 0.75), ("p99", 0.99)):
            result[label] = values[round(q * (len(values) - 1))]
        return result


def classify(value: str) -> str:
    if value.strip().lower() in MISSING_TOKENS:
        return "missing"
    try:
        number = float(value)
    except ValueError:
        return "string"
    return "finite_numeric" if math.isfinite(number) else "infinite"


def profile_csv_files(paths: Iterable[Path], target_column: str | None = None) -> dict:
    schemas: Counter[tuple[str, ...]] = Counter()
    column_kinds: dict[str, Counter[str]] = defaultdict(Counter)
    numeric_stats: dict[str, RunningStats] = defaultdict(RunningStats)
    string_values: dict[str, set[str]] = defaultdict(set)
    missing: Counter[str] = Counter()
    infinite: Counter[str] = Counter()
    row_counts: list[int] = []
    duplicate_rows = 0
    target_stats = RunningStats()
    target_sample = Reservoir(100_000, seed=20260713)
    for path in paths:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"missing header: {path}")
            schemas[tuple(reader.fieldnames)] += 1
            seen: set[tuple[str, ...]] = set()
            rows = 0
            for row in reader:
                rows += 1
                packed = tuple(row[name] for name in reader.fieldnames)
                if packed in seen:
                    duplicate_rows += 1
                else:
                    seen.add(packed)
                for name, value in row.items():
                    kind = classify(value)
                    column_kinds[name][kind] += 1
                    if kind == "missing":
                        missing[name] += 1
                    elif kind == "infinite":
                        infinite[name] += 1
                    elif kind == "finite_numeric":
                        numeric_stats[name].add(float(value))
                    elif len(string_values[name]) < 1000:
                        string_values[name].add(value)
                if target_column and classify(row.get(target_column, "")) == "finite_numeric":
                    value = float(row[target_column])
                    target_stats.add(value)
                    target_sample.add(value)
            row_counts.append(rows)
    return {
        "files": len(row_counts),
        "rows": sum(row_counts),
        "rows_per_file": {
            "min": min(row_counts) if row_counts else 0,
            "max": max(row_counts) if row_counts else 0,
            "mean": (sum(row_counts) / len(row_counts)) if row_counts else 0,
        },
        "schemas": [{"files": count, "columns": list(schema)} for schema, count in schemas.items()],
        "column_observed_kinds": {name: sorted(kinds) for name, kinds in sorted(column_kinds.items())},
        "numeric_summary": {name: stats.as_dict() for name, stats in sorted(numeric_stats.items())},
        "string_summary": {
            name: {"unique_values_capped_at_1000": len(values), "examples": sorted(values)[:10]}
            for name, values in sorted(string_values.items())
        },
        "missing": dict(sorted(missing.items())),
        "infinite": dict(sorted(infinite.items())),
        "exact_duplicate_rows_within_files": duplicate_rows,
        "target": {**target_stats.as_dict(), **target_sample.quantiles(), "quantiles_from_reservoir": len(target_sample.values)}
        if target_column
        else None,
    }


def horizontal_input_audit(paths: Iterable[Path]) -> tuple[dict, dict[str, int]]:
    tvt_input_missing = 0
    tvt_input_present = 0
    tvt_input_target_mismatch = 0
    prediction_rows = 0
    per_well_rows: dict[str, int] = {}
    for path in paths:
        rows = 0
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            has_target = "TVT" in (reader.fieldnames or [])
            for row in reader:
                rows += 1
                input_kind = classify(row.get("TVT_input", ""))
                if input_kind == "missing":
                    tvt_input_missing += 1
                    prediction_rows += 1
                else:
                    tvt_input_present += 1
                    if has_target and float(row["TVT_input"]) != float(row["TVT"]):
                        tvt_input_target_mismatch += 1
        per_well_rows[well_id(path)] = rows
    return {
        "tvt_input_present": tvt_input_present,
        "tvt_input_missing": tvt_input_missing,
        "prediction_rows": prediction_rows,
        "observed_tvt_input_not_equal_tvt": tvt_input_target_mismatch,
    }, per_well_rows


def compare_train_test(train_dir: Path, test_dir: Path) -> dict:
    train_ids = {well_id(path) for path in train_dir.glob(f"*{HORIZONTAL_SUFFIX}")}
    test_ids = {well_id(path) for path in test_dir.glob(f"*{HORIZONTAL_SUFFIX}")}
    shared = sorted(train_ids & test_ids)
    comparisons = {}
    feature_columns = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
    for identifier in shared:
        train_path = train_dir / f"{identifier}{HORIZONTAL_SUFFIX}"
        test_path = test_dir / f"{identifier}{HORIZONTAL_SUFFIX}"
        exact = total = train_only = test_only = 0
        with train_path.open(newline="", encoding="utf-8-sig") as train_handle, test_path.open(
            newline="", encoding="utf-8-sig"
        ) as test_handle:
            train_reader = csv.DictReader(train_handle)
            test_reader = csv.DictReader(test_handle)
            for train_row, test_row in zip_longest(train_reader, test_reader):
                if train_row is None:
                    test_only += 1
                elif test_row is None:
                    train_only += 1
                else:
                    total += 1
                    if all(train_row[name] == test_row[name] for name in feature_columns):
                        exact += 1
        comparisons[identifier] = {
            "aligned_rows": total,
            "exact_feature_rows": exact,
            "train_only_rows": train_only,
            "test_only_rows": test_only,
        }
    return {
        "train_wells": len(train_ids),
        "test_wells": len(test_ids),
        "shared_well_ids": shared,
        "shared_feature_comparison": comparisons,
    }


def audit_sample_submission(path: Path, test_row_counts: dict[str, int], test_dir: Path) -> dict:
    ids: list[str] = []
    prediction_columns: list[str] = []
    dtype_kinds: dict[str, set[str]] = defaultdict(set)
    parsed_by_well: Counter[str] = Counter()
    invalid_ids = 0
    out_of_bounds = 0
    not_prediction_rows = 0
    prediction_indices: dict[str, set[int]] = defaultdict(set)
    for identifier in test_row_counts:
        path_for_well = test_dir / f"{identifier}{HORIZONTAL_SUFFIX}"
        with path_for_well.open(newline="", encoding="utf-8-sig") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                if classify(row["TVT_input"]) == "missing":
                    prediction_indices[identifier].add(index)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        prediction_columns = columns[1:]
        for row in reader:
            identifier = row[columns[0]]
            ids.append(identifier)
            for name in prediction_columns:
                dtype_kinds[name].add(classify(row[name]))
            match = SAMPLE_ID_RE.fullmatch(identifier)
            if not match:
                invalid_ids += 1
                continue
            well = match.group("well")
            index = int(match.group("row"))
            parsed_by_well[well] += 1
            if well not in test_row_counts or index >= test_row_counts[well]:
                out_of_bounds += 1
            elif index not in prediction_indices[well]:
                not_prediction_rows += 1
    expected = {(well, index) for well, indices in prediction_indices.items() for index in indices}
    actual = set()
    for identifier in ids:
        match = SAMPLE_ID_RE.fullmatch(identifier)
        if match:
            actual.add((match.group("well"), int(match.group("row"))))
    return {
        "columns_in_order": columns,
        "rows": len(ids),
        "id_unique": len(ids) == len(set(ids)),
        "invalid_id_format": invalid_ids,
        "rows_by_well": dict(sorted(parsed_by_well.items())),
        "prediction_column_kinds": {name: sorted(kinds) for name, kinds in dtype_kinds.items()},
        "out_of_bounds_ids": out_of_bounds,
        "ids_not_at_missing_tvt_input_rows": not_prediction_rows,
        "missing_expected_prediction_ids": len(expected - actual),
        "unexpected_prediction_ids": len(actual - expected),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    root = args.raw_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"raw data directory not found: {root}")
    train_dir, test_dir = root / "train", root / "test"
    train_horizontal = sorted(train_dir.glob(f"*{HORIZONTAL_SUFFIX}"))
    train_typewell = sorted(train_dir.glob(f"*{TYPEWELL_SUFFIX}"))
    test_horizontal = sorted(test_dir.glob(f"*{HORIZONTAL_SUFFIX}"))
    test_typewell = sorted(test_dir.glob(f"*{TYPEWELL_SUFFIX}"))
    all_files = sorted(path for path in root.rglob("*") if path.is_file())
    categories = {}
    for name, paths in {
        "train_horizontal_csv": train_horizontal,
        "train_typewell_csv": train_typewell,
        "train_png": sorted(train_dir.glob("*.png")),
        "test_horizontal_csv": test_horizontal,
        "test_typewell_csv": test_typewell,
        "sample_submission": [root / "sample_submission.csv"],
        "official_presentation": [root / "AI_wellbore_geology_prediction_task_en.pptx"],
    }.items():
        categories[name] = {"files": len(paths), "bytes": sum(path.stat().st_size for path in paths)}

    train_input, train_rows = horizontal_input_audit(train_horizontal)
    test_input, test_rows = horizontal_input_audit(test_horizontal)
    typewell_groups: dict[str, list[str]] = defaultdict(list)
    for path in train_typewell:
        typewell_groups[typewell_numeric_fingerprint(path)].append(well_id(path))
    repeated_groups = [wells for wells in typewell_groups.values() if len(wells) > 1]

    result = {
        "raw_dir": str(root),
        "file_count": len(all_files),
        "total_bytes": sum(path.stat().st_size for path in all_files),
        "categories": categories,
        "train_horizontal": profile_csv_files(train_horizontal, target_column="TVT"),
        "train_typewell": profile_csv_files(train_typewell),
        "test_horizontal": profile_csv_files(test_horizontal),
        "test_typewell": profile_csv_files(test_typewell),
        "train_prediction_boundary": train_input,
        "test_prediction_boundary": test_input,
        "well_id_unique": len(train_rows) == len(train_horizontal) and len(test_rows) == len(test_horizontal),
        "typewell_profile_groups": {
            "unique_profiles": len(typewell_groups),
            "repeated_profile_groups": len(repeated_groups),
            "wells_in_repeated_groups": sum(len(group) for group in repeated_groups),
            "largest_group": max((len(group) for group in typewell_groups.values()), default=0),
        },
        "train_test_overlap": compare_train_test(train_dir, test_dir),
        "sample_submission": audit_sample_submission(root / "sample_submission.csv", test_rows, test_dir),
    }
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
