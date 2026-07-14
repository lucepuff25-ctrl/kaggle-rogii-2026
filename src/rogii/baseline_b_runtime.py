"""Runtime loading and resource guards for formal Baseline B runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from .baseline import predict_baseline_a
from .features import build_baseline_b_features, build_typewell_prior_features
from .io import (
    INFERENCE_COLUMNS,
    prediction_ids,
    prediction_mask,
    read_horizontal_well,
)
from .quarantine import assert_no_public_sample_overlap


@dataclass(frozen=True)
class ResourceLimits:
    max_peak_rss_mib: float
    max_stage_seconds: float
    min_available_ram_gib: float
    max_load_per_cpu: float

    def validate(self) -> None:
        for name, value in (
            ("max_peak_rss_mib", self.max_peak_rss_mib),
            ("max_stage_seconds", self.max_stage_seconds),
            ("min_available_ram_gib", self.min_available_ram_gib),
            ("max_load_per_cpu", self.max_load_per_cpu),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be numeric")
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class LoadedRows:
    features: pd.DataFrame
    truth: np.ndarray
    baseline_a_predictions: np.ndarray | None
    baseline_a_seconds: float
    ordered_row_ids_sha256: str


def peak_rss_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def available_ram_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            kib = int(line.split()[1])
            return float(kib / 1024.0 / 1024.0)
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def resource_snapshot() -> dict[str, float | int]:
    cpu_count = os.cpu_count() or 1
    return {
        "peak_rss_mib": peak_rss_mib(),
        "available_ram_gib": available_ram_gib(),
        "load_average_1m": float(os.getloadavg()[0]),
        "logical_cpus": int(cpu_count),
    }


def guard_resources(
    limits: ResourceLimits,
    *,
    context: str,
    started: float | None = None,
) -> dict[str, float | int]:
    limits.validate()
    snapshot = resource_snapshot()
    if snapshot["peak_rss_mib"] > limits.max_peak_rss_mib:
        raise RuntimeError(
            f"{context}: peak RSS {snapshot['peak_rss_mib']:.1f} MiB exceeds "
            f"{limits.max_peak_rss_mib:.1f} MiB"
        )
    if snapshot["available_ram_gib"] < limits.min_available_ram_gib:
        raise RuntimeError(
            f"{context}: available RAM {snapshot['available_ram_gib']:.1f} GiB "
            f"is below {limits.min_available_ram_gib:.1f} GiB"
        )
    load_per_cpu = snapshot["load_average_1m"] / snapshot["logical_cpus"]
    if load_per_cpu > limits.max_load_per_cpu:
        raise RuntimeError(
            f"{context}: one-minute load per CPU {load_per_cpu:.2f} exceeds "
            f"{limits.max_load_per_cpu:.2f}"
        )
    if started is not None:
        elapsed = time.perf_counter() - started
        if elapsed > limits.max_stage_seconds:
            raise RuntimeError(
                f"{context}: elapsed {elapsed:.1f}s exceeds "
                f"{limits.max_stage_seconds:.1f}s"
            )
    return snapshot


def load_mapping_rows(
    mapping: pd.DataFrame,
    train_dir: str | Path,
    *,
    include_baseline_a: bool,
    hash_row_ids: bool,
    limits: ResourceLimits,
    stage_started: float,
    context: str,
    use_typewell_tvt_prior: bool = False,
) -> LoadedRows:
    """Load complete wells after quarantine checks and build inference-only features."""
    assert_no_public_sample_overlap(mapping["well_id"], context=context)
    feature_parts: list[pd.DataFrame] = []
    truth_parts: list[np.ndarray] = []
    baseline_parts: list[np.ndarray] = []
    row_hasher = hashlib.sha256()
    baseline_seconds = 0.0
    root = Path(train_dir)

    for index, row in enumerate(mapping.itertuples(index=False), start=1):
        source = root / f"{row.well_id}__horizontal_well.csv"
        frame = read_horizontal_well(source, include_target=True)
        mask = prediction_mask(frame)
        actual_rows = int(mask.sum())
        if actual_rows != int(row.prediction_rows):
            raise ValueError(
                f"prediction rows for {row.well_id}: actual={actual_rows}, "
                f"mapping={row.prediction_rows}"
            )
        inference_frame = frame.loc[:, list(INFERENCE_COLUMNS)]
        if use_typewell_tvt_prior:
            typewell_source = root / f"{row.well_id}__typewell.csv"
            typewell = pd.read_csv(typewell_source, usecols=["TVT"])
            features = build_typewell_prior_features(inference_frame, typewell)
        else:
            features = build_baseline_b_features(inference_frame)
        truth = frame.loc[mask, "TVT"].to_numpy(dtype=np.float64)
        if len(features) != len(truth):
            raise ValueError(f"feature/target row mismatch for {row.well_id}")
        feature_parts.append(features)
        truth_parts.append(truth)
        if hash_row_ids:
            for row_id in prediction_ids(row.well_id, inference_frame):
                row_hasher.update(str(row_id).encode("utf-8"))
                row_hasher.update(b"\n")
        if include_baseline_a:
            started = time.perf_counter()
            baseline_parts.append(predict_baseline_a(inference_frame))
            baseline_seconds += time.perf_counter() - started
        if index % 25 == 0:
            guard_resources(
                limits,
                context=f"{context} after {index} wells",
                started=stage_started,
            )

    features = pd.concat(feature_parts, ignore_index=True)
    truth = np.concatenate(truth_parts)
    baseline = np.concatenate(baseline_parts) if include_baseline_a else None
    guard_resources(limits, context=f"{context} concatenated", started=stage_started)
    return LoadedRows(
        features=features,
        truth=truth,
        baseline_a_predictions=baseline,
        baseline_a_seconds=baseline_seconds,
        ordered_row_ids_sha256=(row_hasher.hexdigest() if hash_row_ids else ""),
    )


def mapping_summary(mapping: pd.DataFrame) -> dict[str, object]:
    return {
        "folds": sorted(int(value) for value in mapping["fold"].unique()),
        "groups": int(mapping["typewell_group"].nunique()),
        "wells": int(len(mapping)),
        "prediction_rows": int(mapping["prediction_rows"].sum()),
    }
