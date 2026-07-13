"""Strict data-loading helpers for the ROGII baseline pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .quarantine import assert_no_public_sample_overlap

HORIZONTAL_SUFFIX = "__horizontal_well.csv"
INFERENCE_COLUMNS = ("MD", "X", "Y", "Z", "GR", "TVT_input")
TARGET_COLUMN = "TVT"
FOLD_COLUMNS = ("well_id", "typewell_group", "fold", "prediction_rows")


@dataclass(frozen=True)
class WellFile:
    well_id: str
    path: Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_horizontal_wells(directory: str | Path) -> list[WellFile]:
    """Discover wells dynamically and return them in stable filename order."""
    root = Path(directory)
    wells: list[WellFile] = []
    seen: set[str] = set()
    for path in sorted(
        root.glob(f"*{HORIZONTAL_SUFFIX}"), key=lambda item: item.name.lower()
    ):
        well_id = path.name[: -len(HORIZONTAL_SUFFIX)].lower()
        if not well_id:
            raise ValueError(f"missing well ID in filename: {path}")
        if well_id in seen:
            raise ValueError(f"duplicate well ID in {root}: {well_id}")
        seen.add(well_id)
        wells.append(WellFile(well_id=well_id, path=path))
    if not wells:
        raise ValueError(f"no horizontal well files found in {root}")
    return wells


def prediction_mask(frame: pd.DataFrame) -> pd.Series:
    """Return the TVT_input-null prediction suffix, failing on malformed wells."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if "TVT_input" not in frame.columns:
        raise KeyError("missing required column: TVT_input")
    if frame.empty:
        raise ValueError("horizontal well is empty")

    numeric = pd.to_numeric(frame["TVT_input"], errors="raise")
    mask = numeric.isna()
    if not mask.any():
        raise ValueError("well has no prediction-zone rows")
    first_prediction = int(np.flatnonzero(mask.to_numpy())[0])
    if first_prediction == 0:
        raise ValueError("well has no known TVT_input prefix")
    if not bool(mask.iloc[first_prediction:].all()):
        raise ValueError("TVT_input prediction zone must be a contiguous suffix")
    known = numeric.iloc[:first_prediction].to_numpy(dtype=np.float64)
    if not np.isfinite(known).all():
        raise ValueError("known TVT_input contains non-finite values")
    return mask


def prediction_row_positions(frame: pd.DataFrame) -> np.ndarray:
    """Return zero-based source-row positions for the prediction suffix."""
    return np.flatnonzero(prediction_mask(frame).to_numpy()).astype(
        np.int64, copy=False
    )


def prediction_ids(well_id: str, frame: pd.DataFrame) -> pd.Series:
    identifier = str(well_id).lower()
    if not identifier or "_" in identifier:
        raise ValueError(f"invalid well ID: {well_id!r}")
    positions = prediction_row_positions(frame)
    return pd.Series(
        [f"{identifier}_{position}" for position in positions],
        dtype="string",
        name="id",
    )


def read_horizontal_well(path: str | Path, *, include_target: bool) -> pd.DataFrame:
    """Read only official inference inputs, plus TVT when CV scoring needs it."""
    source = Path(path)
    columns = [*INFERENCE_COLUMNS]
    if include_target:
        columns.append(TARGET_COLUMN)
    frame = pd.read_csv(source, usecols=columns)
    if list(frame.index) != list(range(len(frame))):
        raise ValueError(f"unexpected row index in {source}")

    for column in INFERENCE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in ("MD", "X", "Y", "Z"):
        if not np.isfinite(frame[column].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"non-finite {column} in {source}")
    gr = frame["GR"].dropna().to_numpy(dtype=np.float64)
    if not np.isfinite(gr).all():
        raise ValueError(f"non-finite GR in {source}")

    mask = prediction_mask(frame)
    if include_target:
        frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="raise")
        target = frame.loc[mask, TARGET_COLUMN].to_numpy(dtype=np.float64)
        if not np.isfinite(target).all():
            raise ValueError(f"non-finite prediction-zone TVT in {source}")
    return frame


def load_fold_mapping(
    path: str | Path,
    *,
    expected_sha256: str,
    n_splits: int,
    expected_wells: int | None = None,
    expected_groups: int | None = None,
) -> pd.DataFrame:
    """Load and independently validate the persisted honest-CV mapping."""
    source = Path(path)
    actual_sha256 = sha256_file(source)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"fold mapping SHA256 mismatch: actual={actual_sha256}, expected={expected_sha256}"
        )
    mapping = pd.read_csv(
        source, dtype={"well_id": "string", "typewell_group": "string"}
    )
    if tuple(mapping.columns) != FOLD_COLUMNS:
        raise ValueError(f"fold mapping columns must be {list(FOLD_COLUMNS)}")
    if mapping.isna().any().any():
        raise ValueError("fold mapping contains missing values")
    mapping["well_id"] = mapping["well_id"].str.lower()
    if mapping["well_id"].duplicated().any():
        raise ValueError("fold mapping contains duplicate wells")
    assert_no_public_sample_overlap(
        mapping["well_id"], context="Baseline A fold mapping"
    )

    for column in ("fold", "prediction_rows"):
        numeric = pd.to_numeric(mapping[column], errors="raise")
        if not (numeric == numeric.astype(np.int64)).all():
            raise ValueError(f"fold mapping {column} must contain integers")
        mapping[column] = numeric.astype(np.int64)
    if sorted(mapping["fold"].unique().tolist()) != list(range(n_splits)):
        raise ValueError(f"fold labels must cover 0..{n_splits - 1}")
    if (mapping["prediction_rows"] <= 0).any():
        raise ValueError("prediction_rows must be positive")
    if mapping.groupby("typewell_group")["fold"].nunique().gt(1).any():
        raise ValueError("a typewell group crosses folds")
    if expected_wells is not None and len(mapping) != expected_wells:
        raise ValueError(
            f"fold mapping wells={len(mapping)}, expected={expected_wells}"
        )
    groups = int(mapping["typewell_group"].nunique())
    if expected_groups is not None and groups != expected_groups:
        raise ValueError(f"fold mapping groups={groups}, expected={expected_groups}")
    return mapping
