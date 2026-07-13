"""Quarantine policy for ROGII's overlapping public sample wells."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


PUBLIC_SAMPLE_OVERLAP_WELLS = frozenset(
    {
        "000d7d20",
        "00bbac68",
        "00e12e8b",
    }
)
QUARANTINE_POLICY_VERSION = "public-sample-overlap-v1"


def _well_ids(frame: pd.DataFrame, well_col: str) -> pd.Series:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if well_col not in frame.columns:
        raise KeyError(f"missing well column: {well_col}")
    if frame[well_col].isna().any():
        raise ValueError("well IDs contain missing values")
    return frame[well_col].astype("string").str.lower()


def public_sample_overlap_mask(
    frame: pd.DataFrame,
    well_col: str = "well_id",
) -> pd.Series:
    """Return a boolean mask for rows belonging to quarantined public samples."""
    return _well_ids(frame, well_col).isin(PUBLIC_SAMPLE_OVERLAP_WELLS)


def partition_public_sample_overlap(
    frame: pd.DataFrame,
    well_col: str = "well_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into honest development rows and quarantined sample rows."""
    mask = public_sample_overlap_mask(frame, well_col)
    return frame.loc[~mask].copy(), frame.loc[mask].copy()


def assert_no_public_sample_overlap(
    well_ids: Iterable[object],
    *,
    context: str = "training or validation",
) -> None:
    """Fail closed if quarantined wells enter an honest experiment."""
    normalized = set()
    for value in well_ids:
        if pd.isna(value):
            raise ValueError("well IDs contain missing values")
        normalized.add(str(value).lower())
    overlap = sorted(normalized & PUBLIC_SAMPLE_OVERLAP_WELLS)
    if overlap:
        joined = ", ".join(overlap)
        raise ValueError(f"quarantined public sample wells in {context}: {joined}")
