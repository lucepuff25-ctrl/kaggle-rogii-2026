"""Submission construction and strict sample-order validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import prediction_ids, sha256_file

SUBMISSION_COLUMNS = ("id", "tvt")


def well_prediction_frame(
    well_id: str,
    horizontal: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    ids = prediction_ids(well_id, horizontal)
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(ids):
        raise ValueError("predictions must match the well's prediction rows")
    if not np.isfinite(values).all():
        raise ValueError("predictions contain NaN or infinite values")
    return pd.DataFrame({"id": ids, "tvt": values})


def _validate_ids(frame: pd.DataFrame, *, context: str) -> pd.Series:
    if tuple(frame.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"{context} columns must be {list(SUBMISSION_COLUMNS)}")
    if frame["id"].isna().any():
        raise ValueError(f"{context} IDs contain missing values")
    ids = frame["id"].astype("string")
    if ids.duplicated().any():
        raise ValueError(f"{context} IDs are not unique")
    return ids


def align_predictions_to_sample(
    predictions: pd.DataFrame,
    sample: pd.DataFrame,
) -> pd.DataFrame:
    """Use the sample IDs as the sole authority for output order."""
    predicted_ids = _validate_ids(predictions, context="predictions")
    sample_ids = _validate_ids(sample, context="sample submission")
    values = pd.to_numeric(predictions["tvt"], errors="raise").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("predictions contain NaN or infinite values")

    predicted_set = set(predicted_ids)
    sample_set = set(sample_ids)
    missing = sorted(sample_set - predicted_set)
    extra = sorted(predicted_set - sample_set)
    if missing or extra:
        raise ValueError(
            f"prediction IDs do not match sample: missing={missing[:5]}, extra={extra[:5]}"
        )
    indexed = pd.Series(values, index=predicted_ids, name="tvt")
    ordered = indexed.loc[sample_ids.to_list()].to_numpy(dtype=np.float64)
    return pd.DataFrame({"id": sample_ids.to_numpy(), "tvt": ordered})


def validate_submission_frame(
    submission: pd.DataFrame, sample: pd.DataFrame
) -> dict[str, int]:
    submitted_ids = _validate_ids(submission, context="submission")
    sample_ids = _validate_ids(sample, context="sample submission")
    if len(submission) != len(sample):
        raise ValueError(
            f"submission rows={len(submission)}, sample rows={len(sample)}"
        )
    if not submitted_ids.reset_index(drop=True).equals(
        sample_ids.reset_index(drop=True)
    ):
        raise ValueError("submission ID order does not exactly match sample submission")
    values = pd.to_numeric(submission["tvt"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("submission tvt contains NaN or infinite values")
    return {"rows": len(submission), "unique_ids": int(submitted_ids.nunique())}


def write_submission(submission: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(destination, index=False, columns=list(SUBMISSION_COLUMNS))
    return destination


def validate_submission_file(
    submission_path: str | Path,
    sample_path: str | Path,
) -> dict[str, int | str]:
    submission_source = Path(submission_path)
    sample_source = Path(sample_path)
    submission = pd.read_csv(submission_source, dtype={"id": "string"})
    sample = pd.read_csv(sample_source, dtype={"id": "string"})
    result: dict[str, int | str] = validate_submission_frame(submission, sample)
    result["path"] = str(submission_source)
    result["sha256"] = sha256_file(submission_source)
    return result
