"""Deterministic, group-exclusive cross-validation utilities."""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_SEED = 20260713


def _group_series(
    frame: pd.DataFrame | pd.Series,
    group_col: str | Sequence[str] | None,
) -> pd.Series:
    if isinstance(frame, pd.Series):
        if group_col is not None:
            raise ValueError("group_col must be None when frame is a Series")
        groups = frame.copy(deep=False)
    elif isinstance(frame, pd.DataFrame):
        if group_col is None:
            raise ValueError("group_col is required for a DataFrame")
        columns = [group_col] if isinstance(group_col, str) else list(group_col)
        if not columns:
            raise ValueError("at least one grouping column is required")
        missing_columns = [column for column in columns if column not in frame.columns]
        if missing_columns:
            raise KeyError(f"missing grouping columns: {missing_columns}")
        if frame[columns].isna().any().any():
            raise ValueError("grouping keys contain missing values")
        if len(columns) == 1:
            groups = frame[columns[0]].copy(deep=False)
        else:
            groups = frame[columns].astype("string").agg("\x1f".join, axis=1)
    else:
        raise TypeError("frame must be a pandas DataFrame or Series")
    if groups.isna().any():
        raise ValueError("grouping keys contain missing values")
    return groups


def _tie_break(value: object, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}\x1f{value!r}".encode("utf-8")).digest()


def assign_group_folds(
    frame: pd.DataFrame | pd.Series,
    group_col: str | Sequence[str] | None = None,
    *,
    n_splits: int = 5,
    seed: int = DEFAULT_SEED,
) -> pd.Series:
    """Assign every row to one validation fold without splitting a group.

    Groups are ordered by descending row count with a seeded stable hash as the
    tie-breaker, then greedily assigned to the currently smallest fold.
    """
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    groups = _group_series(frame, group_col)
    if groups.empty:
        raise ValueError("cannot split an empty dataset")
    counts = groups.value_counts(sort=False, dropna=False)
    if len(counts) < n_splits:
        raise ValueError(f"n_splits={n_splits} exceeds the {len(counts)} unique groups")

    ordered_groups = sorted(counts.index, key=lambda value: (-int(counts[value]), _tie_break(value, seed)))
    fold_sizes = [0] * n_splits
    group_to_fold: dict[object, int] = {}
    for value in ordered_groups:
        fold = min(range(n_splits), key=lambda candidate: (fold_sizes[candidate], candidate))
        group_to_fold[value] = fold
        fold_sizes[fold] += int(counts[value])

    folds = groups.map(group_to_fold).astype(np.int64)
    folds.name = "fold"
    validate_group_folds(groups, folds, n_splits=n_splits)
    return folds


def validate_group_folds(
    groups: pd.Series,
    folds: pd.Series | np.ndarray | Sequence[int],
    *,
    n_splits: int | None = None,
) -> None:
    """Raise when rows are uncovered or a group crosses validation folds."""
    group_values = pd.Series(groups, copy=False).reset_index(drop=True)
    fold_values = pd.Series(folds, copy=False).reset_index(drop=True)
    if len(group_values) != len(fold_values):
        raise ValueError("groups and folds must have the same length")
    if group_values.isna().any():
        raise ValueError("grouping keys contain missing values")
    if fold_values.isna().any():
        raise ValueError("every row must have a validation fold")
    if not pd.api.types.is_integer_dtype(fold_values.dtype):
        raise ValueError("fold labels must be integers")
    unique_folds = sorted(int(value) for value in fold_values.unique())
    if n_splits is not None and unique_folds != list(range(n_splits)):
        raise ValueError(f"fold labels must cover 0..{n_splits - 1}")
    group_fold_counts = pd.DataFrame({"group": group_values, "fold": fold_values}).groupby(
        "group", dropna=False, sort=False
    )["fold"].nunique()
    if (group_fold_counts != 1).any():
        raise ValueError("at least one group appears in multiple validation folds")
