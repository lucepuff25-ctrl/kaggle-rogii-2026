"""Official ROGII competition metric."""

from __future__ import annotations

from typing import Any

import numpy as np


def _one_dimensional_finite(name: str, values: Any) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def mean_squared_error(y_true: Any, y_pred: Any) -> float:
    """Return positional MSE for two finite one-dimensional numeric arrays."""
    truth = _one_dimensional_finite("y_true", y_true)
    prediction = _one_dimensional_finite("y_pred", y_pred)
    if truth.shape != prediction.shape:
        raise ValueError(f"y_true and y_pred must have the same shape: {truth.shape} != {prediction.shape}")
    with np.errstate(over="raise", invalid="raise"):
        try:
            score = np.mean(np.square(truth - prediction), dtype=np.float64)
        except FloatingPointError as exc:
            raise ValueError("MSE is not finite for the supplied values") from exc
    if not np.isfinite(score):
        raise ValueError("MSE is not finite for the supplied values")
    return float(score)
