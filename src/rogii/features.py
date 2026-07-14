"""Leakage-safe, per-well features for ROGII Baseline B."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import INFERENCE_COLUMNS, prediction_mask

BASELINE_B_FEATURE_COLUMNS = (
    "anchor_tvt_input",
    "row_offset",
    "prediction_fraction",
    "md_delta_anchor",
    "x_delta_anchor",
    "y_delta_anchor",
    "z_delta_anchor",
    "gr",
    "gr_delta_anchor",
    "gr_missing",
    "anchor_gr_missing",
)
TYPEWELL_PRIOR_FEATURE_COLUMNS = BASELINE_B_FEATURE_COLUMNS + (
    "typewell_tvt_prior",
)


def build_baseline_b_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build prediction-suffix features using only this well's inference inputs."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if tuple(frame.columns) != INFERENCE_COLUMNS:
        missing = sorted(set(INFERENCE_COLUMNS) - set(frame.columns))
        unexpected = sorted(set(frame.columns) - set(INFERENCE_COLUMNS))
        raise ValueError(
            "Baseline B inference columns mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    numeric = frame.copy()
    for column in INFERENCE_COLUMNS:
        numeric[column] = pd.to_numeric(numeric[column], errors="raise")

    for column in ("MD", "X", "Y", "Z"):
        values = numeric[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite {column} in Baseline B input")
    gr_values = numeric["GR"].dropna().to_numpy(dtype=np.float64)
    if not np.isfinite(gr_values).all():
        raise ValueError("non-finite GR in Baseline B input")

    mask = prediction_mask(numeric)
    first_prediction = int(np.flatnonzero(mask.to_numpy())[0])
    anchor_row = numeric.iloc[first_prediction - 1]
    prediction = numeric.iloc[first_prediction:].reset_index(drop=True)
    prediction_rows = len(prediction)

    anchor_tvt = float(anchor_row["TVT_input"])
    anchor_gr = float(anchor_row["GR"])
    row_offset = np.arange(1, prediction_rows + 1, dtype=np.float64)
    features = pd.DataFrame(
        {
            "anchor_tvt_input": np.full(
                prediction_rows, anchor_tvt, dtype=np.float64
            ),
            "row_offset": row_offset,
            "prediction_fraction": row_offset / float(prediction_rows),
            "md_delta_anchor": prediction["MD"].to_numpy(dtype=np.float64)
            - float(anchor_row["MD"]),
            "x_delta_anchor": prediction["X"].to_numpy(dtype=np.float64)
            - float(anchor_row["X"]),
            "y_delta_anchor": prediction["Y"].to_numpy(dtype=np.float64)
            - float(anchor_row["Y"]),
            "z_delta_anchor": prediction["Z"].to_numpy(dtype=np.float64)
            - float(anchor_row["Z"]),
            "gr": prediction["GR"].to_numpy(dtype=np.float64),
            "gr_delta_anchor": prediction["GR"].to_numpy(dtype=np.float64)
            - anchor_gr,
            "gr_missing": prediction["GR"].isna().to_numpy(dtype=np.float64),
            "anchor_gr_missing": np.full(
                prediction_rows, float(np.isnan(anchor_gr)), dtype=np.float64
            ),
        },
        columns=BASELINE_B_FEATURE_COLUMNS,
    )

    values = features.to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError("Baseline B features contain infinite values")
    nullable = {"gr", "gr_delta_anchor"}
    for column in set(BASELINE_B_FEATURE_COLUMNS) - nullable:
        if features[column].isna().any():
            raise ValueError(f"Baseline B feature {column} contains NaN")
    return features


def build_typewell_prior_features(
    frame: pd.DataFrame, typewell: pd.DataFrame
) -> pd.DataFrame:
    """Add one deterministic typewell-TVT prior without observing target TVT."""
    features = build_baseline_b_features(frame)
    if not isinstance(typewell, pd.DataFrame) or tuple(typewell.columns) != ("TVT",):
        raise ValueError("typewell input must contain exactly the TVT column")
    values = pd.to_numeric(typewell["TVT"], errors="raise").to_numpy(
        dtype=np.float64
    )
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("typewell TVT must contain at least two finite rows")

    mask = prediction_mask(frame)
    first_prediction = int(np.flatnonzero(mask.to_numpy())[0])
    if len(frame) < 2:
        raise ValueError("horizontal well must contain at least two rows")
    horizontal_q = np.arange(len(frame), dtype=np.float64) / float(len(frame) - 1)
    typewell_q = np.arange(len(values), dtype=np.float64) / float(len(values) - 1)
    anchor_q = horizontal_q[first_prediction - 1]
    prior = float(frame.iloc[first_prediction - 1]["TVT_input"]) + np.interp(
        horizontal_q[first_prediction:], typewell_q, values
    ) - np.interp(anchor_q, typewell_q, values)
    if not np.isfinite(prior).all():
        raise ValueError("typewell_tvt_prior contains non-finite values")
    features["typewell_tvt_prior"] = prior
    return features.loc[:, list(TYPEWELL_PRIOR_FEATURE_COLUMNS)]
