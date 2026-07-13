from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rogii.features import BASELINE_B_FEATURE_COLUMNS, build_baseline_b_features
from rogii.io import INFERENCE_COLUMNS


def inference_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MD": [100.0, 101.0, 103.0, 106.0],
            "X": [10.0, 11.0, 14.0, 18.0],
            "Y": [20.0, 21.0, 19.0, 17.0],
            "Z": [-30.0, -31.0, -33.0, -36.0],
            "GR": [5.0, 6.0, 7.0, np.nan],
            "TVT_input": [900.0, 901.0, np.nan, np.nan],
        },
        columns=INFERENCE_COLUMNS,
    )


def test_anchor_and_relative_features_are_exact() -> None:
    features = build_baseline_b_features(inference_frame())

    assert tuple(features.columns) == BASELINE_B_FEATURE_COLUMNS
    assert features["anchor_tvt_input"].tolist() == [901.0, 901.0]
    assert features["row_offset"].tolist() == [1.0, 2.0]
    assert features["prediction_fraction"].tolist() == [0.5, 1.0]
    assert features["md_delta_anchor"].tolist() == [2.0, 5.0]
    assert features["x_delta_anchor"].tolist() == [3.0, 7.0]
    assert features["y_delta_anchor"].tolist() == [-2.0, -4.0]
    assert features["z_delta_anchor"].tolist() == [-2.0, -5.0]
    assert features["gr_missing"].tolist() == [0.0, 1.0]
    assert features["anchor_gr_missing"].tolist() == [0.0, 0.0]
    assert features["gr_delta_anchor"].iloc[0] == 1.0
    assert np.isnan(features["gr_delta_anchor"].iloc[1])


@pytest.mark.parametrize(
    "values,match",
    [
        ([np.nan, np.nan, np.nan, np.nan], "no known"),
        ([900.0, np.nan, 901.0, np.nan], "contiguous suffix"),
        ([900.0, 901.0, 902.0, 903.0], "no prediction-zone"),
    ],
)
def test_malformed_prediction_suffix_fails_closed(values, match) -> None:
    frame = inference_frame()
    frame["TVT_input"] = values
    with pytest.raises(ValueError, match=match):
        build_baseline_b_features(frame)


@pytest.mark.parametrize(
    "forbidden",
    ["TVT", "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA", "Geology"],
)
def test_training_target_and_forbidden_columns_are_rejected(forbidden: str) -> None:
    frame = inference_frame()
    frame[forbidden] = 123.0
    with pytest.raises(ValueError, match="unexpected"):
        build_baseline_b_features(frame)


def test_missing_required_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing"):
        build_baseline_b_features(inference_frame().drop(columns="X"))


def test_gr_missing_values_are_preserved_without_global_fill() -> None:
    frame = inference_frame()
    frame.loc[1, "GR"] = np.nan
    features = build_baseline_b_features(frame)

    assert features["anchor_gr_missing"].tolist() == [1.0, 1.0]
    assert features["gr_missing"].tolist() == [0.0, 1.0]
    assert features["gr_delta_anchor"].isna().all()
    assert not np.isinf(features.to_numpy(dtype=np.float64)).any()
