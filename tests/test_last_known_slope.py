import numpy as np
import pandas as pd
import pytest

from rogii.features import (
    LAST_KNOWN_SLOPE_FEATURE_COLUMNS,
    build_last_known_slope_features,
)


def _well(md=(0.0, 2.0, 3.0, 4.0), tvt=(100.0, 106.0, np.nan, np.nan)):
    return pd.DataFrame(
        {
            "MD": list(md),
            "X": [0.0, 1.0, 2.0, 3.0],
            "Y": [0.0, 0.0, 0.0, 0.0],
            "Z": [0.0, -1.0, -2.0, -3.0],
            "GR": [10.0, 11.0, 12.0, 13.0],
            "TVT_input": list(tvt),
        }
    )


def test_last_known_slope_is_the_only_added_feature() -> None:
    features = build_last_known_slope_features(_well())
    assert tuple(features.columns) == LAST_KNOWN_SLOPE_FEATURE_COLUMNS
    np.testing.assert_allclose(features["last_known_dTVT_dMD"], [3.0, 3.0])


def test_last_known_slope_requires_two_known_rows() -> None:
    frame = _well(tvt=(100.0, np.nan, np.nan, np.nan))
    with pytest.raises(ValueError, match="requires 2 known"):
        build_last_known_slope_features(frame)


def test_last_known_slope_rejects_zero_md_variance() -> None:
    with pytest.raises(ValueError, match="nonzero MD variance"):
        build_last_known_slope_features(_well(md=(2.0, 2.0, 3.0, 4.0)))


def test_last_known_ols_slope_uses_exact_requested_window() -> None:
    frame = pd.DataFrame(
        {
            "MD": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "X": [0.0] * 6,
            "Y": [0.0] * 6,
            "Z": [0.0] * 6,
            "GR": [10.0] * 6,
            "TVT_input": [100.0, 101.0, 104.0, 109.0, np.nan, np.nan],
        }
    )
    features = build_last_known_slope_features(frame, known_window=4)
    np.testing.assert_allclose(features["last_known_dTVT_dMD"], [3.0, 3.0])


@pytest.mark.parametrize("window", [True, 1, 2.5])
def test_last_known_slope_rejects_invalid_window(window) -> None:
    with pytest.raises(ValueError, match="known_window"):
        build_last_known_slope_features(_well(), known_window=window)


def test_last_known_slope_fails_closed_when_window_is_unavailable() -> None:
    with pytest.raises(ValueError, match="requires 3 known"):
        build_last_known_slope_features(_well(), known_window=3)
