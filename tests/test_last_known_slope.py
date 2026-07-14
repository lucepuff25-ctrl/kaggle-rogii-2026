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
    with pytest.raises(ValueError, match="requires two known"):
        build_last_known_slope_features(frame)


def test_last_known_slope_rejects_zero_md_delta() -> None:
    with pytest.raises(ValueError, match="nonzero MD"):
        build_last_known_slope_features(_well(md=(2.0, 2.0, 3.0, 4.0)))
