from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rogii.features import (
    TYPEWELL_GR_SLOPE_FEATURE_COLUMNS,
    build_typewell_gr_slope_features,
)


def _well() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MD": [0.0, 1.0, 2.0, 3.0],
            "X": [0.0] * 4,
            "Y": [0.0] * 4,
            "Z": [0.0] * 4,
            "GR": [10.0, 20.0, 30.0, np.nan],
            "TVT_input": [99.0, 100.0, np.nan, np.nan],
        }
    )


def test_typewell_gr_slope_is_the_only_added_feature() -> None:
    centered = np.arange(-100.0, 101.0)
    typewell = pd.DataFrame(
        {"TVT": 100.0 + centered, "GR": 2.0 + 3.0 * centered + 0.5 * centered**2}
    )
    features = build_typewell_gr_slope_features(_well(), typewell)
    assert tuple(features.columns) == TYPEWELL_GR_SLOPE_FEATURE_COLUMNS
    np.testing.assert_allclose(features["typewell_gr_slope_100m"], [3.0, 3.0])


def test_typewell_gr_slope_ignores_nonfinite_rows() -> None:
    typewell = pd.DataFrame(
        {"TVT": [99.0, 100.0, 101.0, np.nan], "GR": [1.0, 2.0, 3.0, 99.0]}
    )
    features = build_typewell_gr_slope_features(_well(), typewell)
    assert np.isfinite(features["typewell_gr_slope_100m"]).all()


def test_typewell_gr_slope_fails_closed_without_three_local_rows() -> None:
    typewell = pd.DataFrame({"TVT": [0.0, 100.0], "GR": [1.0, 2.0]})
    with pytest.raises(ValueError, match="three finite local rows"):
        build_typewell_gr_slope_features(_well(), typewell)


def test_typewell_gr_slope_rejects_wrong_schema() -> None:
    with pytest.raises(ValueError, match="exactly TVT and GR"):
        build_typewell_gr_slope_features(_well(), pd.DataFrame({"GR": [1.0]}))
