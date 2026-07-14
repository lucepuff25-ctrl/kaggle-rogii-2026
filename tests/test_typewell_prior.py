import numpy as np
import pandas as pd
import pytest

from rogii.features import (
    TYPEWELL_PRIOR_FEATURE_COLUMNS,
    build_typewell_prior_features,
)


def _well() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MD": [0.0, 1.0, 2.0, 3.0],
            "X": [0.0, 1.0, 2.0, 3.0],
            "Y": [0.0, 0.0, 0.0, 0.0],
            "Z": [0.0, -1.0, -2.0, -3.0],
            "GR": [10.0, 11.0, 12.0, 13.0],
            "TVT_input": [100.0, 101.0, np.nan, np.nan],
        }
    )


def test_typewell_prior_is_anchor_aligned_and_deterministic() -> None:
    typewell = pd.DataFrame({"TVT": [10.0, 13.0, 16.0, 19.0]})
    first = build_typewell_prior_features(_well(), typewell)
    second = build_typewell_prior_features(_well(), typewell)
    assert tuple(first.columns) == TYPEWELL_PRIOR_FEATURE_COLUMNS
    assert first.equals(second)
    np.testing.assert_allclose(first["typewell_tvt_prior"], [104.0, 107.0])


def test_typewell_prior_fails_closed_on_bad_curve() -> None:
    with pytest.raises(ValueError, match="at least two finite rows"):
        build_typewell_prior_features(_well(), pd.DataFrame({"TVT": [1.0]}))
