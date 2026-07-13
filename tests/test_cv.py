import numpy as np
import pandas as pd
import pytest

from rogii.cv import assign_group_folds, validate_group_folds


def example_frame() -> pd.DataFrame:
    return pd.DataFrame({"well_id": ["a"] * 4 + ["b"] * 3 + ["c"] * 2 + ["d"] * 2 + ["e"]})


def test_same_well_never_crosses_folds_and_input_is_unchanged() -> None:
    frame = example_frame()
    original = frame.copy(deep=True)
    folds = assign_group_folds(frame, "well_id", n_splits=3)
    assert pd.DataFrame({"well": frame["well_id"], "fold": folds}).groupby("well")["fold"].nunique().eq(1).all()
    pd.testing.assert_frame_equal(frame, original)


def test_result_is_reproducible() -> None:
    first = assign_group_folds(example_frame(), "well_id", n_splits=3, seed=7)
    second = assign_group_folds(example_frame(), "well_id", n_splits=3, seed=7)
    pd.testing.assert_series_equal(first, second)


def test_every_position_is_covered_once() -> None:
    folds = assign_group_folds(example_frame(), "well_id", n_splits=3)
    assert len(folds) == len(example_frame())
    assert not folds.isna().any()
    validation_positions = np.concatenate([np.flatnonzero(folds.to_numpy() == fold) for fold in range(3)])
    assert sorted(validation_positions.tolist()) == list(range(len(folds)))


def test_missing_group_column_or_key_raises() -> None:
    with pytest.raises(KeyError, match="missing grouping columns"):
        assign_group_folds(example_frame(), "missing", n_splits=2)
    with pytest.raises(ValueError, match="missing values"):
        assign_group_folds(pd.DataFrame({"well_id": ["a", None, "b"]}), "well_id", n_splits=2)


def test_tiny_dataset_behavior_is_explicit() -> None:
    with pytest.raises(ValueError, match="empty"):
        assign_group_folds(pd.Series([], dtype="string"), n_splits=2)
    with pytest.raises(ValueError, match="exceeds"):
        assign_group_folds(pd.Series(["only", "only"]), n_splits=2)


def test_validator_rejects_group_leakage_and_uncovered_fold() -> None:
    with pytest.raises(ValueError, match="multiple"):
        validate_group_folds(pd.Series(["a", "a", "b"]), [0, 1, 1], n_splits=2)
    with pytest.raises(ValueError, match="cover"):
        validate_group_folds(pd.Series(["a", "b"]), [0, 0], n_splits=2)


def test_series_and_composite_groups_are_supported() -> None:
    series_folds = assign_group_folds(pd.Series(["a", "a", "b", "c"]), n_splits=2)
    assert series_folds.iloc[0] == series_folds.iloc[1]
    frame = pd.DataFrame({"well": ["a", "a", "a", "a"], "region": [1, 1, 2, 2]})
    composite = assign_group_folds(frame, ["well", "region"], n_splits=2)
    assert composite.iloc[0] == composite.iloc[1]
    assert composite.iloc[2] == composite.iloc[3]


def test_sample_weight_balances_group_weight_deterministically() -> None:
    frame = pd.DataFrame({"source": ["a", "b", "c", "d"], "rows": [100, 60, 40, 1]})
    folds = assign_group_folds(
        frame,
        "source",
        n_splits=2,
        seed=11,
        sample_weight=frame["rows"].to_numpy(),
    )
    repeated = assign_group_folds(
        frame,
        "source",
        n_splits=2,
        seed=11,
        sample_weight=frame["rows"].to_numpy(),
    )
    pd.testing.assert_series_equal(folds, repeated)
    totals = frame.assign(fold=folds).groupby("fold")["rows"].sum().to_dict()
    assert sorted(totals.values()) == [100, 101]


@pytest.mark.parametrize("weights", [[1.0, -1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]])
def test_invalid_sample_weights_raise(weights) -> None:
    frame = pd.DataFrame({"source": ["a", "b", "c", "d"]})
    with pytest.raises(ValueError, match="sample_weight"):
        assign_group_folds(frame, "source", n_splits=2, sample_weight=weights)
