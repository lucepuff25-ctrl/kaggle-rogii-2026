import pandas as pd
import pytest

from rogii.quarantine import (
    PUBLIC_SAMPLE_OVERLAP_WELLS,
    assert_no_public_sample_overlap,
    partition_public_sample_overlap,
    public_sample_overlap_mask,
)


def test_known_public_sample_wells_are_fixed_and_complete() -> None:
    assert PUBLIC_SAMPLE_OVERLAP_WELLS == {"000d7d20", "00bbac68", "00e12e8b"}


def test_all_three_wells_are_quarantined_case_insensitively() -> None:
    frame = pd.DataFrame({"well_id": ["000D7D20", "00BBAC68", "00E12E8B", "honest"]})
    honest, quarantined = partition_public_sample_overlap(frame)
    assert honest["well_id"].tolist() == ["honest"]
    assert set(quarantined["well_id"].str.lower()) == PUBLIC_SAMPLE_OVERLAP_WELLS


def test_mask_and_partition_quarantine_all_matching_rows() -> None:
    frame = pd.DataFrame(
        {
            "well_id": ["honest_a", "000D7D20", "honest_b", "00bbac68", "00bbac68"],
            "value": [1, 2, 3, 4, 5],
        }
    )
    mask = public_sample_overlap_mask(frame)
    assert mask.tolist() == [False, True, False, True, True]
    honest, quarantined = partition_public_sample_overlap(frame)
    assert honest["well_id"].tolist() == ["honest_a", "honest_b"]
    assert quarantined["value"].tolist() == [2, 4, 5]
    assert frame.shape == (5, 2)


def test_assertion_fails_closed_for_quarantined_wells() -> None:
    with pytest.raises(ValueError, match="00e12e8b"):
        assert_no_public_sample_overlap(["honest", "00e12e8b"], context="CV")
    assert_no_public_sample_overlap(["honest_a", "honest_b"])


def test_assertion_rejects_missing_ids_and_accepts_770_honest_wells() -> None:
    with pytest.raises(ValueError, match="missing values"):
        assert_no_public_sample_overlap(["honest", None])
    assert_no_public_sample_overlap([f"honest_{index:03d}" for index in range(770)])


def test_missing_or_null_well_ids_are_rejected() -> None:
    with pytest.raises(KeyError, match="missing well column"):
        public_sample_overlap_mask(pd.DataFrame({"other": [1]}))
    with pytest.raises(ValueError, match="missing values"):
        public_sample_overlap_mask(pd.DataFrame({"well_id": ["a", None]}))
