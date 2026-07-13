from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rogii.io import (
    FOLD_COLUMNS,
    INFERENCE_COLUMNS,
    display_path,
    discover_horizontal_wells,
    load_fold_mapping,
    prediction_ids,
    prediction_mask,
    read_horizontal_well,
    sha256_file,
)
from rogii.quarantine import PUBLIC_SAMPLE_OVERLAP_WELLS


def horizontal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MD": [1.0, 2.0, 3.0, 4.0],
            "X": [10.0] * 4,
            "Y": [20.0] * 4,
            "Z": [-1.0, -2.0, -3.0, -4.0],
            "GR": [30.0, np.nan, 31.0, 32.0],
            "TVT_input": [100.0, 101.0, np.nan, np.nan],
            "TVT": [100.0, 101.0, 102.0, 103.0],
            "ANCC": [999.0] * 4,
        }
    )


def test_prediction_zone_uses_only_tvt_input_and_ids_keep_source_rows() -> None:
    frame = horizontal_frame()
    assert prediction_mask(frame).tolist() == [False, False, True, True]
    assert prediction_ids("ABCDEF12", frame).tolist() == ["abcdef12_2", "abcdef12_3"]


@pytest.mark.parametrize(
    "values",
    [
        [np.nan, np.nan],
        [1.0, np.nan, 2.0],
        [1.0, 2.0],
        [1.0, np.inf, np.nan],
    ],
)
def test_malformed_prediction_regions_fail_closed(values) -> None:
    with pytest.raises(ValueError):
        prediction_mask(pd.DataFrame({"TVT_input": values}))


def test_reader_excludes_train_only_features(tmp_path: Path) -> None:
    path = tmp_path / "abcdef12__horizontal_well.csv"
    horizontal_frame().to_csv(path, index=False)
    loaded = read_horizontal_well(path, include_target=True)
    assert set(loaded.columns) == {*INFERENCE_COLUMNS, "TVT"}
    assert "ANCC" not in loaded.columns


def test_discovery_is_dynamic_and_sorted(tmp_path: Path) -> None:
    for well_id in ("future99", "abc00001"):
        (tmp_path / f"{well_id}__horizontal_well.csv").write_text(
            "TVT_input\n1\n", encoding="utf-8"
        )
    wells = discover_horizontal_wells(tmp_path)
    assert [well.well_id for well in wells] == ["abc00001", "future99"]


def test_display_path_keeps_external_artifact_absolute(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    inside = root / "models" / "algorithm.json"
    outside = tmp_path / "external" / "algorithm.json"
    assert display_path(inside, relative_to=root) == "models/algorithm.json"
    assert display_path(outside, relative_to=root) == outside.resolve().as_posix()


def _write_mapping(path: Path, well_ids: list[str] | None = None) -> None:
    ids = well_ids or [f"honest{i:02d}" for i in range(5)]
    pd.DataFrame(
        {
            "well_id": ids,
            "typewell_group": [f"group{i}" for i in range(5)],
            "fold": list(range(5)),
            "prediction_rows": [1] * 5,
        },
        columns=FOLD_COLUMNS,
    ).to_csv(path, index=False, lineterminator="\n")


def test_fold_mapping_validates_hash_and_quarantine(tmp_path: Path) -> None:
    path = tmp_path / "folds.csv"
    _write_mapping(path)
    loaded = load_fold_mapping(
        path,
        expected_sha256=sha256_file(path),
        n_splits=5,
        expected_wells=5,
        expected_groups=5,
    )
    assert loaded["fold"].tolist() == list(range(5))

    with pytest.raises(ValueError, match="SHA256"):
        load_fold_mapping(path, expected_sha256="0" * 64, n_splits=5)
    quarantined = sorted(PUBLIC_SAMPLE_OVERLAP_WELLS)[0]
    _write_mapping(
        path,
        [quarantined, "honest01", "honest02", "honest03", "honest04"],
    )
    with pytest.raises(ValueError, match="quarantined"):
        load_fold_mapping(path, expected_sha256=sha256_file(path), n_splits=5)
