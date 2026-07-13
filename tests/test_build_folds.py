from pathlib import Path

import pandas as pd

from build_folds import build_folds
from rogii.quarantine import PUBLIC_SAMPLE_OVERLAP_WELLS


def write_well(train_dir: Path, well_id: str, offset: int) -> None:
    (train_dir / f"{well_id}__horizontal_well.csv").write_text(
        "TVT,TVT_input\n1,1\n2,\n3,\n",
        encoding="utf-8",
    )
    (train_dir / f"{well_id}__typewell.csv").write_text(
        f"TVT,GR,Geology\n{offset}.0,2,A\n{offset + 1},3,B\n",
        encoding="utf-8",
    )


def test_build_folds_is_deterministic_and_quarantines_public_samples(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    honest = [f"honest{i}" for i in range(5)]
    for offset, well_id in enumerate(honest + sorted(PUBLIC_SAMPLE_OVERLAP_WELLS), start=1):
        write_well(train_dir, well_id, offset)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    first_mapping, second_mapping = tmp_path / "first.csv", tmp_path / "second.csv"
    first = build_folds(train_dir, first_mapping, tmp_path / "first.json", manifest, n_splits=2, seed=7)
    second = build_folds(train_dir, second_mapping, tmp_path / "second.json", manifest, n_splits=2, seed=7)

    mapping = pd.read_csv(first_mapping)
    assert set(mapping["well_id"]) == set(honest)
    assert not set(mapping["well_id"]) & PUBLIC_SAMPLE_OVERLAP_WELLS
    assert first["original_train_wells"] == 8
    assert first["quarantined_wells"] == 3
    assert first["effective_wells"] == 5
    assert first["fold_mapping_sha256"] == second["fold_mapping_sha256"]
    assert first_mapping.read_bytes() == second_mapping.read_bytes()
