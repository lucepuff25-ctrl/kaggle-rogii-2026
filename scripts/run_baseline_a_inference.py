#!/usr/bin/env python3
"""Generate a dynamic, sample-aligned ROGII Baseline A submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii.baseline import BASELINE_A_METHOD, load_baseline_a_algorithm
from rogii.io import (
    INFERENCE_COLUMNS,
    discover_horizontal_wells,
    read_horizontal_well,
    sha256_file,
)
from rogii.quarantine import QUARANTINE_POLICY_VERSION
from rogii.submission import (
    align_predictions_to_sample,
    validate_submission_file,
    well_prediction_frame,
    write_submission,
)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if config["method"] != BASELINE_A_METHOD:
        raise ValueError(f"unsupported Baseline A method: {config['method']}")
    if config["quarantine_policy_version"] != QUARANTINE_POLICY_VERSION:
        raise ValueError("quarantine policy version mismatch")

    artifact_path = _path(config["artifact_path"])
    artifact = load_baseline_a_algorithm(
        artifact_path,
        expected_fold_mapping_sha256=config["fold_mapping_sha256"],
    )

    started = time.perf_counter()
    prediction_frames: list[pd.DataFrame] = []
    wells = discover_horizontal_wells(_path(config["test_dir"]))
    for well in wells:
        horizontal = read_horizontal_well(well.path, include_target=False)
        inference_frame = horizontal.loc[:, list(INFERENCE_COLUMNS)]
        predictions = artifact.predict(inference_frame)
        prediction_frames.append(
            well_prediction_frame(well.well_id, horizontal, predictions)
        )
    raw_predictions = pd.concat(prediction_frames, ignore_index=True)
    sample_path = _path(config["sample_submission"])
    sample = pd.read_csv(sample_path, dtype={"id": "string"})
    submission = align_predictions_to_sample(raw_predictions, sample)
    destination = write_submission(submission, _path(config["submission_path"]))
    elapsed = time.perf_counter() - started

    validation = validate_submission_file(destination, sample_path)
    result = {
        "status": "ok",
        "method": BASELINE_A_METHOD,
        "test_wells": len(wells),
        "prediction_rows": len(submission),
        "inference_seconds": elapsed,
        "nan_count": int(np.isnan(submission["tvt"].to_numpy(dtype=np.float64)).sum()),
        "inf_count": int(np.isinf(submission["tvt"].to_numpy(dtype=np.float64)).sum()),
        "gpu": "none",
        "trained_model": False,
        "artifact_path": artifact_path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_validated": True,
        "validation": validation,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
