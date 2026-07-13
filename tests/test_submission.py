from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rogii.submission import (
    align_predictions_to_sample,
    validate_submission_file,
    validate_submission_frame,
    well_prediction_frame,
)


def test_predictions_are_reordered_to_exact_sample_order() -> None:
    predictions = pd.DataFrame(
        {"id": ["future02_3", "future01_2"], "tvt": [203.0, 102.0]}
    )
    sample = pd.DataFrame({"id": ["future01_2", "future02_3"], "tvt": [0.0, 0.0]})
    submission = align_predictions_to_sample(predictions, sample)
    assert submission.to_dict("list") == {
        "id": ["future01_2", "future02_3"],
        "tvt": [102.0, 203.0],
    }
    assert validate_submission_frame(submission, sample) == {"rows": 2, "unique_ids": 2}


def test_unknown_well_ids_are_built_dynamically() -> None:
    horizontal = pd.DataFrame({"TVT_input": [10.0, 11.0, np.nan, np.nan]})
    result = well_prediction_frame("FUTURE99", horizontal, np.array([11.0, 11.0]))
    assert result["id"].tolist() == ["future99_2", "future99_3"]


@pytest.mark.parametrize(
    "predictions,match",
    [
        (pd.DataFrame({"id": ["a_1", "a_1"], "tvt": [1.0, 2.0]}), "not unique"),
        (pd.DataFrame({"id": ["a_1", "extra_2"], "tvt": [1.0, 2.0]}), "do not match"),
        (pd.DataFrame({"id": ["a_1", "b_2"], "tvt": [1.0, np.inf]}), "NaN or infinite"),
    ],
)
def test_bad_predictions_are_rejected(predictions, match) -> None:
    sample = pd.DataFrame({"id": ["a_1", "b_2"], "tvt": [0.0, 0.0]})
    with pytest.raises(ValueError, match=match):
        align_predictions_to_sample(predictions, sample)


def test_file_validator_checks_order_and_returns_sha(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.csv"
    submission_path = tmp_path / "submission.csv"
    sample = pd.DataFrame({"id": ["a_1", "b_2"], "tvt": [0.0, 0.0]})
    sample.to_csv(sample_path, index=False)
    pd.DataFrame({"id": ["a_1", "b_2"], "tvt": [1.0, 2.0]}).to_csv(
        submission_path, index=False
    )
    result = validate_submission_file(submission_path, sample_path)
    assert result["rows"] == 2
    assert len(str(result["sha256"])) == 64

    pd.DataFrame({"id": ["b_2", "a_1"], "tvt": [2.0, 1.0]}).to_csv(
        submission_path, index=False
    )
    with pytest.raises(ValueError, match="order"):
        validate_submission_file(submission_path, sample_path)
