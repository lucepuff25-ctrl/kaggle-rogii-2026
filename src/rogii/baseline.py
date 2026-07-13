"""Deterministic, train-free ROGII Baseline A."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import prediction_mask
from .quarantine import QUARANTINE_POLICY_VERSION

BASELINE_A_METHOD = "last_known_tvt_input"
BASELINE_A_USED_FIELDS = ("TVT_input", "row_order")
BASELINE_A_ARTIFACT_TYPE = "rogii_baseline_a_algorithm"
BASELINE_A_ARTIFACT_SCHEMA_VERSION = 1


def predict_baseline_a(frame: pd.DataFrame) -> np.ndarray:
    """Extend the final known TVT_input value through the prediction suffix."""
    mask = prediction_mask(frame)
    first_prediction = int(np.flatnonzero(mask.to_numpy())[0])
    anchor = float(
        pd.to_numeric(frame["TVT_input"], errors="raise").iloc[first_prediction - 1]
    )
    predictions = np.full(int(mask.sum()), anchor, dtype=np.float64)
    if not np.isfinite(predictions).all():
        raise ValueError("Baseline A produced non-finite predictions")
    return predictions


@dataclass(frozen=True)
class BaselineAAlgorithm:
    """Validated algorithm contract with no fitted or target-derived state."""

    artifact_type: str
    schema_version: int
    method: str
    used_fields: tuple[str, ...]
    fold_mapping_sha256: str
    quarantine_policy_version: str

    @classmethod
    def create(cls, *, fold_mapping_sha256: str) -> "BaselineAAlgorithm":
        artifact = cls(
            artifact_type=BASELINE_A_ARTIFACT_TYPE,
            schema_version=BASELINE_A_ARTIFACT_SCHEMA_VERSION,
            method=BASELINE_A_METHOD,
            used_fields=BASELINE_A_USED_FIELDS,
            fold_mapping_sha256=fold_mapping_sha256,
            quarantine_policy_version=QUARANTINE_POLICY_VERSION,
        )
        artifact.validate(expected_fold_mapping_sha256=fold_mapping_sha256)
        return artifact

    def validate(self, *, expected_fold_mapping_sha256: str) -> None:
        if self.artifact_type != BASELINE_A_ARTIFACT_TYPE:
            raise ValueError("Baseline A artifact type mismatch")
        if self.schema_version != BASELINE_A_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("Baseline A artifact schema version mismatch")
        if self.method != BASELINE_A_METHOD:
            raise ValueError("Baseline A artifact method mismatch")
        if self.used_fields != BASELINE_A_USED_FIELDS:
            raise ValueError("Baseline A artifact used_fields mismatch")
        if self.quarantine_policy_version != QUARANTINE_POLICY_VERSION:
            raise ValueError("Baseline A artifact quarantine policy mismatch")
        digest = self.fold_mapping_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("Baseline A artifact fold SHA256 is invalid")
        if digest != expected_fold_mapping_sha256:
            raise ValueError("Baseline A artifact fold SHA256 mismatch")

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return predict_baseline_a(frame)

    def as_json_object(self) -> dict[str, int | str | list[str]]:
        return {
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "method": self.method,
            "used_fields": list(self.used_fields),
            "fold_mapping_sha256": self.fold_mapping_sha256,
            "quarantine_policy_version": self.quarantine_policy_version,
        }


def save_baseline_a_algorithm(artifact: BaselineAAlgorithm, path: str | Path) -> Path:
    artifact.validate(expected_fold_mapping_sha256=artifact.fold_mapping_sha256)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(artifact.as_json_object(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def load_baseline_a_algorithm(
    path: str | Path,
    *,
    expected_fold_mapping_sha256: str,
) -> BaselineAAlgorithm:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_keys = {
        "artifact_type",
        "schema_version",
        "method",
        "used_fields",
        "fold_mapping_sha256",
        "quarantine_policy_version",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("Baseline A artifact keys mismatch")
    if not isinstance(payload["used_fields"], list):
        raise ValueError("Baseline A artifact used_fields must be a list")
    artifact = BaselineAAlgorithm(
        artifact_type=payload["artifact_type"],
        schema_version=payload["schema_version"],
        method=payload["method"],
        used_fields=tuple(payload["used_fields"]),
        fold_mapping_sha256=payload["fold_mapping_sha256"],
        quarantine_policy_version=payload["quarantine_policy_version"],
    )
    artifact.validate(expected_fold_mapping_sha256=expected_fold_mapping_sha256)
    return artifact
