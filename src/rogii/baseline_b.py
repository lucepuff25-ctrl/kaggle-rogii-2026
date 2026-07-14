"""Single-model CPU LightGBM Baseline B primitives."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import (
    BASELINE_B_FEATURE_COLUMNS,
    LAST_KNOWN_SLOPE_FEATURE_COLUMNS,
    TYPEWELL_GR_SLOPE_FEATURE_COLUMNS,
    TYPEWELL_PRIOR_FEATURE_COLUMNS,
)
from .io import FOLD_COLUMNS, sha256_file
from .quarantine import QUARANTINE_POLICY_VERSION, assert_no_public_sample_overlap

BASELINE_B_METHOD = "lightgbm_anchor_residual"
BASELINE_B_TARGET = "TVT_minus_last_known_TVT_input"
BASELINE_B_ARTIFACT_TYPE = "rogii_baseline_b_lightgbm"
BASELINE_B_ARTIFACT_SCHEMA_VERSION = 2
BASELINE_B_PARAMETER_KEYS = frozenset(
    {
        "objective",
        "metric",
        "learning_rate",
        "num_leaves",
        "min_data_in_leaf",
        "feature_fraction",
        "bagging_fraction",
        "bagging_freq",
        "device_type",
        "deterministic",
        "force_col_wise",
        "num_threads",
        "seed",
        "verbosity",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "artifact_type",
        "schema_version",
        "method",
        "target_definition",
        "feature_columns",
        "parameters",
        "num_boost_round",
        "fold_mapping_sha256",
        "quarantine_policy_version",
        "training_scope",
        "validation_fold",
        "training_wells",
        "training_groups",
        "training_rows",
        "training_well_ids_sha256",
        "training_group_ids_sha256",
        "lightgbm_version",
        "model_sha256",
    }
)


@dataclass(frozen=True)
class SelectionLimits:
    max_groups: int
    max_wells: int
    max_prediction_rows: int

    def validate(self) -> None:
        for name, value in (
            ("max_groups", self.max_groups),
            ("max_wells", self.max_wells),
            ("max_prediction_rows", self.max_prediction_rows),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


def validate_baseline_b_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Reject parameter drift, GPU use, sampling, and non-deterministic settings."""
    if not isinstance(parameters, dict) or set(parameters) != BASELINE_B_PARAMETER_KEYS:
        raise ValueError(
            "Baseline B parameter keys mismatch: "
            f"expected={sorted(BASELINE_B_PARAMETER_KEYS)}"
        )
    if parameters["objective"] != "regression" or parameters["metric"] != "l2":
        raise ValueError("Baseline B must use regression with l2 metric")
    if parameters["device_type"] != "cpu":
        raise ValueError("Baseline B device_type must be cpu")
    if parameters["deterministic"] is not True:
        raise ValueError("Baseline B deterministic must be true")
    if parameters["force_col_wise"] is not True:
        raise ValueError("Baseline B force_col_wise must be true")
    if parameters["feature_fraction"] != 1.0:
        raise ValueError("Baseline B feature_fraction must be 1.0")
    if parameters["bagging_fraction"] != 1.0 or parameters["bagging_freq"] != 0:
        raise ValueError("Baseline B bagging must be disabled")
    if parameters["verbosity"] != -1:
        raise ValueError("Baseline B verbosity must be -1")

    integer_minimums = {
        "num_leaves": 2,
        "min_data_in_leaf": 1,
        "num_threads": 1,
        "seed": 0,
    }
    for name, minimum in integer_minimums.items():
        value = parameters[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"Baseline B {name} must be an integer >= {minimum}")
    learning_rate = parameters["learning_rate"]
    if (
        not isinstance(learning_rate, (int, float))
        or isinstance(learning_rate, bool)
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("Baseline B learning_rate must be finite and positive")
    return dict(parameters)


def _validate_features(features: pd.DataFrame) -> None:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if tuple(features.columns) not in (
        BASELINE_B_FEATURE_COLUMNS,
        LAST_KNOWN_SLOPE_FEATURE_COLUMNS,
        TYPEWELL_GR_SLOPE_FEATURE_COLUMNS,
        TYPEWELL_PRIOR_FEATURE_COLUMNS,
    ):
        raise ValueError("Baseline B feature columns or order mismatch")
    if features.empty:
        raise ValueError("Baseline B features must not be empty")
    values = features.to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError("Baseline B features contain infinite values")
    nullable = {"gr", "gr_delta_anchor"}
    for column in set(features.columns) - nullable:
        if features[column].isna().any():
            raise ValueError(f"Baseline B feature {column} contains NaN")


def residual_target(truth: Iterable[float], features: pd.DataFrame) -> np.ndarray:
    """Return TVT residuals relative to each row's last-known prefix anchor."""
    _validate_features(features)
    values = np.asarray(truth, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(features):
        raise ValueError("truth must be one-dimensional and align with features")
    if not np.isfinite(values).all():
        raise ValueError("truth contains NaN or infinite values")
    residuals = values - features["anchor_tvt_input"].to_numpy(dtype=np.float64)
    if not np.isfinite(residuals).all():
        raise ValueError("Baseline B residual target is not finite")
    return residuals


def reconstruct_tvt(
    residual_predictions: Iterable[float], features: pd.DataFrame
) -> np.ndarray:
    """Add the per-row anchor exactly once to residual predictions."""
    _validate_features(features)
    residuals = np.asarray(residual_predictions, dtype=np.float64)
    if residuals.ndim != 1 or len(residuals) != len(features):
        raise ValueError("residual predictions must align with features")
    predictions = residuals + features["anchor_tvt_input"].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(predictions).all():
        raise ValueError("Baseline B predictions contain NaN or infinite values")
    return predictions


def train_baseline_b(
    features: pd.DataFrame,
    residuals: Iterable[float],
    *,
    parameters: dict[str, Any],
    num_boost_round: int,
) -> lgb.Booster:
    """Fit one LightGBM model using only the explicitly supplied training rows."""
    _validate_features(features)
    target = np.asarray(residuals, dtype=np.float64)
    if target.ndim != 1 or len(target) != len(features):
        raise ValueError("residual target must align with training features")
    if not np.isfinite(target).all():
        raise ValueError("residual target contains NaN or infinite values")
    if not isinstance(num_boost_round, int) or isinstance(num_boost_round, bool):
        raise ValueError("num_boost_round must be a positive integer")
    if num_boost_round <= 0:
        raise ValueError("num_boost_round must be a positive integer")
    validated_parameters = validate_baseline_b_parameters(parameters)
    dataset = lgb.Dataset(
        features,
        label=target,
        feature_name=list(features.columns),
        free_raw_data=True,
    )
    return lgb.train(
        validated_parameters,
        dataset,
        num_boost_round=num_boost_round,
        callbacks=[lgb.log_evaluation(period=0)],
    )


def predict_baseline_b(booster: lgb.Booster, features: pd.DataFrame) -> np.ndarray:
    _validate_features(features)
    if booster.feature_name() != list(features.columns):
        raise ValueError("Baseline B booster feature names mismatch")
    residuals = booster.predict(
        features,
        num_iteration=booster.current_iteration(),
        validate_features=True,
    )
    return reconstruct_tvt(residuals, features)


def identifier_sha256(values: Iterable[object]) -> str:
    normalized = sorted({str(value).lower() for value in values})
    if not normalized:
        raise ValueError("identifier collection must not be empty")
    return hashlib.sha256(("\n".join(normalized) + "\n").encode("utf-8")).hexdigest()


def _ordered_groups(pool: pd.DataFrame, seed: int) -> list[pd.DataFrame]:
    by_fold: dict[int, list[tuple[str, pd.DataFrame]]] = {}
    for group_id, group in pool.groupby("typewell_group", sort=False):
        folds = group["fold"].unique().tolist()
        if len(folds) != 1:
            raise ValueError("a typewell group crosses folds")
        key = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()
        by_fold.setdefault(int(folds[0]), []).append(
            (key, group.sort_values("well_id").copy())
        )
    for groups in by_fold.values():
        groups.sort(key=lambda item: item[0])

    ordered: list[pd.DataFrame] = []
    maximum = max((len(groups) for groups in by_fold.values()), default=0)
    for position in range(maximum):
        for fold in sorted(by_fold):
            groups = by_fold[fold]
            if position < len(groups):
                ordered.append(groups[position][1])
    return ordered


def _bounded_group_selection(
    pool: pd.DataFrame,
    *,
    limits: SelectionLimits,
    seed: int,
) -> pd.DataFrame:
    limits.validate()
    selected: list[pd.DataFrame] = []
    wells = 0
    prediction_rows = 0
    for group in _ordered_groups(pool, seed):
        if len(selected) >= limits.max_groups:
            break
        group_wells = len(group)
        group_rows = int(group["prediction_rows"].sum())
        if wells + group_wells > limits.max_wells:
            continue
        if prediction_rows + group_rows > limits.max_prediction_rows:
            continue
        selected.append(group)
        wells += group_wells
        prediction_rows += group_rows
    if not selected:
        raise ValueError("selection limits exclude every complete typewell group")
    return pd.concat(selected, ignore_index=True).sort_values("well_id").reset_index(
        drop=True
    )


def build_bounded_fold_split(
    mapping: pd.DataFrame,
    *,
    validation_fold: int,
    train_limits: SelectionLimits,
    validation_limits: SelectionLimits,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select whole groups for one bounded fold without observing targets."""
    if not isinstance(mapping, pd.DataFrame):
        raise TypeError("mapping must be a pandas DataFrame")
    if tuple(mapping.columns) != FOLD_COLUMNS:
        raise ValueError("fold mapping columns mismatch")
    if mapping.isna().any().any():
        raise ValueError("fold mapping contains missing values")
    assert_no_public_sample_overlap(mapping["well_id"], context="Baseline B mapping")
    if mapping["well_id"].duplicated().any():
        raise ValueError("fold mapping contains duplicate wells")
    if mapping.groupby("typewell_group")["fold"].nunique().gt(1).any():
        raise ValueError("a typewell group crosses folds")
    if validation_fold not in set(mapping["fold"]):
        raise ValueError(f"validation fold is absent: {validation_fold}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    train_pool = mapping.loc[mapping["fold"] != validation_fold].copy()
    validation_pool = mapping.loc[mapping["fold"] == validation_fold].copy()
    train = _bounded_group_selection(train_pool, limits=train_limits, seed=seed)
    validation = _bounded_group_selection(
        validation_pool, limits=validation_limits, seed=seed
    )

    expected_training_folds = set(train_pool["fold"])
    if set(train["fold"]) != expected_training_folds:
        raise ValueError("training limits must retain at least one group from every fold")
    if set(validation["fold"]) != {validation_fold}:
        raise ValueError("validation selection crossed the requested fold")
    if set(train["well_id"]) & set(validation["well_id"]):
        raise ValueError("training and validation wells overlap")
    if set(train["typewell_group"]) & set(validation["typewell_group"]):
        raise ValueError("training and validation typewell groups overlap")
    assert_no_public_sample_overlap(train["well_id"], context="Baseline B training")
    assert_no_public_sample_overlap(
        validation["well_id"], context="Baseline B validation"
    )
    return train, validation


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    expected_fold_mapping_sha256: str,
    expected_parameters: dict[str, Any],
    expected_validation_fold: int | None,
    expected_feature_columns: tuple[str, ...] = BASELINE_B_FEATURE_COLUMNS,
) -> None:
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise ValueError("Baseline B manifest keys mismatch")
    if manifest["artifact_type"] != BASELINE_B_ARTIFACT_TYPE:
        raise ValueError("Baseline B artifact type mismatch")
    if manifest["schema_version"] != BASELINE_B_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Baseline B artifact schema mismatch")
    if manifest["method"] != BASELINE_B_METHOD:
        raise ValueError("Baseline B method mismatch")
    if manifest["target_definition"] != BASELINE_B_TARGET:
        raise ValueError("Baseline B target definition mismatch")
    if tuple(manifest["feature_columns"]) != expected_feature_columns:
        raise ValueError("Baseline B manifest feature columns mismatch")
    if manifest["parameters"] != validate_baseline_b_parameters(expected_parameters):
        raise ValueError("Baseline B manifest parameters mismatch")
    if manifest["fold_mapping_sha256"] != expected_fold_mapping_sha256:
        raise ValueError("Baseline B fold SHA256 mismatch")
    if manifest["quarantine_policy_version"] != QUARANTINE_POLICY_VERSION:
        raise ValueError("Baseline B quarantine policy mismatch")
    expected_scope = (
        "all_effective_wells" if expected_validation_fold is None else "cv_fold"
    )
    if manifest["training_scope"] != expected_scope:
        raise ValueError("Baseline B training scope mismatch")
    if manifest["validation_fold"] != expected_validation_fold:
        raise ValueError("Baseline B validation fold mismatch")
    if not isinstance(manifest["num_boost_round"], int) or manifest[
        "num_boost_round"
    ] <= 0:
        raise ValueError("Baseline B manifest boost rounds are invalid")
    for name in ("training_wells", "training_groups", "training_rows"):
        if not isinstance(manifest[name], int) or manifest[name] <= 0:
            raise ValueError(f"Baseline B manifest {name} is invalid")
    for name in (
        "fold_mapping_sha256",
        "training_well_ids_sha256",
        "training_group_ids_sha256",
        "model_sha256",
    ):
        if not _is_sha256(manifest[name]):
            raise ValueError(f"Baseline B manifest {name} is invalid")
    if not isinstance(manifest["lightgbm_version"], str) or not manifest[
        "lightgbm_version"
    ]:
        raise ValueError("Baseline B manifest LightGBM version is invalid")


def save_baseline_b_artifact(
    booster: lgb.Booster,
    artifact_dir: str | Path,
    *,
    parameters: dict[str, Any],
    num_boost_round: int,
    fold_mapping_sha256: str,
    validation_fold: int | None,
    training_mapping: pd.DataFrame,
    training_rows: int,
) -> tuple[Path, Path]:
    """Save the native model plus a strict provenance manifest."""
    validated_parameters = validate_baseline_b_parameters(parameters)
    assert_no_public_sample_overlap(
        training_mapping["well_id"], context="Baseline B artifact training"
    )
    if booster.current_iteration() != num_boost_round:
        raise ValueError("Baseline B booster iteration count mismatch")
    feature_columns = tuple(booster.feature_name())
    if feature_columns not in (
        BASELINE_B_FEATURE_COLUMNS,
        LAST_KNOWN_SLOPE_FEATURE_COLUMNS,
        TYPEWELL_GR_SLOPE_FEATURE_COLUMNS,
        TYPEWELL_PRIOR_FEATURE_COLUMNS,
    ):
        raise ValueError("Baseline B artifact feature columns mismatch")
    destination = Path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "model.txt"
    manifest_path = destination / "manifest.json"
    booster.save_model(str(model_path), num_iteration=booster.current_iteration())
    manifest = {
        "artifact_type": BASELINE_B_ARTIFACT_TYPE,
        "schema_version": BASELINE_B_ARTIFACT_SCHEMA_VERSION,
        "method": BASELINE_B_METHOD,
        "target_definition": BASELINE_B_TARGET,
        "feature_columns": list(feature_columns),
        "parameters": validated_parameters,
        "num_boost_round": num_boost_round,
        "fold_mapping_sha256": fold_mapping_sha256,
        "quarantine_policy_version": QUARANTINE_POLICY_VERSION,
        "training_scope": (
            "all_effective_wells" if validation_fold is None else "cv_fold"
        ),
        "validation_fold": validation_fold,
        "training_wells": int(len(training_mapping)),
        "training_groups": int(training_mapping["typewell_group"].nunique()),
        "training_rows": int(training_rows),
        "training_well_ids_sha256": identifier_sha256(
            training_mapping["well_id"]
        ),
        "training_group_ids_sha256": identifier_sha256(
            training_mapping["typewell_group"]
        ),
        "lightgbm_version": lgb.__version__,
        "model_sha256": sha256_file(model_path),
    }
    _validate_manifest(
        manifest,
        expected_fold_mapping_sha256=fold_mapping_sha256,
        expected_parameters=validated_parameters,
        expected_validation_fold=validation_fold,
        expected_feature_columns=feature_columns,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return model_path, manifest_path


def load_baseline_b_artifact(
    artifact_dir: str | Path,
    *,
    expected_fold_mapping_sha256: str,
    expected_parameters: dict[str, Any],
    expected_validation_fold: int | None,
    expected_feature_columns: tuple[str, ...] = BASELINE_B_FEATURE_COLUMNS,
) -> tuple[lgb.Booster, dict[str, Any]]:
    source = Path(artifact_dir)
    model_path = source / "model.txt"
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(
        manifest,
        expected_fold_mapping_sha256=expected_fold_mapping_sha256,
        expected_parameters=expected_parameters,
        expected_validation_fold=expected_validation_fold,
        expected_feature_columns=expected_feature_columns,
    )
    if sha256_file(model_path) != manifest["model_sha256"]:
        raise ValueError("Baseline B model SHA256 mismatch")
    booster = lgb.Booster(model_file=str(model_path))
    if booster.feature_name() != list(expected_feature_columns):
        raise ValueError("Baseline B saved feature names mismatch")
    if booster.current_iteration() != manifest["num_boost_round"]:
        raise ValueError("Baseline B saved iteration count mismatch")
    return booster, manifest
