"""Frozen formal-run contract for ROGII Baseline B."""

from __future__ import annotations

from typing import Any

from .baseline_b import validate_baseline_b_parameters

BASELINE_B_FIXED_SEED = 20260713
BASELINE_B_FIXED_NUM_BOOST_ROUND = 64
BASELINE_B_FIXED_PARAMETERS: dict[str, Any] = {
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "deterministic": True,
    "device_type": "cpu",
    "feature_fraction": 1.0,
    "force_col_wise": True,
    "learning_rate": 0.05,
    "metric": "l2",
    "min_data_in_leaf": 100,
    "num_leaves": 31,
    "num_threads": 4,
    "objective": "regression",
    "seed": BASELINE_B_FIXED_SEED,
    "verbosity": -1,
}


def validate_full_run_contract(
    *,
    parameters: dict[str, Any],
    num_boost_round: int,
    seed: int,
) -> None:
    validated = validate_baseline_b_parameters(parameters)
    if validated != BASELINE_B_FIXED_PARAMETERS:
        raise ValueError("formal Baseline B parameters differ from the frozen smoke contract")
    if num_boost_round != BASELINE_B_FIXED_NUM_BOOST_ROUND:
        raise ValueError("formal Baseline B num_boost_round must remain 64")
    if seed != BASELINE_B_FIXED_SEED:
        raise ValueError("formal Baseline B seed must remain 20260713")
