"""ROGII competition foundations: validation splits and official metric."""

from .cv import DEFAULT_SEED, assign_group_folds, validate_group_folds
from .metric import mean_squared_error
from .quarantine import (
    PUBLIC_SAMPLE_OVERLAP_WELLS,
    assert_no_public_sample_overlap,
    partition_public_sample_overlap,
    public_sample_overlap_mask,
)

__all__ = [
    "DEFAULT_SEED",
    "PUBLIC_SAMPLE_OVERLAP_WELLS",
    "assert_no_public_sample_overlap",
    "assign_group_folds",
    "mean_squared_error",
    "partition_public_sample_overlap",
    "public_sample_overlap_mask",
    "validate_group_folds",
]
