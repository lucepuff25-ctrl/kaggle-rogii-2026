"""ROGII competition foundations: validation splits and official metric."""

from .cv import DEFAULT_SEED, assign_group_folds, validate_group_folds
from .metric import mean_squared_error

__all__ = ["DEFAULT_SEED", "assign_group_folds", "validate_group_folds", "mean_squared_error"]
