import numpy as np
import pandas as pd
import pytest

from rogii.metric import mean_squared_error, root_mean_squared_error


def test_perfect_prediction_is_zero_and_python_float() -> None:
    score = mean_squared_error(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert score == 0.0
    assert type(score) is float


def test_manual_example_and_pandas_series() -> None:
    assert mean_squared_error(pd.Series([1.0, 3.0]), pd.Series([2.0, 5.0])) == 2.5
    assert root_mean_squared_error(pd.Series([1.0, 3.0]), pd.Series([2.0, 5.0])) == np.sqrt(2.5)


def test_length_mismatch_does_not_broadcast() -> None:
    with pytest.raises(ValueError, match="same shape"):
        mean_squared_error([1.0, 2.0], [1.0])


@pytest.mark.parametrize(
    ("truth", "prediction"),
    [([1.0, np.nan], [1.0, 2.0]), ([1.0, 2.0], [1.0, np.inf])],
)
def test_nan_and_infinity_raise(truth, prediction) -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        mean_squared_error(truth, prediction)


@pytest.mark.parametrize(
    ("truth", "prediction"),
    [([[1.0], [2.0]], [[1.0], [2.0]]), (1.0, 1.0), ([], [])],
)
def test_wrong_shape_or_empty_raises(truth, prediction) -> None:
    with pytest.raises(ValueError):
        mean_squared_error(truth, prediction)


def test_row_order_is_positional_and_changes_score() -> None:
    truth = pd.Series([10.0, 20.0], index=["id_a", "id_b"])
    aligned = pd.Series([10.0, 20.0], index=["id_a", "id_b"])
    reordered_values = pd.Series([20.0, 10.0], index=["id_b", "id_a"])
    assert mean_squared_error(truth, aligned) == 0.0
    assert mean_squared_error(truth, reordered_values) == 100.0


def test_rmse_is_square_root_of_mse_and_perfect_is_zero() -> None:
    truth = np.array([0.0, 2.0, 4.0])
    prediction = np.array([1.0, 1.0, 5.0])
    assert root_mean_squared_error(truth, truth) == 0.0
    assert root_mean_squared_error(truth, prediction) ** 2 == pytest.approx(
        mean_squared_error(truth, prediction)
    )
