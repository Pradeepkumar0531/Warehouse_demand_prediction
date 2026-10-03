import numpy as np

from data_generator import generate_warehouse_data
from forecast_data import make_windows, chronological_split


def test_windows_use_previous_24_hours_and_next_hour():
    data = generate_warehouse_data(num_days=3, seed=42)

    X, y, target_indices = make_windows(data)

    assert X.shape == (72 - 24, 24)
    assert y.shape == (72 - 24,)
    assert target_indices[0] == 24

    np.testing.assert_allclose(
        X[0],
        data["demand"].to_numpy()[:24],
    )
    np.testing.assert_allclose(y[0], data["demand"].iloc[24])


def test_target_indices_increase_without_overlap_errors():
    data = generate_warehouse_data(num_days=3, seed=42)

    X, y, target_indices = make_windows(data)

    assert np.all(np.diff(target_indices) == 1)
    assert len(X) == len(y) == len(target_indices)


def test_chronological_split_preserves_time_order():
    data = generate_warehouse_data(num_days=365, seed=42)
    X, y, target_indices = make_windows(data)

    splits = chronological_split(
        X,
        y,
        target_indices,
        data["timestamp"],
    )

    train_last = splits["train"]["target_timestamp"].iloc[-1]
    validation_first = splits["validation"]["target_timestamp"].iloc[0]
    validation_last = splits["validation"]["target_timestamp"].iloc[-1]
    test_first = splits["test"]["target_timestamp"].iloc[0]

    assert train_last < validation_first
    assert validation_last < test_first


def test_split_sizes_add_up():
    data = generate_warehouse_data(num_days=365, seed=42)
    X, y, target_indices = make_windows(data)

    splits = chronological_split(
        X,
        y,
        target_indices,
        data["timestamp"],
    )

    total = sum(len(split["y"]) for split in splits.values())

    assert total == len(y)


def test_validation_boundary_uses_only_past_demand():
    data = generate_warehouse_data(num_days=10, seed=42)
    X, y, target_indices = make_windows(data)

    splits = chronological_split(
        X,
        y,
        target_indices,
        data["timestamp"],
    )

    first_validation_index = splits["validation"]["target_index"][0]
    validation_X = splits["validation"]["X"][0]

    expected_history = data["demand"].to_numpy()[
        first_validation_index - 24:first_validation_index
    ]

    np.testing.assert_allclose(validation_X, expected_history)
