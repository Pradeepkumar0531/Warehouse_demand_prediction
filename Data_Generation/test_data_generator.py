import numpy as np
import pandas as pd

from data_generator import generate_warehouse_data


def test_expected_length_and_columns():
    data = generate_warehouse_data(365, 42)

    assert len(data) == 365 * 24
    assert data["timestamp"].is_unique

    expected = {
        "timestamp", "hour", "day_of_week", "baseline", "trend",
        "daily_seasonality", "weekly_effect", "event",
        "event_type", "noise", "demand",
    }
    assert set(data.columns) == expected


def test_same_seed_reproduces_data():
    first = generate_warehouse_data(30, 42)
    second = generate_warehouse_data(30, 42)

    pd.testing.assert_frame_equal(first, second)


def test_different_seed_changes_randomness():
    first = generate_warehouse_data(30, 42)
    second = generate_warehouse_data(30, 7)

    assert not np.array_equal(
        first["noise"].to_numpy(),
        second["noise"].to_numpy(),
    )


def test_demand_is_non_negative():
    data = generate_warehouse_data(30, 42)

    assert (data["demand"] >= 0).all()


def test_demand_matches_components():
    data = generate_warehouse_data(30, 42)

    reconstructed = (
        data["baseline"]
        + data["trend"]
        + data["daily_seasonality"]
        + data["weekly_effect"]
        + data["event"]
        + data["noise"]
    )

    expected = np.maximum(0.0, reconstructed)

    np.testing.assert_allclose(
        data["demand"].to_numpy(),
        expected.to_numpy(),
        rtol=0,
        atol=1e-12,
    )


def test_weekly_effects():
    data = generate_warehouse_data(14, 42)

    expected = {
        0: 10.0,
        1: 15.0,
        2: 12.0,
        3: 18.0,
        4: 25.0,
        5: -15.0,
        6: -25.0,
    }

    for day, value in expected.items():
        rows = data.loc[data["day_of_week"] == day, "weekly_effect"]
        assert np.allclose(rows.to_numpy(), value)


def test_event_values_match_labels():
    data = generate_warehouse_data(365, 42)

    spikes = data["event_type"] == "spike"
    drops = data["event_type"] == "drop"
    normal = data["event_type"] == "none"

    assert (data.loc[spikes, "event"] >= 100).all()
    assert (data.loc[spikes, "event"] <= 220).all()

    assert (data.loc[drops, "event"] <= -80).all()
    assert (data.loc[drops, "event"] >= -140).all()

    assert np.allclose(data.loc[normal, "event"], 0.0)

