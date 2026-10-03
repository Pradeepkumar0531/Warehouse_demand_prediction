"""Synthetic warehouse demand generator."""

import sys
from pathlib import Path

# Add project root (Ware_house_project) to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

BASELINE = 300.0
TREND_PER_HOUR = 0.005
DAILY_SIN_AMPLITUDE = 45.0
DAILY_COS_AMPLITUDE = 20.0
NOISE_STD = 12.0

SPIKE_PROBABILITY = 0.005
SPIKE_MIN = 100.0
SPIKE_MAX = 220.0

DROP_PROBABILITY = 0.003
DROP_MIN = 80.0
DROP_MAX = 140.0

WEEKLY_EFFECTS = {
    0: 10.0,    # Monday
    1: 15.0,    # Tuesday
    2: 12.0,    # Wednesday
    3: 18.0,    # Thursday
    4: 25.0,    # Friday
    5: -15.0,   # Saturday
    6: -25.0,   # Sunday
}


def generate_warehouse_data(num_days=365, seed=42):
    """Generate hourly synthetic warehouse demand."""
    if num_days <= 0:
        raise ValueError("num_days must be positive")

    rng = np.random.default_rng(seed)
    num_hours = num_days * 24

    timestamps = pd.date_range(
        start="2025-01-01",
        periods=num_hours,
        freq="h",
    )

    t = np.arange(num_hours, dtype=float)
    hour = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()

    baseline = np.full(num_hours, BASELINE)
    trend = TREND_PER_HOUR * t

    daily_seasonality = (
        DAILY_SIN_AMPLITUDE * np.sin(2 * np.pi * hour / 24)
        + DAILY_COS_AMPLITUDE * np.cos(2 * np.pi * hour / 24)
    )

    weekly_effect = np.array([WEEKLY_EFFECTS[d] for d in day_of_week])

    event_draw = rng.random(num_hours)
    event = np.zeros(num_hours)
    event_type = np.full(num_hours, "none", dtype=object)

    spike_mask = event_draw < SPIKE_PROBABILITY
    drop_mask = (
        (event_draw >= SPIKE_PROBABILITY)
        & (event_draw < SPIKE_PROBABILITY + DROP_PROBABILITY)
    )

    event[spike_mask] = rng.uniform(
        SPIKE_MIN, SPIKE_MAX, spike_mask.sum()
    )
    event_type[spike_mask] = "spike"

    event[drop_mask] = -rng.uniform(
        DROP_MIN, DROP_MAX, drop_mask.sum()
    )
    event_type[drop_mask] = "drop"

    noise = rng.normal(0.0, NOISE_STD, num_hours)

    demand = np.maximum(
        0.0,
        baseline
        + trend
        + daily_seasonality
        + weekly_effect
        + event
        + noise,
    )

    return pd.DataFrame({
        "timestamp": timestamps,
        "hour": hour,
        "day_of_week": day_of_week,
        "baseline": baseline,
        "trend": trend,
        "daily_seasonality": daily_seasonality,
        "weekly_effect": weekly_effect,
        "event": event,
        "event_type": event_type,
        "noise": noise,
        "demand": demand,
    })



if __name__ == "__main__":
    output_path = "data/warehouse_demand1.csv"

    data = generate_warehouse_data(num_days=365, seed=42)
    data.to_csv(output_path, index=False)

    print(f"Generated {len(data)} hourly observations.")
    print(f"Saved to: {output_path}")
    print(f"Demand mean: {data['demand'].mean():.2f}")
    print(f"Demand min: {data['demand'].min():.2f}")
    print(f"Demand max: {data['demand'].max():.2f}")
    print(f"Spikes: {(data['event_type'] == 'spike').sum()}")
    print(f"Drops: {(data['event_type'] == 'drop').sum()}")
