"""Prepare 24-hour windows for next-hour demand forecasting."""

import sys
from pathlib import Path

# Add project root (Ware_house_project) to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


WINDOW_SIZE = 24


def make_windows(data, window_size=WINDOW_SIZE):
    """Turn a continuous demand series into 24-hour inputs and next-hour targets."""
    demand = data["demand"].to_numpy(dtype=np.float32)

    if len(demand) <= window_size:
        raise ValueError("Not enough observations for the requested window size")

    inputs = []
    targets = []
    target_indices = []

    for target_index in range(window_size, len(demand)):
        start = target_index - window_size
        inputs.append(demand[start:target_index])
        targets.append(demand[target_index])
        target_indices.append(target_index)

    return (
        np.asarray(inputs, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(target_indices, dtype=np.int64),
    )


def chronological_split(
    inputs,
    targets,
    target_indices,
    timestamps,
    train_fraction=0.70,
    validation_fraction=0.15,
):
    """Split samples in time order using the target timestamp."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must be less than 1")

    sample_count = len(targets)
    train_end = int(sample_count * train_fraction)
    validation_end = int(sample_count * (train_fraction + validation_fraction))

    splits = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, sample_count),
    }

    result = {}

    for name, (start, end) in splits.items():
        indices = target_indices[start:end]

        result[name] = {
            "X": inputs[start:end],
            "y": targets[start:end],
            "target_index": indices,
            "target_timestamp": pd.Series(
                pd.to_datetime(timestamps.iloc[indices].to_numpy())
            ),
        }

    return result



if __name__ == "__main__":
    raw_path = "./data/warehouse_demand1.csv"

    data = pd.read_csv(raw_path, parse_dates=["timestamp"])

    X, y, target_indices = make_windows(data)
    splits = chronological_split(
        X,
        y,
        target_indices,
        data["timestamp"],
    )

    for name, split in splits.items():
        np.savez(
            f"{name}_data.npz",
            X=split["X"],
            y=split["y"],
            target_index=split["target_index"],
            target_timestamp=split["target_timestamp"].astype(str).to_numpy(),
        )
        print(
            f"{name}: {len(split['y'])} samples "
            f"-> {name}_data.npz"
        )
