"""Tests for Stage 9 reproducibility helpers."""

import numpy as np

from reproducibility import (
    compare_generated_data,
    environment_summary,
)


def test_hash_array_is_stable():
    from reproducibility import hash_array

    values = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float32,
    )

    first = hash_array(values)
    second = hash_array(values.copy())

    assert first == second


def test_same_seed_generates_same_environment():
    result = compare_generated_data()

    assert result["same"]
    assert result["first_hash"] == result["second_hash"]


def test_environment_contains_seed():
    environment = environment_summary()

    assert environment["seed"] == 42
    assert environment["python"]
    assert environment["numpy"]
    assert environment["torch"]


def test_numpy_array_bytes_are_order_stable():
    from reproducibility import hash_array

    values = np.arange(
        12,
        dtype=np.float32,
    ).reshape(3, 4)

    first = hash_array(values)
    second = hash_array(values.copy())

    assert first == second
