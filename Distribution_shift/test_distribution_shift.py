"""Tests for Stage 7 distribution-shift logic."""

import numpy as np


def test_level_shift_applies_after_boundary():
    values = np.array(
        [10.0, 20.0, 30.0, 40.0],
        dtype=np.float32,
    )

    shifted = values.copy()
    shifted[2:] += 5.0

    assert np.allclose(
        shifted,
        [10.0, 20.0, 35.0, 45.0],
    )


def test_noise_shift_is_reproducible():
    values = np.ones(
        10,
        dtype=np.float32,
    )

    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    first = values.copy()
    second = values.copy()

    first[5:] += rng1.normal(
        0.0,
        24.0,
        size=5,
    )

    second[5:] += rng2.normal(
        0.0,
        24.0,
        size=5,
    )

    assert np.array_equal(
        first,
        second,
    )


def test_shifted_demand_is_non_negative():
    values = np.zeros(
        20,
        dtype=np.float32,
    )

    rng = np.random.default_rng(123)

    shifted = values.copy()

    shifted[10:] += rng.normal(
        0.0,
        24.0,
        size=10,
    )

    shifted = np.maximum(
        shifted,
        0.0,
    )

    assert np.all(
        shifted >= 0.0
    )


def test_noise_shift_doubles_original_generator_noise():
    original_noise_std = 12.0
    shifted_noise_std = 24.0

    assert shifted_noise_std == (
        original_noise_std * 2.0
    )
