"""Tests for Stage 8 failure-investigation helpers."""

import numpy as np

from failure_investigation import (
    average_final_query_attention,
    top_error_rows,
)


def test_top_error_rows_are_sorted_descending():
    test = {
        "X": np.array(
            [
                [1.0, 2.0],
                [4.0, 5.0],
                [8.0, 9.0],
            ],
            dtype=np.float32,
        ),
        "y": np.array(
            [2.0, 10.0, 7.0],
            dtype=np.float32,
        ),
        "target_index": np.array(
            [2, 3, 4],
            dtype=np.int64,
        ),
        "target_timestamp": np.array(
            [
                "2025-01-01 02:00:00",
                "2025-01-01 03:00:00",
                "2025-01-01 04:00:00",
            ]
        ),
    }

    prediction = np.array(
        [1.0, 5.0, 9.0],
        dtype=np.float32,
    )

    rows, errors = top_error_rows(
        test,
        prediction,
        count=3,
    )

    assert np.all(
        errors >= 0.0
    )

    assert rows[0]["absolute_error"] >= rows[1]["absolute_error"]


def test_target_jump_is_actual_minus_previous():
    test = {
        "X": np.array(
            [
                [4.0, 5.0],
            ],
            dtype=np.float32,
        ),
        "y": np.array(
            [9.0],
            dtype=np.float32,
        ),
        "target_index": np.array(
            [2],
            dtype=np.int64,
        ),
        "target_timestamp": np.array(
            ["2025-01-01 02:00:00"]
        ),
    }

    prediction = np.array(
        [6.0],
        dtype=np.float32,
    )

    rows, _ = top_error_rows(
        test,
        prediction,
        count=1,
    )

    assert np.isclose(
        rows[0]["target_jump"],
        4.0,
    )


def test_average_attention_uses_final_query_and_24_lags():
    weights = np.zeros(
        (2, 24, 24),
        dtype=np.float64,
    )

    weights[0, -1, :] = np.arange(
        1.0,
        25.0,
    )

    weights[1, -1, :] = np.arange(
        24.0,
        0.0,
        -1.0,
    )

    weights[:, -1, :] /= weights[:, -1, :].sum(
        axis=-1,
        keepdims=True,
    )

    lag_hours, average = (
        average_final_query_attention(
            weights
        )
    )

    assert len(lag_hours) == 24
    assert np.array_equal(
        lag_hours,
        np.arange(
            23,
            -1,
            -1,
        ),
    )

    assert np.isclose(
        average.sum(),
        1.0,
    )

    assert np.isclose(
        average[0],
        0.5 * (
            (1.0 / 300.0)
            + (24.0 / 300.0)
        ),
    )
