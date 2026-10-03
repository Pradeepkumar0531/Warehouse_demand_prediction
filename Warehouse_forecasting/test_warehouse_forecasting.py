"""Tests for Stage 6 warehouse forecasting."""

import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Attention_mechanism.attention import SelfAttention
from Warehouse_forecasting.warehouse_forecasting import (
    WarehouseAttentionModel,
    fit_standardizer,
    mean24_baseline,
    metrics,
    persistence_baseline,
    previous_day_baseline,
    sinusoidal_position_encoding,
)


def test_persistence_baseline():
    X = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=np.float32,
    )

    prediction = persistence_baseline(X)

    assert np.array_equal(
        prediction,
        [3, 6],
    )


def test_previous_day_baseline():
    X = np.arange(
        48,
        dtype=np.float32,
    ).reshape(2, 24)

    prediction = previous_day_baseline(X)

    assert np.array_equal(
        prediction,
        [0, 24],
    )


def test_mean24_baseline():
    X = np.array(
        [
            [1, 2, 3, 4],
            [4, 4, 4, 4],
        ],
        dtype=np.float32,
    )

    prediction = mean24_baseline(X)

    assert np.allclose(
        prediction,
        [2.5, 4.0],
    )


def test_training_standardizer():
    X = np.array(
        [
            [1, 2],
            [3, 4],
        ],
        dtype=np.float32,
    )

    mean, std = fit_standardizer(X)

    assert np.isclose(
        mean,
        2.5,
    )

    assert np.isclose(
        std,
        np.std(X),
    )


def test_metrics():
    result = metrics(
        np.array(
            [2.0, 4.0]
        ),
        np.array(
            [1.0, 5.0]
        ),
    )

    assert np.isclose(
        result["MAE"],
        1.0,
    )

    assert np.isclose(
        result["RMSE"],
        1.0,
    )

    assert np.isclose(
        result["MAPE"],
        37.5,
    )


def test_position_encoding():
    encoding = sinusoidal_position_encoding(
        24,
        4,
    )

    assert encoding.shape == (
        24,
        4,
    )

    assert torch.isfinite(
        encoding
    ).all()

    assert torch.allclose(
        encoding[0],
        torch.tensor(
            [0.0, 1.0, 0.0, 1.0]
        ),
    )


def test_attention_model_shapes():
    model = WarehouseAttentionModel()

    X = torch.ones(
        3,
        24,
    )

    prediction, weights = model(X)

    assert prediction.shape == (
        3,
    )

    assert weights.shape == (
        3,
        24,
        24,
    )

    assert torch.isfinite(
        prediction
    ).all()

    assert torch.isfinite(
        weights
    ).all()


def test_attention_rows_sum_to_one():
    model = WarehouseAttentionModel()

    X = torch.randn(
        2,
        24,
    )

    _, weights = model(X)

    row_sums = weights.sum(
        dim=-1
    )

    assert torch.allclose(
        row_sums,
        torch.ones_like(
            row_sums
        ),
        atol=1e-6,
    )


def test_model_uses_existing_self_attention():
    model = WarehouseAttentionModel()

    assert isinstance(
        model.attention,
        SelfAttention,
    )
