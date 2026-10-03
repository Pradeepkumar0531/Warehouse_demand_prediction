"""Stage 8 - Failure investigation for warehouse forecasting."""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from Warehouse_forecasting.warehouse_forecasting import (
    WarehouseAttentionModel,
    load_split,
    predict,
)


OUTPUT_DIR = Path(__file__).resolve().parent

TEST_PATH = (
    PROJECT_ROOT
    / "Data_Generation"
    / "test_data.npz"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "Warehouse_forecasting"
    / "warehouse_attention_model.pt"
)

TOP_ERRORS_PATH = (
    OUTPUT_DIR
    / "top_forecast_errors.csv"
)

ATTENTION_PATH = (
    OUTPUT_DIR
    / "average_attention_by_lag.csv"
)

ERROR_PLOT_PATH = (
    OUTPUT_DIR
    / "forecast_error_analysis.png"
)

ATTENTION_PLOT_PATH = (
    OUTPUT_DIR
    / "average_attention_by_lag.png"
)


def load_stage6_model():
    """Load the fixed model selected during Stage 6."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Stage 6 model checkpoint was not found:\n"
            f"{MODEL_PATH}\n\n"
            "Run this first:\n"
            "python Warehouse_forecasting/"
            "warehouse_forecasting.py"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model = WarehouseAttentionModel(
        d_model=4
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    return (
        model,
        float(checkpoint["train_mean"]),
        float(checkpoint["train_std"]),
    )


def top_error_rows(
    test,
    prediction,
    count=20,
):
    """Build a table for the largest absolute forecast errors."""
    absolute_error = np.abs(
        prediction - test["y"]
    )

    order = np.argsort(
        absolute_error
    )[::-1]

    rows = []

    for rank, local_index in enumerate(
        order[:count],
        start=1,
    ):
        previous = float(
            test["X"][
                local_index,
                -1
            ]
        )

        actual = float(
            test["y"][
                local_index
            ]
        )

        predicted = float(
            prediction[
                local_index
            ]
        )

        target_jump = (
            actual - previous
        )

        rows.append(
            {
                "rank": rank,
                "target_timestamp": str(
                    test["target_timestamp"][
                        local_index
                    ]
                ),
                "target_index": int(
                    test["target_index"][
                        local_index
                    ]
                ),
                "previous_demand": previous,
                "actual": actual,
                "prediction": predicted,
                "absolute_error": float(
                    absolute_error[
                        local_index
                    ]
                ),
                "target_jump": float(
                    target_jump
                ),
            }
        )

    return rows, absolute_error


def average_final_query_attention(
    attention_weights
):
    """Average the final query's 24 attention weights."""
    final_query_weights = (
        attention_weights[:, -1, :]
    )

    average_attention = (
        final_query_weights.mean(
            axis=0
        )
    )

    # Input position 23 is the most recent observed hour.
    # Therefore position 0 is 23 hours before the target.
    lag_hours = np.arange(
        23,
        -1,
        -1,
    )

    return (
        lag_hours,
        average_attention,
    )


def main():
    if not TEST_PATH.exists():
        raise FileNotFoundError(
            "Stage 2 test data was not found:\n"
            f"{TEST_PATH}"
        )

    test = load_split(
        TEST_PATH
    )

    model, mean, std = load_stage6_model()

    prediction, attention_weights = predict(
        model,
        test["X"],
        mean,
        std,
    )

    rows, absolute_error = top_error_rows(
        test,
        prediction,
        count=20,
    )

    with TOP_ERRORS_PATH.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    lag_hours, average_attention = (
        average_final_query_attention(
            attention_weights
        )
    )

    attention_rows = []

    for lag, value in zip(
        lag_hours,
        average_attention,
    ):
        attention_rows.append(
            {
                "lag_hours": int(lag),
                "average_attention_weight": float(
                    value
                ),
            }
        )

    with ATTENTION_PATH.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "lag_hours",
                "average_attention_weight",
            ],
        )

        writer.writeheader()
        writer.writerows(
            attention_rows
        )

    # Plot the ten largest-error forecasts.
    sample = rows[:10]

    x = np.arange(
        len(sample)
    )

    actual = [
        row["actual"]
        for row in sample
    ]

    predicted = [
        row["prediction"]
        for row in sample
    ]

    labels = [
        row["target_timestamp"][
            5:16
        ]
        for row in sample
    ]

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        x,
        actual,
        marker="o",
        label="Actual",
    )

    plt.plot(
        x,
        predicted,
        marker="o",
        label="Prediction",
    )

    plt.xticks(
        x,
        labels,
        rotation=45,
        ha="right",
    )

    plt.xlabel(
        "Highest-error test cases"
    )
    plt.ylabel(
        "Demand"
    )
    plt.title(
        "Highest-Error Attention Forecasts"
    )
    plt.legend()
    plt.grid(
        True,
        alpha=0.3,
    )
    plt.tight_layout()

    plt.savefig(
        ERROR_PLOT_PATH,
        dpi=160,
    )

    plt.close()

    # Plot average attention by input lag.
    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        lag_hours,
        average_attention,
        marker="o",
    )

    plt.xlabel(
        "Hours before target"
    )
    plt.ylabel(
        "Average attention weight"
    )
    plt.title(
        "Average Attention by Lag"
    )
    plt.grid(
        True,
        alpha=0.3,
    )
    plt.tight_layout()

    plt.savefig(
        ATTENTION_PLOT_PATH,
        dpi=160,
    )

    plt.close()

    print(
        "Stage 8 - Failure Investigation"
    )
    print(
        f"Test forecasts: {len(test['y'])}"
    )
    print(
        "Mean absolute error: "
        f"{absolute_error.mean():.4f}"
    )
    print(
        "Largest absolute error: "
        f"{absolute_error.max():.4f}"
    )

    jumps = [
        row["target_jump"]
        for row in rows
    ]

    print(
        "Mean target jump among top 20 errors: "
        f"{np.mean(jumps):.4f}"
    )

    print()
    print("Top five errors:")

    for row in rows[:5]:
        print(
            f"{row['rank']}. "
            f"{row['target_timestamp']} | "
            f"actual={row['actual']:.2f} | "
            f"prediction={row['prediction']:.2f} | "
            f"error={row['absolute_error']:.2f} | "
            f"jump={row['target_jump']:.2f}"
        )

    print()
    print(
        f"Top errors: {TOP_ERRORS_PATH}"
    )
    print(
        f"Attention summary: {ATTENTION_PATH}"
    )
    print(
        f"Error plot: {ERROR_PLOT_PATH}"
    )
    print(
        f"Attention plot: {ATTENTION_PLOT_PATH}"
    )


if __name__ == "__main__":
    main()
