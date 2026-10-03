"""Stage 7 - Distribution-shift stress tests for warehouse forecasting."""

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
    mean24_baseline,
    metrics,
    persistence_baseline,
    predict,
    previous_day_baseline,
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

RESULTS_PATH = (
    OUTPUT_DIR
    / "distribution_shift_results.csv"
)

PLOT_PATH = (
    OUTPUT_DIR
    / "distribution_shift_results.png"
)

LEVEL_SHIFT = 40.0
NOISE_STD = 24.0
NOISE_SEED = 123


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


def reconstruct_test_series(X, y):
    """Reconstruct the observed test sequence."""
    return np.concatenate(
        [
            X[0],
            y,
        ]
    ).astype(np.float32)


def make_windows(series):
    """Create 24-hour input windows for a one-step target."""
    X = []
    y = []

    for index in range(
        24,
        len(series),
    ):
        X.append(
            series[index - 24:index]
        )
        y.append(
            series[index]
        )

    return (
        np.asarray(
            X,
            dtype=np.float32,
        ),
        np.asarray(
            y,
            dtype=np.float32,
        ),
    )


def evaluate_scenario(
    model,
    mean,
    std,
    X,
    y,
):
    """Evaluate all Stage 6 forecasting methods."""
    attention_prediction, _ = predict(
        model,
        X,
        mean,
        std,
    )

    predictions = {
        "Persistence": persistence_baseline(X),
        "Previous-day same-hour": previous_day_baseline(X),
        "24-hour mean": mean24_baseline(X),
        "Attention": attention_prediction,
    }

    results = {}

    for name, prediction in predictions.items():
        results[name] = metrics(
            y,
            prediction,
        )

    return results


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

    original_series = reconstruct_test_series(
        test["X"],
        test["y"],
    )

    # Keep the first 24 observations as pre-shift context.
    # The change begins at the first forecasted target.
    shift_start = 24

    level_shifted = original_series.copy()

    level_shifted[shift_start:] += LEVEL_SHIFT

    rng = np.random.default_rng(
        NOISE_SEED
    )

    noise_shifted = original_series.copy()

    noise_shifted[shift_start:] += rng.normal(
        loc=0.0,
        scale=NOISE_STD,
        size=len(noise_shifted) - shift_start,
    )

    noise_shifted = np.maximum(
        noise_shifted,
        0.0,
    )

    scenarios = {
        "Original test": original_series,
        "Level shift +40": level_shifted,
        "Noise std 24": noise_shifted,
    }

    rows = []

    for scenario_name, series in scenarios.items():
        X_scenario, y_scenario = make_windows(
            series
        )

        results = evaluate_scenario(
            model,
            mean,
            std,
            X_scenario,
            y_scenario,
        )

        for model_name, result in results.items():
            rows.append(
                {
                    "scenario": scenario_name,
                    "model": model_name,
                    "MAE": result["MAE"],
                    "RMSE": result["RMSE"],
                    "MAPE_percent": result["MAPE"],
                }
            )

    with RESULTS_PATH.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "scenario",
                "model",
                "MAE",
                "RMSE",
                "MAPE_percent",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    scenario_order = [
        "Original test",
        "Level shift +40",
        "Noise std 24",
    ]

    model_order = [
        "Persistence",
        "Previous-day same-hour",
        "24-hour mean",
        "Attention",
    ]

    plt.figure(
        figsize=(9, 5)
    )

    for model_name in model_order:
        values = []

        for scenario_name in scenario_order:
            row = next(
                row
                for row in rows
                if row["scenario"] == scenario_name
                and row["model"] == model_name
            )

            values.append(
                row["MAE"]
            )

        plt.plot(
            scenario_order,
            values,
            marker="o",
            label=model_name,
        )

    plt.xlabel("Scenario")
    plt.ylabel("MAE")
    plt.title(
        "Warehouse Distribution-Shift Stress Test"
    )
    plt.legend()
    plt.grid(
        True,
        alpha=0.3,
    )
    plt.tight_layout()

    plt.savefig(
        PLOT_PATH,
        dpi=160,
    )

    plt.close()

    print(
        "Stage 7 - Distribution Shift"
    )
    print(
        f"Level shift: +{LEVEL_SHIFT}"
    )
    print(
        f"Noise standard deviation: {NOISE_STD}"
    )
    print(
        f"Noise seed: {NOISE_SEED}"
    )
    print()

    for row in rows:
        print(
            f"{row['scenario']} | "
            f"{row['model']}: "
            f"MAE={row['MAE']:.4f}, "
            f"RMSE={row['RMSE']:.4f}, "
            f"MAPE={row['MAPE_percent']:.2f}%"
        )

    print()
    print(
        f"Results: {RESULTS_PATH}"
    )
    print(
        f"Plot: {PLOT_PATH}"
    )


if __name__ == "__main__":
    main()
