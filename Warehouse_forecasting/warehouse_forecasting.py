
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Attention_mechanism.attention import SelfAttention


DATA_DIR = PROJECT_ROOT / "Data_Generation"
OUTPUT_DIR = Path(__file__).resolve().parent

TRAIN_PATH = DATA_DIR / "train_data.npz"
VALIDATION_PATH = DATA_DIR / "validation_data.npz"
TEST_PATH = DATA_DIR / "test_data.npz"

RESULTS_PATH = OUTPUT_DIR / "warehouse_forecasting_results.csv"
HISTORY_PATH = OUTPUT_DIR / "warehouse_training_history.csv"
PLOT_PATH = OUTPUT_DIR / "warehouse_training_history.png"
MODEL_PATH = OUTPUT_DIR / "warehouse_attention_model.pt"

WINDOW_SIZE = 24
D_MODEL = 4
EPOCHS = 200
LEARNING_RATE = 0.01
SEED = 42


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------

def load_split(path):
    """Load one already-created chronological forecasting split."""
    data = np.load(path, allow_pickle=True)

    return {
        "X": data["X"].astype(np.float32),
        "y": data["y"].astype(np.float32),
        "target_index": data["target_index"].astype(np.int64),
        "target_timestamp": data["target_timestamp"],
    }


# ------------------------------------------------------------
# Baselines and metrics
# ------------------------------------------------------------

def persistence_baseline(X):
    """Predict the next hour with the most recent observed hour."""
    return X[:, -1]


def previous_day_baseline(X):
    """Predict with the same hour from the previous day."""
    return X[:, 0]


def mean24_baseline(X):
    """Predict with the mean of the previous 24 observations."""
    return X.mean(axis=1)


def metrics(y_true, y_pred):
    """Calculate MAE, RMSE, and MAPE."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    error = y_pred - y_true

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    non_zero = np.abs(y_true) > 1e-12

    mape = (
        np.mean(
            np.abs(error[non_zero] / y_true[non_zero])
        )
        * 100.0
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
    }


# ------------------------------------------------------------
# Normalization
# ------------------------------------------------------------

def fit_standardizer(X):
    """Fit normalization using training inputs only."""
    mean = float(X.mean())
    std = float(X.std())

    if std == 0.0:
        raise ValueError(
            "Training standard deviation must be non-zero"
        )

    return mean, std


def standardize(X, mean, std):
    """Apply the training-fitted standardization."""
    return (X - mean) / std


# ------------------------------------------------------------
# Positional encoding
# ------------------------------------------------------------

def sinusoidal_position_encoding(sequence_length, d_model):
    """Create explicit deterministic sine/cosine position features."""
    positions = torch.arange(
        sequence_length,
        dtype=torch.float32,
    ).unsqueeze(1)

    frequencies = torch.exp(
        torch.arange(
            0,
            d_model,
            2,
            dtype=torch.float32,
        )
        * (-math.log(10000.0) / d_model)
    )

    encoding = torch.zeros(
        sequence_length,
        d_model,
        dtype=torch.float32,
    )

    encoding[:, 0::2] = torch.sin(
        positions * frequencies
    )

    encoding[:, 1::2] = torch.cos(
        positions * frequencies
    )

    return encoding


# ------------------------------------------------------------
# Attention forecasting model
# ------------------------------------------------------------

class WarehouseAttentionModel(nn.Module):
    """Small single-head attention model for next-hour prediction."""

    def __init__(
        self,
        d_model=D_MODEL,
        sequence_length=WINDOW_SIZE,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            1,
            d_model,
        )

        self.attention = SelfAttention(
            d_model=d_model,
            scale=True,
        )

        self.output_head = nn.Linear(
            d_model,
            1,
        )

        position_encoding = sinusoidal_position_encoding(
            sequence_length,
            d_model,
        )

        self.register_buffer(
            "position_encoding",
            position_encoding,
        )

    def forward(self, X):
        # X: (batch, 24)
        X = X.unsqueeze(-1)

        # (batch, 24, 1) -> (batch, 24, d_model)
        X = self.input_projection(X)

        # Give each time position explicit identity.
        X = X + self.position_encoding.unsqueeze(0)

        output, weights = self.attention(X)

        # Use the final observed position for the next-hour prediction.
        prediction = self.output_head(
            output[:, -1, :]
        ).squeeze(-1)

        return prediction, weights


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train_attention_model(
    train_X,
    train_y,
    validation_X,
    validation_y,
):
    """Train the model and restore the lowest validation-loss state."""
    torch.manual_seed(SEED)

    train_mean, train_std = fit_standardizer(
        train_X
    )

    X_train = torch.from_numpy(
        standardize(
            train_X,
            train_mean,
            train_std,
        )
    )

    y_train = torch.from_numpy(
        standardize(
            train_y,
            train_mean,
            train_std,
        )
    )

    X_validation = torch.from_numpy(
        standardize(
            validation_X,
            train_mean,
            train_std,
        )
    )

    y_validation = torch.from_numpy(
        standardize(
            validation_y,
            train_mean,
            train_std,
        )
    )

    model = WarehouseAttentionModel()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = []

    best_validation_loss = float("inf")
    best_state = None

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        prediction, _ = model(X_train)

        train_loss = (
            (prediction - y_train) ** 2
        ).mean()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            validation_prediction, _ = model(
                X_validation
            )

            validation_loss = (
                (
                    validation_prediction
                    - y_validation
                ) ** 2
            ).mean()

        train_value = float(
            train_loss.item()
        )

        validation_value = float(
            validation_loss.item()
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_value,
                "validation_loss": validation_value,
            }
        )

        if validation_value < best_validation_loss:
            best_validation_loss = validation_value

            best_state = {
                name: value.detach().clone()
                for name, value
                in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(
            "No best validation state was recorded"
        )

    model.load_state_dict(best_state)

    return (
        model,
        history,
        train_mean,
        train_std,
    )


# ------------------------------------------------------------
# Prediction and checkpoint
# ------------------------------------------------------------

def predict(model, X, mean, std):
    """Return predictions on the original demand scale."""
    X_scaled = standardize(
        X,
        mean,
        std,
    )

    model.eval()

    with torch.no_grad():
        prediction, weights = model(
            torch.from_numpy(X_scaled)
        )

    prediction = (
        prediction.numpy() * std + mean
    )

    return (
        prediction,
        weights.numpy(),
    )


def save_checkpoint(model, mean, std):
    """Save the trained model and training-only normalization."""
    torch.save(
        {
            "model_state": model.state_dict(),
            "train_mean": mean,
            "train_std": std,
        },
        MODEL_PATH,
    )


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------

def main():
    train = load_split(TRAIN_PATH)
    validation = load_split(VALIDATION_PATH)
    test = load_split(TEST_PATH)

    print("Stage 6 - Warehouse forecasting")
    print(
        "Split sizes: "
        f"train={len(train['y'])}, "
        f"validation={len(validation['y'])}, "
        f"test={len(test['y'])}"
    )

    model, history, train_mean, train_std = (
        train_attention_model(
            train["X"],
            train["y"],
            validation["X"],
            validation["y"],
        )
    )

    save_checkpoint(
        model,
        train_mean,
        train_std,
    )

    attention_prediction, _ = predict(
        model,
        test["X"],
        train_mean,
        train_std,
    )

    predictions = {
        "Persistence": persistence_baseline(
            test["X"]
        ),
        "Previous-day same-hour": previous_day_baseline(
            test["X"]
        ),
        "24-hour mean": mean24_baseline(
            test["X"]
        ),
        "Attention": attention_prediction,
    }

    rows = []

    for name, prediction in predictions.items():
        result = metrics(
            test["y"],
            prediction,
        )

        rows.append(
            {
                "model": name,
                "MAE": result["MAE"],
                "RMSE": result["RMSE"],
                "MAPE_percent": result["MAPE"],
            }
        )

    # Save numerical results.
    with RESULTS_PATH.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "MAE",
                "RMSE",
                "MAPE_percent",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    # Save training history.
    with HISTORY_PATH.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "validation_loss",
            ],
        )

        writer.writeheader()
        writer.writerows(history)

    # Save training plot.
    epochs = [
        row["epoch"]
        for row in history
    ]

    train_loss = [
        row["train_loss"]
        for row in history
    ]

    validation_loss = [
        row["validation_loss"]
        for row in history
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        train_loss,
        label="Train loss",
    )

    plt.plot(
        epochs,
        validation_loss,
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Standardized MSE")
    plt.title(
        "Warehouse Attention Training"
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

    # Console summary.
    print(
        f"Train mean: {train_mean:.6f}"
    )

    print(
        f"Train standard deviation: "
        f"{train_std:.6f}"
    )

    print()

    for row in rows:
        print(
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
        f"History: {HISTORY_PATH}"
    )
    print(
        f"Plot: {PLOT_PATH}"
    )
    print(
        f"Model: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
