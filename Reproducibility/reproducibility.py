"""Stage 9 - Reproducibility checks for the warehouse attention project."""

import hashlib
import platform
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from Data_Generation.data_generator import (
    generate_warehouse_data,
)
from Warehouse_forecasting.warehouse_forecasting import (
    load_split,
    train_attention_model,
)


REPORT_PATH = (
    Path(__file__).resolve().parent
    / "reproducibility_report.txt"
)

SEED = 42


def hash_array(array):
    """Return a stable SHA-256 hash for a numpy array."""
    contiguous = np.ascontiguousarray(array)

    return hashlib.sha256(
        contiguous.tobytes()
    ).hexdigest()


def compare_generated_data():
    """Generate the synthetic environment twice with the same seed."""
    first = generate_warehouse_data(
        num_days=365,
        seed=SEED,
    )

    second = generate_warehouse_data(
        num_days=365,
        seed=SEED,
    )

    same_values = first.equals(
        second
    )

    first_hash = hash_array(
        first["demand"].to_numpy()
    )

    second_hash = hash_array(
        second["demand"].to_numpy()
    )

    return {
        "same": same_values,
        "first_hash": first_hash,
        "second_hash": second_hash,
    }


def compare_attention_training():
    """Train the same model twice and compare parameters and losses."""
    train_path = (
        PROJECT_ROOT
        / "Data_Generation"
        / "train_data.npz"
    )

    validation_path = (
        PROJECT_ROOT
        / "Data_Generation"
        / "validation_data.npz"
    )

    train = load_split(
        train_path
    )

    validation = load_split(
        validation_path
    )

    first_model, first_history, first_mean, first_std = (
        train_attention_model(
            train["X"],
            train["y"],
            validation["X"],
            validation["y"],
        )
    )

    second_model, second_history, second_mean, second_std = (
        train_attention_model(
            train["X"],
            train["y"],
            validation["X"],
            validation["y"],
        )
    )

    same_history = (
        first_history == second_history
    )

    same_mean = (
        first_mean == second_mean
    )

    same_std = (
        first_std == second_std
    )

    same_parameters = True

    for name, first_parameter in (
        first_model.state_dict().items()
    ):
        second_parameter = (
            second_model.state_dict()[name]
        )

        if not torch.equal(
            first_parameter,
            second_parameter,
        ):
            same_parameters = False
            break

    return {
        "same_history": same_history,
        "same_parameters": same_parameters,
        "same_mean": same_mean,
        "same_std": same_std,
        "first_final_train_loss": (
            first_history[-1]["train_loss"]
        ),
        "second_final_train_loss": (
            second_history[-1]["train_loss"]
        ),
    }


def environment_summary():
    """Record the Python and core-library versions."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "seed": SEED,
    }


def main():
    print(
        "Stage 9 - Reproducibility"
    )

    data_result = compare_generated_data()

    training_result = (
        compare_attention_training()
    )

    environment = environment_summary()

    all_passed = (
        data_result["same"]
        and data_result["first_hash"]
        == data_result["second_hash"]
        and training_result["same_history"]
        and training_result["same_parameters"]
        and training_result["same_mean"]
        and training_result["same_std"]
    )

    lines = [
        "Stage 9 - Reproducibility Report",
        "",
        "Synthetic data:",
        f"  same values: {data_result['same']}",
        f"  first demand hash: {data_result['first_hash']}",
        f"  second demand hash: {data_result['second_hash']}",
        "",
        "Attention training:",
        f"  same loss history: {training_result['same_history']}",
        f"  same parameters: {training_result['same_parameters']}",
        f"  same training mean: {training_result['same_mean']}",
        f"  same training std: {training_result['same_std']}",
        f"  first final train loss: {training_result['first_final_train_loss']:.12f}",
        f"  second final train loss: {training_result['second_final_train_loss']:.12f}",
        "",
        "Environment:",
        f"  Python: {environment['python']}",
        f"  Platform: {environment['platform']}",
        f"  NumPy: {environment['numpy']}",
        f"  PyTorch: {environment['torch']}",
        f"  Seed: {environment['seed']}",
        "",
        f"REPRODUCIBILITY CHECK: {'PASS' if all_passed else 'FAIL'}",
    ]

    report = "\n".join(
        lines
    )

    REPORT_PATH.write_text(
        report
        + "\n"
    )

    print(report)
    print()
    print(
        f"Report: {REPORT_PATH}"
    )

    if not all_passed:
        raise SystemExit(
            "Reproducibility check failed."
        )


if __name__ == "__main__":
    main()
