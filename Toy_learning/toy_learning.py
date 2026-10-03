# Small supervised task for observing attention learning dynamics.

import sys
from pathlib import Path
# Add project root (Ware_house_project) to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import csv
import matplotlib.pyplot as plt

import torch
from torch import nn
from Attention_mechanism.attention import SelfAttention


class ToyAttentionModel(nn.Module):
    """Attention followed by a linear head for next-value prediction."""

    def __init__(self, d_model=4, scale=True):
        super().__init__()
        self.attention = SelfAttention(d_model=d_model, scale=scale)
        self.head = nn.Linear(d_model, 1)

    def forward(self, X):
        output, weights = self.attention(X)
        prediction = self.head(output[:, -1, :]).squeeze(-1)
        return prediction, weights


def make_toy_data(n_samples=512, sequence_length=8, seed=42):
    """Create a deterministic AR-style next-value prediction problem."""
    generator = torch.Generator().manual_seed(seed)
    X = torch.randn(n_samples, sequence_length, 1, generator=generator)

    # The target depends mostly on recent observations, giving attention
    # something meaningful to learn while keeping the task easy to inspect.
    target = (
        0.60 * X[:, -1, 0]
        + 0.25 * X[:, -2, 0]
        + 0.10 * X[:, -3, 0]
        + 0.05 * X[:, -4, 0]
    )

    # Pad the scalar feature to a small model dimension.
    X = X.repeat(1, 1, 4)
    return X, target


def train_model(scale, X, target, epochs=100, learning_rate=0.01, seed=7):
    """Train one scaled or unscaled attention model and return its losses."""
    torch.manual_seed(seed)
    model = ToyAttentionModel(d_model=X.shape[-1], scale=scale)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    for _ in range(epochs):
        prediction, _ = model(X)
        loss = ((prediction - target) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return model, losses

if __name__ == "__main__":
    OUTPUT = Path(__file__).resolve().parent / "toy_learning_results.csv"
    FIGURE = Path(__file__).resolve().parent / "toy_learning_dynamics.png"

    X, target = make_toy_data()

    scaled_model, scaled_losses = train_model(
        True,
        X,
        target
    )

    unscaled_model, unscaled_losses = train_model(
        False,
        X,
        target
    )

    # Save numerical results.
    with OUTPUT.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "scaled_loss", "unscaled_loss"])

        for epoch, (scaled, unscaled) in enumerate(
            zip(scaled_losses, unscaled_losses),
            start=1
        ):
            writer.writerow([epoch, scaled, unscaled])

    # Create the learning-dynamics plot.
    epochs = range(1, len(scaled_losses) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        scaled_losses,
        label="Scaled attention"
    )

    plt.plot(
        epochs,
        unscaled_losses,
        label="Unscaled attention"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Toy Learning Dynamics")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the figure in the same directory as this script.
    plt.savefig(FIGURE, dpi=160)

    # Display it when running interactively.
    plt.show()

    # Close the figure after displaying/saving it.
    plt.close()

    print("Toy learning experiment")
    print(f"Epochs: {len(scaled_losses)}")
    print(f"Scaled final loss: {scaled_losses[-1]:.6f}")
    print(f"Unscaled final loss: {unscaled_losses[-1]:.6f}")
    print(f"Scaled initial loss: {scaled_losses[0]:.6f}")
    print(f"Unscaled initial loss: {unscaled_losses[0]:.6f}")
    print(f"Results: {OUTPUT}")
    print(f"Figure: {FIGURE}")