"""Verify and visualize the first-principles attention implementation."""

from pathlib import Path
import math

import matplotlib.pyplot as plt
import torch

from attention import SelfAttention


OUTPUT_DIR = Path(__file__).resolve().parent / "verification_outputs"


def finite_difference_gradient(model, X, target, parameter, epsilon=1e-5):
    """Estimate every parameter gradient with the central-difference formula."""
    numerical = torch.zeros_like(parameter)

    with torch.no_grad():
        for index in range(parameter.numel()):
            original = parameter.view(-1)[index].item()

            parameter.view(-1)[index] = original + epsilon
            plus = ((model(X)[0] - target) ** 2).mean().item()

            parameter.view(-1)[index] = original - epsilon
            minus = ((model(X)[0] - target) ** 2).mean().item()

            parameter.view(-1)[index] = original
            numerical.view(-1)[index] = (plus - minus) / (2 * epsilon)

    return numerical


def relative_error(autograd, numerical):
    """Return a scale-aware gradient error."""
    numerator = torch.linalg.vector_norm(autograd - numerical)
    denominator = torch.linalg.vector_norm(autograd) + torch.linalg.vector_norm(numerical)
    return (numerator / denominator.clamp_min(1e-12)).item()


def verify_shapes_and_weights():
    """Check tensor shapes and the probability properties of attention weights."""
    torch.manual_seed(7)
    model = SelfAttention(d_model=4, d_k=4, d_v=4)
    X = torch.randn(2, 6, 4)
    output, weights = model(X)

    assert output.shape == (2, 6, 4)
    assert weights.shape == (2, 6, 6)
    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 6), atol=1e-6)

    return weights[0].detach()


def verify_gradient():
    """Compare autograd gradients against central finite differences."""
    torch.manual_seed(11)

    # Float64 reduces finite-difference round-off for this small numerical check.
    model = SelfAttention(d_model=3, d_k=3, d_v=3).double()
    X = torch.randn(2, 4, 3, dtype=torch.float64)
    target = torch.randn(2, 4, 3, dtype=torch.float64)

    output, _ = model(X)
    loss = ((output - target) ** 2).mean()
    loss.backward()

    errors = {}
    for name in ("W_Q", "W_K", "W_V"):
        parameter = getattr(model, name)
        numerical = finite_difference_gradient(model, X, target, parameter)
        errors[name] = relative_error(parameter.grad, numerical)

    maximum_error = max(errors.values())
    assert maximum_error < 1e-6, f"Gradient check failed: {errors}"
    return errors


def verify_numerical_stability():
    """Show that max-subtraction prevents exponential overflow in softmax."""
    scores = torch.tensor([[1000.0, 999.0, 998.0]])

    naive = torch.exp(scores) / torch.exp(scores).sum(dim=-1, keepdim=True)
    stable_scores = scores - scores.max(dim=-1, keepdim=True).values
    stable = torch.exp(stable_scores) / torch.exp(stable_scores).sum(dim=-1, keepdim=True)

    assert not torch.isfinite(naive).all()
    assert torch.isfinite(stable).all()
    assert torch.allclose(stable.sum(dim=-1), torch.ones(1), atol=1e-6)

    return naive, stable


def visualize(weights, gradient_errors, naive, stable):
    """Save one compact figure containing the verification evidence."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))

    image = axes[0].imshow(weights.numpy(), aspect="auto")
    axes[0].set_title("Attention weights")
    axes[0].set_xlabel("Key position")
    axes[0].set_ylabel("Query position")
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    names = list(gradient_errors)
    values = [gradient_errors[name] for name in names]
    axes[1].bar(names, values)
    axes[1].set_title("Finite-difference relative error")
    axes[1].set_ylabel("Relative error")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axes[2].plot(stable.numpy().ravel(), marker="o", label="Stable")
    axes[2].plot(naive.numpy().ravel(), marker="x", label="Naive")
    axes[2].set_title("Softmax at large logits")
    axes[2].set_xlabel("Logit position")
    axes[2].set_ylabel("Probability")
    axes[2].legend()

    figure.tight_layout()
    path = OUTPUT_DIR / "attention_verification.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def main():
    weights = verify_shapes_and_weights()
    gradient_errors = verify_gradient()
    naive, stable = verify_numerical_stability()
    figure_path = visualize(weights, gradient_errors, naive, stable)

    print("Attention verification passed.")
    print(f"Output shape: (2, 6, 4)")
    print(f"Weight shape: {tuple(weights.shape)}")
    print(f"First attention row sum: {weights[0].sum().item():.6f}")
    print("Gradient relative errors:")
    for name, error in gradient_errors.items():
        print(f"  {name}: {error:.3e}")
    print(f"Naive softmax finite: {torch.isfinite(naive).all().item()}")
    print(f"Stable softmax finite: {torch.isfinite(stable).all().item()}")
    print(f"Visualization: {figure_path}")


if __name__ == "__main__":
    main()
