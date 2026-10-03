# Finite-difference gradient verification for first-principles attention.

import sys
from pathlib import Path

# Add project root (Ware_house_project) to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from Attention_mechanism.attention import SelfAttention

import torch
from Attention_mechanism.attention import SelfAttention


def scalar_loss(model, X, target):
    output, _ = model(X)
    return ((output - target) ** 2).mean()


def finite_difference(model, X, target, parameter, epsilon=1e-5):
    """Return a finite-difference gradient for every scalar in one parameter."""
    numerical = torch.zeros_like(parameter)
    flat = parameter.view(-1)
    numerical_flat = numerical.view(-1)

    for i in range(flat.numel()):
        original = flat[i].item()

        with torch.no_grad():
            flat[i] = original + epsilon
        loss_plus = scalar_loss(model, X, target).item()

        with torch.no_grad():
            flat[i] = original - epsilon
        loss_minus = scalar_loss(model, X, target).item()

        with torch.no_grad():
            flat[i] = original

        numerical_flat[i] = (loss_plus - loss_minus) / (2.0 * epsilon)

    return numerical


def relative_error(autograd, numerical):
    numerator = torch.linalg.vector_norm(autograd - numerical)
    denominator = torch.linalg.vector_norm(autograd) + torch.linalg.vector_norm(numerical)
    return (numerator / denominator.clamp_min(1e-12)).item()


def run_gradient_check(seed=7):
    torch.manual_seed(seed)

    model = SelfAttention(d_model=3, d_k=3, d_v=3).double()
    X = torch.randn(2, 4, 3, dtype=torch.float64)
    target = torch.randn(2, 4, 3, dtype=torch.float64)

    loss = scalar_loss(model, X, target)
    loss.backward()

    results = {}
    for name, parameter in model.named_parameters():
        numerical = finite_difference(model, X, target, parameter)
        error = relative_error(parameter.grad, numerical)
        results[name] = {
            "relative_error": error,
            "max_abs_error": (parameter.grad - numerical).abs().max().item(),
        }

    return results


if __name__ == "__main__":
    results = run_gradient_check()
    print("Gradient verification")
    for name, values in results.items():
        print(
            f"{name}: relative_error={values['relative_error']:.3e}, "
            f"max_abs_error={values['max_abs_error']:.3e}"
        )
