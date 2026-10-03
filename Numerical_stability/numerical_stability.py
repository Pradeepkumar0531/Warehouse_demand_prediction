# Demonstrate why stable softmax is required for attention scores.

import sys
from pathlib import Path

# Add project root (Ware_house_project) to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
from Attention_mechanism.attention import SelfAttention


def naive_softmax(scores):
    exponentials = torch.exp(scores)
    return exponentials / exponentials.sum(dim=-1, keepdim=True)


def stable_softmax(scores):
    shifted = scores - scores.max(dim=-1, keepdim=True).values
    exponentials = torch.exp(shifted)
    return exponentials / exponentials.sum(dim=-1, keepdim=True)


if __name__ == "__main__":
    scores = torch.tensor([[1000.0, 999.0, 998.0]])

    naive = naive_softmax(scores)
    stable = stable_softmax(scores)

    print("Numerical stability")
    print(f"Input scores: {scores.tolist()}")
    print(f"Naive softmax: {naive.tolist()}")
    print(f"Naive finite: {torch.isfinite(naive).all().item()}")
    print(f"Stable softmax: {stable.tolist()}")
    print(f"Stable finite: {torch.isfinite(stable).all().item()}")
    print(f"Stable row sum: {stable.sum(dim=-1).item():.6f}")

    model = SelfAttention(d_model=2)
    X = torch.tensor([[[1.0e10, 1.0e10], [1.0e10, 1.0e10]]])
    _, weights = model(X)

    print(f"Attention weights finite: {torch.isfinite(weights).all().item()}")
    print(f"Attention rows sum to: {weights.sum(dim=-1).tolist()}")
