import sys
from pathlib import Path

# Add project root (Ware_house_project) to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from Numerical_stability.numerical_stability import naive_softmax, stable_softmax


def test_naive_softmax_overflows_on_large_scores():
    scores = torch.tensor([[1000.0, 999.0, 998.0]])
    result = naive_softmax(scores)
    assert not torch.isfinite(result).all()


def test_stable_softmax_remains_finite():
    scores = torch.tensor([[1000.0, 999.0, 998.0]])
    result = stable_softmax(scores)
    assert torch.isfinite(result).all()
    assert torch.allclose(result.sum(dim=-1), torch.ones(1))


def test_attention_weights_remain_finite():
    torch.manual_seed(7)
    from Attention_mechanism.attention import SelfAttention

    model = SelfAttention(d_model=2)
    X = torch.tensor([[[1.0e10, 1.0e10], [1.0e10, 1.0e10]]])
    _, weights = model(X)
    assert torch.isfinite(weights).all()
