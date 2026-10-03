import torch

from toy_learning import ToyAttentionModel, make_toy_data, train_model


def test_toy_data_is_deterministic():
    X1, y1 = make_toy_data(seed=42)
    X2, y2 = make_toy_data(seed=42)
    torch.testing.assert_close(X1, X2)
    torch.testing.assert_close(y1, y2)


def test_toy_model_output_shape():
    X, target = make_toy_data(n_samples=16)
    model = ToyAttentionModel()
    prediction, weights = model(X)
    assert prediction.shape == target.shape
    assert weights.shape == (16, 8, 8)


def test_toy_training_reduces_loss():
    X, target = make_toy_data(n_samples=256)
    _, losses = train_model(True, X, target, epochs=30)
    assert losses[-1] < losses[0]
    assert torch.isfinite(torch.tensor(losses)).all()
