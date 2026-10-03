import torch
from gradient_check import run_gradient_check


def test_gradient_check_is_close():
    results = run_gradient_check()
    for values in results.values():
        assert values["relative_error"] < 1e-6


def test_gradient_check_has_no_nan():
    results = run_gradient_check()
    for values in results.values():
        assert torch.isfinite(torch.tensor(values["relative_error"]))
        assert torch.isfinite(torch.tensor(values["max_abs_error"]))
