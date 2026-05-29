"""Tests for src/model.py."""

import torch
import torch.nn as nn

from src.model import MLP


def test_forward_default_shape():
    model = MLP()
    x = torch.randn(32, 5)
    y = model(x)
    assert y.shape == (32, 1)


def test_forward_custom_dims():
    model = MLP(input_dim=3, hidden_dims=(16, 8), output_dim=1)
    x = torch.randn(10, 3)
    assert model(x).shape == (10, 1)


def test_no_dropout_when_zero():
    model = MLP(dropout=0.0)
    has_dropout = any(isinstance(m, nn.Dropout) for m in model.net)
    assert not has_dropout


def test_dropout_present_when_nonzero():
    model = MLP(dropout=0.3)
    has_dropout = any(isinstance(m, nn.Dropout) for m in model.net)
    assert has_dropout


def test_output_layer_is_linear():
    """Last module in net must be Linear (no activation capping output range)."""
    model = MLP()
    assert isinstance(model.net[-1], nn.Linear)


def test_gradients_flow_through_all_params():
    model = MLP()
    x = torch.randn(16, 5)
    loss = model(x).sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_batch_size_one():
    model = MLP()
    x = torch.randn(1, 5)
    assert model(x).shape == (1, 1)


def test_single_hidden_layer():
    model = MLP(input_dim=5, hidden_dims=(32,))
    x = torch.randn(8, 5)
    assert model(x).shape == (8, 1)
