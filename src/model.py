"""
MLP architecture for well log regression.

Parametrizable feed-forward network with ReLU activations and optional dropout.
Default configuration: 5 → 64 → 64 → 32 → 1.

Called by: src/train.py
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron for well log regression.

    Args:
        input_dim: Number of input features. Default: 5 (GR, RT, RILM, NPHI, SP).
        hidden_dims: Sequence of hidden layer sizes. Default: (64, 64, 32).
        output_dim: Number of output units. Default: 1 (DEN).
        dropout: Dropout probability applied after each hidden ReLU. 0 disables it.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dims: tuple[int, ...] = (64, 64, 32),
        output_dim: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        return self.net(x)
