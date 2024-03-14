"""
Custom modules.

Author: Pierre Lelievre
"""

import torch
from torch import nn


# Activation functions


class PMish(nn.Module):
    """
    Parameterized version of Mish activation function [arXiv:1908.08681].
    As weight -> -inf, PMish -> Mish
       weight -> +inf, PMish -> identity function
    """
    def __init__(self, init=0.0):
        super().__init__()
        weight = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(weight, init)
        self.register_parameter('weight', weight)

    def forward(self, x):
        alpha = nn.functional.sigmoid(self.weight)
        return x * nn.functional.tanh(alpha + nn.functional.softplus(x))
