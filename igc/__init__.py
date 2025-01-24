"""
Integrated Gradient Correlation (IGC) utils implemented in PyTorch.

This module provides functions to compute:
- Gradients
- Integrated Gradients
- Integrated Gradient Correlation
- Integrated Gradient Auto-Correlation
- IGC error
- IGaC error

A sub-module naive_2v0 provides (for demonstration purpose only):
- IG mean and std
- naive correlation
- naive ttest

A sub-module bsc_1v0 provides (for demonstration purpose only):
- Baseline Shapley
- Baseline Shapley Correlation

Versions:
1v0: Functions can only study model inputs individually.
2v0: Optimized computations. Functions accept embedded inputs.
     Functions can study multiple model inputs simultaneously.

Author: Pierre Lelievre
"""

from .igc_2v0 import (
    grad, int_grad, int_grad_corr, igc_error, int_grad_auto_corr, igac_error)
