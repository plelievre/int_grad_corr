"""
Integrated Gradient Correlation (IGC) utils implemented in PyTorch.

The module provides functions to compute:
- Gradients
- Integrated Gradients
- Integrated Gradient Correlation
- Baseline Shapley
- Baseline Shapley Correlation

Author: Pierre Lelievre
"""

from .igc_1v0 import (
    grad_1_x, grad_dtld, int_grad_1_x, int_grad_dtld, int_grad_corr_dtld)
from .bsc_1v0 import bsl_shap_1_x, bsl_shap_dtld, bsl_shap_corr_dtld
