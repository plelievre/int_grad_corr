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

from .igc_1v0 import grad, int_grad, int_grad_corr
from .bsc_1v0 import bsl_shap, bsl_shap_corr
