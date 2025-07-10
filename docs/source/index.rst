
Integrated Gradient Correlation
===============================

Integrated Gradient Correlation (IGC) is a Python/PyTorch package that provides
a unique dataset-wise attribution method.

It is designed to improve the interpretability of deep neural networks at a
*task-level*, rather than an *instance-level* (as available attribution methods
generally do). For more theoretical details, please refer to the original
`paper <http://arxiv.org/abs/2404.13910>`_
:cite:`LelievreIntegratedGradientCorrelation2024`.

This package primarily focuses on a class computing IGC attributions for PyTorch
modules. Nonetheless, it also offers utilities to calculate simple gradients,
Integrated Gradients (IG), and some naive dataset-wise attribution methods.

Usage
-----

.. toctree::
   :titlesonly:
   :maxdepth: 1

   guide
   examples

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :titlesonly:

   api

Citations
---------

- | **Integrated Gradient Correlation: a Dataset-wise Attribution Method**
  | `Pierre Lelièvre <https://plelievre.com>`_, Chien-Chung Chen
  | *Department of Psychology, National Taiwan University*
  | |badge_1|

.. |badge_1| image:: http://img.shields.io/badge/DOI-10.48550/arXiv.2404.13910-B31B1B.svg
   :target: https://doi.org/10.48550/arXiv.2404.13910

- | Package (latest version)
  | |badge_2|

.. |badge_2| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.15852412.svg
   :target: https://doi.org/10.5281/zenodo.15852412

License
-------

The IGC library is freely available under the `MIT License`_.

Copyright 2024 Pierre Lelièvre

.. _MIT License: https://github.com/plelievre/int_grad_corr/blob/main/LICENSE

Index
-----

* :ref:`genindex`

Bibliography
------------

.. bibliography::
   :all:
