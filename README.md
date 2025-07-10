# Integrated Gradient Correlation

Integrated Gradient Correlation (IGC) is a Python/PyTorch package that provides
a unique dataset-wise attribution method.

It is designed to improve the interpretability of deep neural networks at a
*task-level*, rather than an *instance-level* (as available attribution methods
generally do). For more theoretical details, please refer to the original
[paper](http://arxiv.org/abs/2404.13910).

This package primarily focuses on a class computing IGC attributions for PyTorch
modules. Nonetheless, it also offers utilities to calculate simple gradients,
Integrated Gradients (IG), and some naive dataset-wise attribution methods.

## Usage

- [User Guide](https://plelievre.github.io/int_grad_corr/guide.html)

- [Examples](https://plelievre.github.io/int_grad_corr/examples.html)

## API Reference

- [API Reference](https://plelievre.github.io/int_grad_corr/api.html)

## Citations

- **Integrated Gradient Correlation: a Dataset-wise Attribution Method**\
    [Pierre Lelièvre](https://plelievre.com), Chien-Chung Chen\
    *Department of Psychology, National Taiwan University*

    [![DOI](http://img.shields.io/badge/DOI-10.48550/arXiv.2404.13910-B31B1B.svg)](https://doi.org/10.48550/arXiv.2404.13910)


- **Package** (latest version)

    [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15852412.svg)](https://doi.org/10.5281/zenodo.15852412)

## License

The IGC library is freely available under the [MIT License](https://github.com/plelievre/int_grad_corr/blob/main/LICENSE).

Copyright 2024 Pierre Lelièvre
