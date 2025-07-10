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

## How to cite this work?

- Original paper

    [![DOI:10.48550/arXiv.2404.13910](http://img.shields.io/badge/DOI-10.48550/arXiv.2404.13910-B31B1B.svg)](https://doi.org/10.48550/arXiv.2404.13910)

    **Integrated Gradient Correlation: a Dataset-wise Attribution Method**\
    [Pierre Lelièvre](https://plelievre.com), Chien-Chung Chen\
    *Department of Psychology, National Taiwan University*

    ```bibtex
    @misc{lelièvre2024igc,
        title={Integrated Gradient Correlation: a Dataset-wise Attribution Method},
        author={Pierre Lelièvre and Chien-Chung Chen},
        year={2024},
        eprint={2404.13910},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }
    ```

- Package

    [![DOI:10.48550/arXiv.2404.13910](http://img.shields.io/badge/DOI-10.48550/arXiv.2404.13910-B31B1B.svg)](https://doi.org/10.48550/arXiv.2404.13910)

    **To be defined**\
    [Pierre Lelièvre](https://plelievre.com)\
    *Department of Psychology, National Taiwan University*

    ```bibtex
    @misc{lelièvre2025igc,
        title={to be defined},
        author={Pierre Lelièvre},
        year={2025},
    }
    ```

## Index

- [Index](https://plelievre.github.io/int_grad_corr/genindex.html)

## License

The IGC library is freely available under the [MIT License](https://github.com/plelievre/int_grad_corr/blob/main/LICENSE).

Copyright 2024 Pierre Lelièvre
