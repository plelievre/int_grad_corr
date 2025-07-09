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

- [User Guide](https://github.com/plelievre/int_grad_corr.github.io)

- [Examples](https://github.com/plelievre/int_grad_corr.github.io)

## API Reference

- [API Reference](https://github.com/plelievre/int_grad_corr.github.io)

## How to cite this work?

- For general and theoretical aspects

    **Integrated Gradient Correlation: a Dataset-wise Attribution Method** ([arXiv](http://arxiv.org/abs/2404.13910))\
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

- For the use of this package

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

- [Index](https://github.com/plelievre/int_grad_corr.github.io)

## License

The IGC library is freely available under the [MIT License](https://github.com/plelievre/int_grad_corr/blob/main/LICENSE).

Copyright 2024 Pierre Lelièvre
