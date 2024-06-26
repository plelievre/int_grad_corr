# Integrated Gradient Correlation
[arXiv](http://arxiv.org/abs/2404.13910) | [BibTeX](#bibtex)

[**Integrated Gradient Correlation: a Dataset-wise Attribution Method**](http://arxiv.org/abs/2404.13910)<br/>
[Pierre Lelièvre](https://plelievre.com)\*, Chien-Chung Chen\*<br>
<sub><sup>\*Department of Psychology, National Taiwan University<sub><sup>


| ![IGC fMRI (luminance contrast)](assets/igc_fmri_contrast.jpg) |
|:--:|
| *IGC maps associated with the prediction of image luminance contrast from fMRI data.* |

| ![IGC MNIST](assets/igc_mnist.jpg) |
|:--:|
| *IGC maps for the MNIST dataset w.r.t. all possible digit classes.* |

## Requirements
A suitable [conda](https://conda.io/) environment named **int_grad_corr** can be
created and activated with:

```
conda env create -f environment.yml
conda activate int_grad_corr
```

If you have a GPU supporting [CUDA](https://developer.nvidia.com/cuda-downloads),
you can also use:

```
conda env create -f environment_gpu.yml
conda activate int_grad_corr
```

## Usage
The paper demonstrates the IGC method on several examples, but only the
[MNIST](http://yann.lecun.com/exdb/mnist/) dataset is easy and light to
download. As a result, we provide a [model](mnist/model_mnist_1v0.py) and a
[Jupyter Notebook](igc_mnist.ipynb) for this dataset only.

However, the core python code to compute IGC (located [here](igc/igc_1v0.py))
can be easily employed with your own models and data.

## BibTeX

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
