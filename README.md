# Integrated Gradient Correlation
[arXiv](http://arxiv.org/abs/2404.13910) | [BibTeX](#bibtex)

[**Integrated Gradient Correlation: a Dataset-wise Attribution Method**](http://arxiv.org/abs/2404.13910)<br/>
[Pierre Lelièvre](https://plelievre.com)\*, Chien-Chung Chen\*<br>
<sub><sup>\*Department of Psychology, National Taiwan University<sub><sup>

| ![IGC benchmark](assets/igc_benchmark.jpg) |
|:--:|
| *IGC attributions associated with the prediction of localized image statistics from natural images.* |

| ![IGC fMRI (luminance contrast)](assets/igc_fmri_contrast.jpg) |
|:--:|
| *IGC attributions associated with the prediction of image luminance contrast from fMRI data.* |

## Requirements
A suitable [conda](https://conda.io/) environment named **int_grad_corr** can
be created and activated with:

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
The paper demonstrates IGC on several real-life examples but we
only provide a [Jupyter Notebook](igc_benchmark.ipynb) to reproduce the results
found in the benchmark section. This models predict localized image statistics
from the 73k natural images of the [NSD dataset](http://naturalscenesdataset.org)).

However, the core python code computing IGC (located [here](igc/igc_2v0.py))
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
