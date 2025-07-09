Examples
========

Environment Installation
------------------------

- Create and activate a `conda`_ environment named `int_grad_corr``

.. code-block:: console

   conda env create -f examples/environment.yml
   conda activate int_grad_corr

- | If you have a GPU supporting `CUDA`_, you can also use the file named
  | `environment_gpu.yml`.

- Install the IGC package with `pip`_

.. code-block:: console

   pip install .

.. _conda: https://www.anaconda.com/docs/getting-started/miniconda/main
.. _CUDA: https://developer.nvidia.com/cuda-downloads
.. _pip: https://pip.pypa.io

IGC Benchmark
-------------

In order to assess the relevancy of IGC attributions compared to other naive
dataset-wise attribution methods, we propose synthetic experiments. We first
define some localized image statistics computed on random images, and then try
to recover the generating masks/rules from the pairs of original images and
computed statistics. To make this procedure feasible, generated random images
are designed to respect the spatial frequency distribution of natural images
(i.e. having some spatial redundancy). For more details, look at the `Benchmark`
section of the original IGC `paper`_.

.. note::

   These example are available as a `notebook`_.

.. image:: _static/igc_simulation.jpeg

.. _paper: http://arxiv.org/abs/2404.13910
.. _notebook: https://github.com/plelievre/int_grad_corr/blob/main/examples/igc_simulation.ipynb

Representation of Image Statistics in the Brain
-----------------------------------------------

IGC is employed on a deep decoding model that predicts image statistics from the
neural activity (fMRI) induced by the vision of corresponding image stimuli.
IGC attributions then enable the investigation of the representation of image
statistics in the brain.

.. note::

   Not released yet. See more information in the original IGC `paper`_.

.. image:: _static/igc_imst.jpeg


Visual Receptive Field of Neural Populations
--------------------------------------------

IGC is employed on a deep encoding model that predicts the neural activity
(fMRI) from original image stimuli. IGC attributions then reflect the localized
image area for which each group of neurons is responsive, i.e. their population
receptive field (pRF).

.. note::

   Not released yet. See more information in the original IGC `paper`_.

.. image:: _static/igc_prf.jpeg
