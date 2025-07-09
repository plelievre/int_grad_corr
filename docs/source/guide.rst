User Guide
==========

Requirements
------------

+-------+----------------+
| Name  | Version        |
+=======+================+
|python | >= 3.9         |
+-------+----------------+
|numpy  |                |
+-------+----------------+
|scipy  |                |
+-------+----------------+
|torch  | >= 2           |
+-------+----------------+
|tqdm   | >= 4           |
+-------+----------------+

Installation
------------

- Clone the `repository`_

.. code-block:: console

   git clone https://github.com/plelievre/int_grad_corr.git

- Install the package with `pip`_

.. code-block:: console

   pip install .

.. _repository: https://github.com/plelievre/int_grad_corr
.. _pip: https://pip.pypa.io

Basic Usage
-----------

.. note::

   The following example is available as a `notebook`_.

.. _notebook: https://github.com/plelievre/int_grad_corr/blob/main/examples/basic_usage.ipynb

Create a dataset
^^^^^^^^^^^^^^^^

.. code-block:: python

   import torch
   import torch.nn as nn
   from torch.utils.data import Dataset as TorchDataset


   class Dataset(TorchDataset):
       def __init__(self, n_samples, x_size, seed=None):
           self.n_samples = n_samples
           self.x_size = x_size
           # Random generator
           self.rng = None
           if seed is not None:
               self.rng = torch.Generator().manual_seed(seed)

       def __len__(self):
           return self.n_samples

       def __getitem__(self, idx):
           x = torch.rand(self.x_size, generator=self.rng)
           y = x.mean()
           return x, y.unsqueeze(dim=0)


   dataset = Dataset(n_samples=100, x_size=5, seed=100)

   print(f"x: {dataset[0][0]}")
   print(f"y: {dataset[0][1].item():.4f}")

.. code-block:: pycon
   :caption: >>>

   x: tensor([0.1117, 0.8158, 0.2626, 0.4839, 0.6765])
   y: 0.2771

Create a model
^^^^^^^^^^^^^^

.. code-block:: python

   class Model(nn.Module):
       def __init__(self, x_size, hidden_size, seed=None):
           super().__init__()
           if seed is not None:
               torch.manual_seed(seed)
           self.lin1 = nn.Linear(x_size, hidden_size)
           self.relu = nn.ReLU()
           self.lin2 = nn.Linear(hidden_size, 1)

       def forward(self, x):
           return self.lin2(self.relu(self.lin1(x)))

Init IGC attribution method
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from igc import IntGradCorr

   attr = IntGradCorr(model, dataset)


Compute IGC attributions
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   igc = attr.compute(           # Compute IGC attributions
       x_0=8,                    # with 8 random baselines sampled from the dataset
       y_idx=None,               # for all model output components
       n_steps=64,               # with 64 steps for each individual supporting IG
       batch_size=(4, 8, None),  # with 4 'x' samples, 8 baselines, and all y
       x_seed=100,               #   components per batch. It could also be defined
       x_0_seed=101,             #   by an integer as: batch_size=32
       n_x=None,                 # and 'x' sampled over the whole dataset
   )

   print()
   print(igc)

.. code-block:: pycon
   :caption: >>>

   batch size: 32 (4, 8, 1)
   igc: 100%|██████████| 25/25 [00:00<00:00, 290.17it/s, ig err:  0.000095]
   igc err:  0.002362

   [[ 0.31934428 -0.20948061  0.05070509 -0.08051807  0.02530288]]
