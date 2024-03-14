"""
Dataloader utils.

These utils require a specific pattern. See below for a generic example.

————————

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset


class _Dataset(TorchDataset):
    def __init__(self, data, seed=None):
        self.data = data
        self.rng = None
        if seed is not None:
            self.rng = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_i = self.data[idx]
        # Use self.rng for data augmentation. Caution, self.rng may be None.
        return data_i


class Dataset:
    def __init__(self, data):
        self.data = data

    @torch.no_grad()
    def dtld_func(self, batch_size, seed=None, num_workers=0):
        return DataLoader(
            _Dataset(self.data, seed),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            worker_init_fn=set_worker_seed,
            generator=torch.Generator().manual_seed(seed))

————————

Author: Pierre Lelievre
"""

import torch


# Utils


def set_dtld_seed(dtld, seed):
    """
    Update dataloader seeds after initialization.
    """
    dtld.generator.manual_seed(seed)
    if dtld.dataset.rng is not None:
        dtld.dataset.rng.manual_seed(seed)


def set_worker_seed(worker_id):
    """
    Set seeds of dataloader's workers.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.rng.manual_seed(dataset.rng.initial_seed() + int(1e6*worker_id))
