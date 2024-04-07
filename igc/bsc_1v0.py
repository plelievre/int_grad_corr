"""
Baseline Shapley Correlation (BSC) utils.

For optimization purposes, dataloader 'dtld_func', and gradient 'forward_func'
functions require some specific patterns. See below for generic examples.

———————— dtld_func

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset


class _Dataset(TorchDataset):
    def __init__(self, x, y, seed=None):
        self.x = x  # Input
        self.y = y  # Output
        self.rng = None
        if seed is not None:
            self.rng = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_i = self.x[idx]
        y_i = self.x[idx]
        # Use self.rng for data augmentation. Caution, self.rng may be None.
        return x_i, y_i


class Dataset:
    def __init__(self, x, y):
        self.x = x  # Input
        self.y = y  # Output

    @torch.no_grad()
    def dtld_func(self, batch_size, seed=None):
        return DataLoader(
            _Dataset(self.x, self.y, seed),
            batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed))

———————— forward_func

import torch
from torch import nn


class Model:
    def __init__(self, ...):
        self.module = Module(...)  # Inherited from nn.Module

    @torch.no_grad()
    def forward_func(self, x, y_idx):
        # Prepare model
        self.module.eval()
        # Eval
        y_r = self.module(x)
        y_r = torch.gather(y_r, dim=1, index=y_idx.unsqueeze(dim=1))
        return y_r.squeeze(dim=1).cpu().numpy()

Author: Pierre Lelievre
"""

import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr

import torch
from torch import nn

from .igc_1v0 import (
    _prepare_y_idx, _prepare_x_dtld, _prepare_x_0_dtld, _record_x_y,
    _set_dtld_seed)


# Baseline Shapley


@torch.no_grad()
def _bsl_shap_1_x_1_y(forward_func, x, y_idx, x_0_dtld, n_iter, n_x_0_batch,
                      x_0_batch_size, perm_seed, device, forward_kwargs,
                      check_error):
    n_features = x.numel()
    batch_size = x_0_batch_size * n_iter
    # Prepare inputs
    repeat_iter = (n_iter,) + (1,)*x.dim()
    repeat_batch = (batch_size,) + (1,)*x.dim()
    x = x.unsqueeze(dim=0).repeat(*repeat_batch).to(device)
    y_idx = y_idx.expand(batch_size).to(device)
    # Compute baseline Shapley
    y_0 = 0.0
    y_r = forward_func(
        x=x[0].unsqueeze(dim=0), y_idx=y_idx[0].unsqueeze(dim=0),
        **forward_kwargs)[0]
    bsl_shap_ = np.zeros(x.size()[1:], dtype=np.float32)
    for i, (x_0_i, _) in enumerate(x_0_dtld):
        # Break when n_x_0_batch is reached
        if i == n_x_0_batch:
            break
        # Send baseline to device
        x_0_i = x_0_i.to(device)
        # Compute y_0_i
        y_0_i = forward_func(
            x=x_0_i, y_idx=y_idx[:x_0_batch_size], **forward_kwargs)
        y_0 += np.sum(y_0_i)
        # Prepare and send baseline to device
        x_0_i = x_0_i.repeat(*repeat_iter).to(device)
        # Prepare permutations
        rng = np.random.default_rng(int(perm_seed+1e3*(i+1)))
        feature_mask = np.arange(
            n_features)[None, :].repeat(batch_size, axis=0)
        # Iterate over features
        prev_eval = np.tile(y_0_i, n_iter)
        mask = torch.zeros_like(x)
        feature_permutation = torch.as_tensor(
            rng.permuted(feature_mask, axis=1), device=device)
        for j in range(n_features):
            mask_j = nn.functional.one_hot(
                feature_permutation[:, j], num_classes=n_features)
            mask_j = mask_j.view(x.size())
            mask += mask_j
            x_j = torch.where(mask != 0, x, x_0_i)
            modified_eval = forward_func(
                x=x_j, y_idx=y_idx, **forward_kwargs)
            eval_diff = modified_eval - prev_eval
            bsl_shap_ += np.sum(eval_diff[(...,) + (None,)*(
                x.dim()-1)] * mask_j.cpu().numpy(), axis=0)
            prev_eval = modified_eval
    # Normalize across baselines (and iterations)
    n_x_0 = float(n_x_0_batch * x_0_batch_size)
    y_0 /= n_x_0
    bsl_shap_ /= n_x_0 * n_iter
    # Check baseline Shapley error
    if check_error:
        print(f'error: {np.sum(bsl_shap_) - y_r + y_0}')
    return y_0, y_r, bsl_shap_


@torch.no_grad()
def bsl_shap(forward_func, x=None, y_idx=None, x_0=None, n_iter=32,
             x_size=None, y_size=None, dtld_func=None, x_0_batch_size=None,
             x_seed=None, x_0_seed=100, dtype=torch.float32, device='cpu',
             dtld_kwargs=None, forward_kwargs=None, check_error=False,
             description=''):
    """
    Compute baseline Shapley (bsl_shap:array [n_x, n_y_idx, ...]):
        - for all inputs (x:array [n_x, ...]), for n_x inputs sampled from
        the dataloader (x:int), or all inputs of the dataloader (x:None)
        - w.r.t. one specific output component index (y_idx:int), multiple
        component indices (y_idx:array), or all components (y_idx:None)
        - using a predifined baseline (x_0:array), n baselines sampled from
        the dataloader (x_0:int), or initialized to a zero vector (x_0:None)
    """
    if forward_kwargs is None:
        forward_kwargs = {}
    # Prepare x dataloader
    x_batch_size = 1
    x_dtld, n_x, n_x_batch, x_size = _prepare_x_dtld(
        x, x_size, dtld_func, x_batch_size, x_seed, dtype, device, dtld_kwargs)
    # Prepare y_idx
    n_y_idx, y_idx = _prepare_y_idx(y_idx, y_size, device, x_batch_size)
    # Prepare x_0 dataloader
    x_0_dtld, n_x_0_batch, x_0_batch_size = _prepare_x_0_dtld(
        x_0, x_size, dtld_func, x_batch_size, x_0_batch_size, x_0_seed, dtype,
        device, dtld_kwargs)
    # Prepare outputs
    x = np.zeros((n_x,) + x_size, dtype=np.float32)
    y = np.zeros((n_x, n_y_idx), dtype=np.float32)
    y_0 = np.zeros((n_x, n_y_idx), dtype=np.float32)
    y_r = np.zeros((n_x, n_y_idx), dtype=np.float32)
    bsl_shap_ = np.zeros((n_x, n_y_idx) + x_size, dtype=np.float32)
    # Iterate over dataloader
    if not description:
        description = 'bs'
    for i, (x_i, y_i) in enumerate(
            tqdm(x_dtld, total=n_x_batch, desc=description)):
        # Break when n_x_batch is reached
        if i == n_x_batch:
            break
        # Record x, y
        x_i_np, y_i_np = _record_x_y(x_i, y_i, y_idx, x_batch_size)
        x[i] += x_i_np[0]
        if y_i_np is not None:
            y[i] += y_i_np[0]
        # Iterate over y_idx
        if not isinstance(x_0_dtld, list):
            _set_dtld_seed(x_0_dtld, int(x_0_seed+1e6*(i+1)))
        for j in range(n_y_idx):
            y_0_ij, y_r_ij, bsl_shap_ij = _bsl_shap_1_x_1_y(
                forward_func, x_i[0], y_idx[j], x_0_dtld, n_iter, n_x_0_batch,
                x_0_batch_size, x_0_seed, device, forward_kwargs, check_error)
            y_0[i, j] += y_0_ij
            y_r[i, j] += y_r_ij
            bsl_shap_[i, j] += bsl_shap_ij
    return x, y, y_0, y_r, bsl_shap_


# Baseline Shapley correlation


@torch.no_grad()
def bsl_shap_corr(forward_func, dtld_func, x_size, y_size, y_idx=None,
                  x_0=None, n_iter=32, x_0_batch_size=None, x_0_seed=100,
                  dtype=torch.float32, device='cpu', dtld_kwargs=None,
                  forward_kwargs=None, check_error=False, n_x=None):
    """
    Compute baseline Shapley correlation (bsc:array [n_y_idx, ...]):
        - for all inputs of the dataloader (except if n_x is defined)
        - w.r.t. one specific output component index (y_idx:int), multiple
        component indices (y_idx:array), or all components (y_idx:None)
        - using a predifined baseline (x_0:array), n baselines sampled from
        the dataloader (x_0:int), or initialized to a zero vector (x_0:None)
    """
    # Check x_size
    if isinstance(x_size, int):
        x_size = (x_size,)
    # Prepare y_idx
    n_y_idx, y_idx = _prepare_y_idx(y_idx, y_size, device)
    # Define x_seed
    x_seed = None
    if n_x is not None:
        x_seed = int(1e9 + x_0_seed)
    # Prepare outputs
    corr = np.zeros(n_y_idx, dtype=np.float32)
    bsl_shap_corr_ = np.zeros((n_y_idx,) + x_size, dtype=np.float32)
    # Iterate over y_idx
    for i, j in enumerate(y_idx):
        # Compute baseline Shapley
        _, y, y_0, y_r, bsl_shap_ = bsl_shap(
            forward_func, n_x, j, x_0, n_iter, x_size, y_size, dtld_func,
            x_0_batch_size, x_seed, x_0_seed, dtype, device, dtld_kwargs,
            forward_kwargs, check_error=False,
            description=f'bsc {i+1}/{n_y_idx}')
        y, y_0, y_r, bsl_shap_ = y[:, 0], y_0[:, 0], y_r[:, 0], bsl_shap_[:, 0]
        # Compute output correlation
        if check_error:
            corr[i] += pearsonr(y_r, y)[0]
        # Compute baseline Shapley correlation
        mu_y, std_y, std_y_r = np.mean(y), np.std(y), np.std(y_r)
        bsl_shap_corr_i = np.mean(
            bsl_shap_ * (y - mu_y)[(...,) + (None,)*len(x_size)], axis=0)
        bsl_shap_corr_i /= std_y
        bsl_shap_corr_i /= std_y_r
        bsl_shap_corr_[i] += bsl_shap_corr_i
    # Check baseline Shapley correlation error
    if check_error:
        bsc_sum = np.sum(np.reshape(bsl_shap_corr_, (n_y_idx, -1)), axis=1)
        print(f'error : {bsc_sum - corr}')
    return bsl_shap_corr_
