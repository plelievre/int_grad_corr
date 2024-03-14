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
    _iter_y_idx, _prepare_y_idx, _prepare_x, _prepare_x_0_dtld, _record_x_y,
    _set_dtld_seed)


# Baseline Shapley


@torch.no_grad()
def _bsl_shap_1_x_1_y(x_0_dtld, forward_func, x, y_idx, n_batch, n_iter,
                      n_x_0_per_batch, perm_seed=100, device='cpu',
                      check_error=False, forward_kwargs=None):
    if forward_kwargs is None:
        forward_kwargs = {}
    n_features = x.numel()
    batch_size = n_x_0_per_batch * n_iter
    # Prepare inputs
    repeat_iter = (n_iter,) + (1,)*x.dim()
    repeat_batch = (batch_size,) + (1,)*x.dim()
    x = x.unsqueeze(dim=0).repeat(*repeat_batch).to(device)
    y_idx = y_idx.repeat(batch_size).to(device)
    # Compute baseline Shapley
    y_0 = 0.0
    y_r = forward_func(
        x=x[0].unsqueeze(dim=0), y_idx=y_idx[0].unsqueeze(dim=0),
        **forward_kwargs)[0]
    bsl_shap = np.zeros(x.size()[1:], dtype=np.float32)
    for i, (x_0_i, _) in enumerate(x_0_dtld):
        # Break when n_batch is reached
        if i == n_batch:
            break
        # Send baseline to device
        x_0_i = x_0_i.to(device)
        # Compute y_0_i
        y_0_i = forward_func(
            x=x_0_i, y_idx=y_idx[:n_x_0_per_batch], **forward_kwargs)
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
            bsl_shap += np.sum(eval_diff[(...,) + (None,)*(
                x.dim()-1)] * mask_j.cpu().numpy(), axis=0)
            prev_eval = modified_eval
    # Normalize across baselines (and iterations)
    n_x_0 = float(n_batch * n_x_0_per_batch)
    y_0 /= n_x_0
    bsl_shap /= n_x_0 * n_iter
    # Check baseline Shapley error
    if check_error:
        print(f'error: {np.sum(bsl_shap) - y_r + y_0}')
    return y_0, y_r, bsl_shap


@torch.no_grad()
def bsl_shap_1_x(dtld_func, forward_func, x, y_size, y_idx=None, x_0=None,
                 n_iter=32, n_x_0_per_batch=None, x_0_seed=100,
                 dtype=torch.float32, device='cpu', check_error=False,
                 dtld_kwargs=None, forward_kwargs=None):
    """
    Compute baseline Shapley:
        - for one input (x:array)
        - w.r.t. one specific output component index (y_idx:int), multiple
        component indices (y_idx:array), or all components (y_idx:None)
        - using a predifined baseline (x_0:array), n baselines sampled from
        the dataloader (x_0:int), or initialized to a zero vector (x_0:None).
    """
    # Prepare y_idx
    y_size, y_idx = _prepare_y_idx(y_size, y_idx, device)
    # Prepare x
    x = _prepare_x(x, y_size, dtype, device, unsqueeze=True)
    # Prepare x_0 (baseline) dataloader
    x_0_dtld, n_batch, n_x_0_per_batch = _prepare_x_0_dtld(
        dtld_func, x_0, x.size()[1:], n_x_0_per_batch, x_0_seed, dtype, device,
        dtld_kwargs)
    # Prepare outputs
    y_0 = np.zeros(y_size, dtype=np.float32)
    y_r = np.zeros(y_size, dtype=np.float32)
    bsl_shap = np.zeros(x.size(), dtype=np.float32)
    # Iterate over y
    for i in tqdm(range(y_size), total=y_size, desc='bs'):
        y_0_i, y_r_i, bsl_shap_i = _bsl_shap_1_x_1_y(
            x_0_dtld, forward_func, x[i], y_idx[i], n_batch, n_iter,
            n_x_0_per_batch, x_0_seed, device, check_error, forward_kwargs)
        y_0[i] += y_0_i
        y_r[i] += y_r_i
        bsl_shap[i] += bsl_shap_i
    return y_0, y_r, bsl_shap


@torch.no_grad()
def bsl_shap_dtld(dtld_func, forward_func, x_size, y_size, n_samples=None,
                  y_idx=None, x_0=32, n_iter=32, n_x_0_per_batch=None,
                  seed=None, x_0_seed=100, dtype=torch.float32, device='cpu',
                  check_error=False, dtld_kwargs=None, forward_kwargs=None,
                  description=''):
    """
    Compute baseline Shapley:
        - for n inputs sampled from the dataloader (n_samples:int),
        or all inputs (n_samples:None)
        - w.r.t. one specific output component index (y_idx:int), multiple
        component indices (y_idx:array), or all components (y_idx:None)
        - using a predifined baseline (x_0:array), n baselines sampled from
        the dataloader (x_0:int), or initialized to a zero vector (x_0:None).
    """
    if isinstance(x_size, int):
        x_size = (x_size,)
    if dtld_kwargs is None:
        dtld_kwargs = {}
    # Prepare dataloader
    dtld = dtld_func(batch_size=1, seed=seed, **dtld_kwargs)
    if n_samples is None:
        n_samples = len(dtld)
    # Prepare y_idx
    y_size, y_idx = _prepare_y_idx(y_size, y_idx, device)
    # Prepare x_0 (baseline) dataloader
    x_0_dtld, n_batch, n_x_0_per_batch = _prepare_x_0_dtld(
        dtld_func, x_0, x_size, n_x_0_per_batch, x_0_seed, dtype, device,
        dtld_kwargs)
    # Prepare outputs
    x = np.zeros((n_samples,) + x_size, dtype=np.float32)
    y = np.zeros((n_samples, y_size), dtype=np.float32)
    y_0 = np.zeros((n_samples, y_size), dtype=np.float32)
    y_r = np.zeros((n_samples, y_size), dtype=np.float32)
    bsl_shap = np.zeros((n_samples, y_size) + x_size, dtype=np.float32)
    # Iterate over dataset samples
    if not description:
        description = 'bs'
    for i, (x_i, y_i) in enumerate(
            tqdm(dtld, total=n_samples, desc=description)):
        # Break when n_samples is reached
        if i == n_samples:
            break
        # Record x, y
        x_i_np, y_i_np = _record_x_y(x_i, y_i, y_idx)
        x[i] += x_i_np
        y[i] += y_i_np
        # Iterate over y
        if not isinstance(x_0_dtld, list):
            _set_dtld_seed(x_0_dtld, int(x_0_seed+1e6*(i+1)))
        for j in range(y_size):
            y_0_ij, y_r_ij, bsl_shap_ij = _bsl_shap_1_x_1_y(
                x_0_dtld, forward_func, x_i[0], y_idx[j], n_batch, n_iter,
                n_x_0_per_batch, x_0_seed, device, check_error, forward_kwargs)
            y_0[i, j] += y_0_ij
            y_r[i, j] += y_r_ij
            bsl_shap[i, j] += bsl_shap_ij
    return x, y, y_0, y_r, bsl_shap


# Baseline Shapley correlation


@torch.no_grad()
def bsl_shap_corr_dtld(dtld_func, forward_func, x_size, y_size, y_idx=None,
                       x_0=32, n_iter=32, n_x_0_per_batch=None, x_0_seed=100,
                       dtype=torch.float32, device='cpu', check_error=False,
                       dtld_kwargs=None, forward_kwargs=None, n_samples=None):
    """
    Compute baseline Shapley correlation:
        - for all inputs from the dataloader
        - w.r.t. one specific output component index (y_idx:int), multiple
        component indices (y_idx:array), or all components (y_idx:None)
        - using a predifined baseline (x_0:array), n baselines sampled from
        the dataloader (x_0:int), or initialized to a zero vector (x_0:None).
    """
    if isinstance(x_size, int):
        x_size = (x_size,)
    y_size, y_idx = _iter_y_idx(y_size, y_idx)
    seed = None
    if n_samples is not None:
        seed = int(1e9 + x_0_seed)
    # Prepare outputs
    corr = np.zeros(y_size, dtype=np.float32)
    bsl_shap_corr = np.zeros((y_size,) + x_size, dtype=np.float32)
    # Iterate over y
    for i, j in enumerate(y_idx):
        _, y, y_0, y_r, bsl_shap = bsl_shap_dtld(
            dtld_func, forward_func, x_size, y_size, n_samples, j, x_0,
            n_iter, n_x_0_per_batch, seed, x_0_seed, dtype, device, False,
            dtld_kwargs, forward_kwargs, description=f'bsc {i+1}/{y_size}')
        y, y_0, y_r, bsl_shap = y[:, 0], y_0[:, 0], y_r[:, 0], bsl_shap[:, 0]
        # Compute output correlation
        if check_error:
            corr[i] += pearsonr(y_r, y)[0]
        # Compute baseline Shapley correlation
        mu_y, std_y, std_y_r = np.mean(y), np.std(y), np.std(y_r)
        bsl_shap_corr_i = np.mean(
            bsl_shap * (y - mu_y)[(...,) + (None,)*len(x_size)], axis=0)
        bsl_shap_corr_i /= std_y
        bsl_shap_corr_i /= std_y_r
        bsl_shap_corr[i] += bsl_shap_corr_i
    # Check baseline Shapley correlation error
    if check_error:
        error = np.sum(np.reshape(bsl_shap_corr, (y_size, -1)), axis=1) - corr
        print(f'error : {error}')
    return bsl_shap_corr
