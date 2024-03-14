"""
Integrated Gradient Correlation (IGC) utils.

For optimization purposes, dataloader 'dtld_func', and gradient 'grad_func'
functions require specific patterns. See below for generic examples.

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

———————— grad_func

import torch
from torch import nn


class Model:
    def __init__(self, ...):
        self.module = Module(...)  # Inherited from nn.Module

    def grad_func(self, x, y_idx):
        # Prepare x
        x.requires_grad_(True)
        # Prepare model
        self.module.eval()
        self.module.zero_grad(set_to_none=True)
        # Eval and backprop
        y_r = self.module(x)
        y_r = torch.gather(y_r, dim=1, index=y_idx.unsqueeze(dim=1))
        y_r.backward(gradient=torch.ones_like(y_r))
        return y_r.squeeze(dim=1).detach().cpu().numpy(), x.grad.cpu().numpy()

————————

Author: Pierre Lelievre
"""

import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr


# Utils


def _iter_y_idx(y_size, y_idx):
    if y_idx is None:
        y_idx = range(y_size)
    elif isinstance(y_idx, int):
        y_size = 1
        y_idx = (y_idx,)
    else:
        y_idx = np.array(y_idx, dtype=np.int64).tolist()
        y_size = len(y_idx)
    return y_size, y_idx


@torch.no_grad()
def _prepare_y_idx(y_size, y_idx, device='cpu'):
    if y_idx is None:
        y_idx = torch.arange(y_size, dtype=torch.int64, device=device)
    elif isinstance(y_idx, int):
        y_size = 1
        y_idx = torch.full((y_size,), y_idx, dtype=torch.int64, device=device)
    else:
        y_idx = torch.tensor(y_idx, dtype=torch.int64, device=device)
        y_size = y_idx.size(0)
    return y_size, y_idx


@torch.no_grad()
def _prepare_x(x, y_size, dtype=torch.float32, device='cpu', unsqueeze=False):
    x = torch.as_tensor(x, dtype=dtype, device=device)
    if unsqueeze:
        x = x.unsqueeze(dim=0)
    repeat = (y_size,) + (1,)*(x.dim()-1)
    return x.repeat(*repeat)


@torch.no_grad()
def _prepare_x_0_dtld(dtld_func, x_0, x_size, n_x_0_per_batch=None, seed=100,
                      dtype=torch.float32, device='cpu', dtld_kwargs=None):
    # Random baseline (x_0 is n_x_0)
    if isinstance(x_0, int):
        if n_x_0_per_batch is None:
            n_x_0_per_batch = x_0
        n_x_0_per_batch = min(x_0, max(1, n_x_0_per_batch))
        n_batch = max(1, int(np.ceil(x_0 / n_x_0_per_batch)))
        if dtld_kwargs is None:
            dtld_kwargs = {}
        x_0_dtld = dtld_func(
            batch_size=n_x_0_per_batch, seed=seed, **dtld_kwargs)
        assert len(x_0_dtld) > n_batch
        return x_0_dtld, n_batch, n_x_0_per_batch
    # Predefined baseline
    if x_0 is None:
        x_0 = torch.zeros(1, *x_size, dtype=dtype, device=device)
    elif not torch.is_tensor(x_0):
        x_0 = torch.as_tensor(x_0, dtype=dtype, device=device)
    if x_0.dim() == len(x_size):
        x_0 = x_0.unsqueeze(dim=0)
    assert x_0.size()[1:] == x_size, 'Incompatible x and x_0 shapes.'
    return [(x_0, None)], 1, x_0.size(0)


@torch.no_grad()
def _record_x_y(x, y, y_idx):
    x = x.squeeze(dim=0).cpu().numpy()
    y = torch.gather(y.squeeze(dim=0), dim=0, index=y_idx.cpu()).cpu().numpy()
    return x, y


def _set_dtld_seed(dtld, seed):
    """
    Update dataloader seeds after initialization.
    """
    dtld.generator.manual_seed(seed)
    if dtld.dataset.rng is not None:
        dtld.dataset.rng.manual_seed(seed)


# Gradients


def grad_1_x(grad_func, x, y_size, y_idx=None, dtype=torch.float32,
             device='cpu', grad_kwargs=None):
    """
    Compute gradients:
        - for one input (x:array)
        - w.r.t. one specific output component index (y_idx:int), multiple
        component indices (y_idx:array), or all components (y_idx:None)
    """
    if grad_kwargs is None:
        grad_kwargs = {}
    # Prepare y_idx
    y_size, y_idx = _prepare_y_idx(y_size, y_idx, device)
    # Prepare x
    x = _prepare_x(x, y_size, dtype, device, unsqueeze=True)
    # Compute gradients
    return grad_func(x=x, y_idx=y_idx, **grad_kwargs)


def grad_dtld(dtld_func, grad_func, x_size, y_size, n_samples=None, y_idx=None,
              seed=None, dtype=torch.float32, device='cpu', dtld_kwargs=None,
              grad_kwargs=None):
    """
    Compute gradients:
        - for n inputs sampled from the dataloader (n_samples:int),
        or all inputs (n_samples:None)
        - w.r.t. one specific output component index (y_idx:int), multiple
        component indices (y_idx:array), or all components (y_idx:None)
    """
    if isinstance(x_size, int):
        x_size = (x_size,)
    if dtld_kwargs is None:
        dtld_kwargs = {}
    if grad_kwargs is None:
        grad_kwargs = {}
    # Prepare dataloader
    dtld = dtld_func(batch_size=1, seed=seed, **dtld_kwargs)
    if n_samples is None:
        n_samples = len(dtld)
    # Prepare y_idx
    y_size, y_idx = _prepare_y_idx(y_size, y_idx, device)
    # Prepare outputs
    x = np.zeros((n_samples,) + x_size, dtype=np.float32)
    y = np.zeros((n_samples, y_size), dtype=np.float32)
    y_r = np.zeros((n_samples, y_size), dtype=np.float32)
    grad = np.zeros((n_samples, y_size) + x_size, dtype=np.float32)
    # Iterate over dataset samples
    for i, (x_i, y_i) in enumerate(dtld):
        # Break when n_samples is reached
        if i == n_samples:
            break
        # Record x, y
        x_i_np, y_i_np = _record_x_y(x_i, y_i, y_idx)
        x[i] += x_i_np
        y[i] += y_i_np
        # Prepare x
        x_i = _prepare_x(x_i, y_size, dtype, device)
        # Compute gradients
        y_r_i, grad_i = grad_func(x=x_i, y_idx=y_idx, **grad_kwargs)
        y_r[i] += y_r_i
        grad[i] += grad_i
    return x, y, y_r, grad


# Integrated gradients


def _int_grad_1_x_1_y(x_0_dtld, grad_func, x, y_idx, n_batch, n_steps,
                      n_x_0_per_batch, dtype=torch.float32, device='cpu',
                      check_error=False, grad_kwargs=None):
    if grad_kwargs is None:
        grad_kwargs = {}
    with torch.no_grad():
        # Prepare inputs
        repeat = (n_x_0_per_batch,) + (1,)*x.dim()
        x = x.unsqueeze(dim=0).repeat(*repeat).to(device)
        y_idx = y_idx.repeat(n_steps * n_x_0_per_batch).to(device)
    # Compute integrated gradients
    y_0 = 0.0
    y_r = 0.0
    int_grad = np.zeros(x.size()[1:], dtype=np.float32)
    for i, (x_0_i, _) in enumerate(x_0_dtld):
        # Break when n_batch is reached
        if i == n_batch:
            break
        # Send baseline to device
        x_0_i = x_0_i.to(device)
        # Generate inputs along a linear path between x_0 and x
        with torch.no_grad():
            w = torch.linspace(
                0.0, 1.0, n_steps, dtype=dtype,
                device=device)[(...,) + (None,) * x.dim()]
            x_s_i = (1.0 - w) * x_0_i.unsqueeze(dim=0)\
                + w * x.unsqueeze(dim=0)
            x_s_i = x_s_i.flatten(0, 1)
        # Compute predictions and gradients
        y_r_i, grads_i = grad_func(x=x_s_i, y_idx=y_idx, **grad_kwargs)
        y_r_i = y_r_i.reshape((n_steps, n_x_0_per_batch))
        grads_i = grads_i.reshape((n_steps,) + x.size())
        # Compute integrated gradients (Riemann sums, trapezoidal rule)
        y_0 += np.sum(y_r_i[0])
        y_r += np.sum(y_r_i[-1])
        int_grad_i = grads_i[:-1] + grads_i[1:]
        int_grad_i = 0.5 * np.mean(int_grad_i, axis=0)
        int_grad_i *= (x - x_0_i).cpu().numpy()
        int_grad += np.sum(int_grad_i, axis=0)
    # Normalize across baselines
    n_x_0 = float(n_batch * n_x_0_per_batch)
    y_0 /= n_x_0
    y_r /= n_x_0
    int_grad /= n_x_0
    # Check integrated gradients error
    if check_error:
        print(f'error: {np.sum(int_grad) - y_r + y_0}')
    return y_0, y_r, int_grad


def int_grad_1_x(dtld_func, grad_func, x, y_size, y_idx=None, x_0=None,
                 n_steps=32, n_x_0_per_batch=None, x_0_seed=100,
                 dtype=torch.float32, device='cpu', check_error=False,
                 dtld_kwargs=None, grad_kwargs=None):
    """
    Compute integrated gradients:
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
    int_grad = np.zeros(x.size(), dtype=np.float32)
    # Iterate over y
    for i in tqdm(range(y_size), total=y_size, desc='ig'):
        y_0_i, y_r_i, int_grad_i = _int_grad_1_x_1_y(
            x_0_dtld, grad_func, x[i], y_idx[i], n_batch, n_steps,
            n_x_0_per_batch, dtype, device, check_error, grad_kwargs)
        y_0[i] += y_0_i
        y_r[i] += y_r_i
        int_grad[i] += int_grad_i
    return y_0, y_r, int_grad


def int_grad_dtld(dtld_func, grad_func, x_size, y_size, n_samples=None,
                  y_idx=None, x_0=None, n_steps=32, n_x_0_per_batch=None,
                  seed=None, x_0_seed=100, dtype=torch.float32, device='cpu',
                  check_error=False, dtld_kwargs=None, grad_kwargs=None,
                  description=''):
    """
    Compute integrated gradients:
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
    int_grad = np.zeros((n_samples, y_size) + x_size, dtype=np.float32)
    # Iterate over dataset samples
    if not description:
        description = 'ig'
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
            y_0_ij, y_r_ij, int_grad_ij = _int_grad_1_x_1_y(
                x_0_dtld, grad_func, x_i[0], y_idx[j], n_batch, n_steps,
                n_x_0_per_batch, dtype, device, check_error, grad_kwargs)
            y_0[i, j] += y_0_ij
            y_r[i, j] += y_r_ij
            int_grad[i, j] += int_grad_ij
    return x, y, y_0, y_r, int_grad


# Integrated gradient correlation


def int_grad_corr_dtld(dtld_func, grad_func, x_size, y_size, y_idx=None,
                       x_0=None, n_steps=32, n_x_0_per_batch=None,
                       x_0_seed=100, dtype=torch.float32, device='cpu',
                       check_error=False, dtld_kwargs=None, grad_kwargs=None):
    """
    Compute integrated gradient correlation:
        - for all inputs from the dataloader
        - w.r.t. one specific output component index (y_idx:int), multiple
        component indices (y_idx:array), or all components (y_idx:None)
        - using a predifined baseline (x_0:array), n baselines sampled from
        the dataloader (x_0:int), or initialized to a zero vector (x_0:None).
    """
    if isinstance(x_size, int):
        x_size = (x_size,)
    y_size, y_idx = _iter_y_idx(y_size, y_idx)
    # Prepare outputs
    corr = np.zeros(y_size, dtype=np.float32)
    int_grad_corr = np.zeros((y_size,) + x_size, dtype=np.float32)
    # Iterate over y
    for i, j in enumerate(y_idx):
        _, y, y_0, y_r, int_grad = int_grad_dtld(
            dtld_func, grad_func, x_size, y_size, None, j, x_0, n_steps,
            n_x_0_per_batch, None, x_0_seed, dtype, device, False, dtld_kwargs,
            grad_kwargs, description=f'igc {i+1}/{y_size}')
        y, y_0, y_r, int_grad = y[:, 0], y_0[:, 0], y_r[:, 0], int_grad[:, 0]
        # Compute output correlation
        if check_error:
            corr[i] += pearsonr(y_r, y)[0]
        # Compute integrated gradient correlation
        mu_y, std_y, std_y_r = np.mean(y), np.std(y), np.std(y_r)
        int_grad_corr_i = np.mean(
            int_grad * (y - mu_y)[(...,) + (None,)*len(x_size)], axis=0)
        int_grad_corr_i /= std_y
        int_grad_corr_i /= std_y_r
        int_grad_corr[i] += int_grad_corr_i
    # Check integrated gradient correlation error
    if check_error:
        error = np.sum(np.reshape(int_grad_corr, (y_size, -1)), axis=1) - corr
        print(f'error : {error}')
    return int_grad_corr
