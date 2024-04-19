"""
Integrated Gradient Correlation (IGC) utils.

For optimization purposes, dataloader 'dtld_func' and gradient 'grad_func'
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


@torch.no_grad()
def _prepare_y_idx(y_idx, y_size, device, x_batch_size=None):
    # All components
    if y_idx is None:
        assert y_size is not None, 'y_size must be defined if y_idx is None.'
        n_y_idx = y_size
        y_idx = torch.arange(y_size, dtype=torch.int64, device=device)
    # One specific component
    elif isinstance(y_idx, int):
        n_y_idx = 1
        y_idx = torch.full((1,), y_idx, dtype=torch.int64, device=device)
    # Multiple components
    else:
        y_idx = torch.as_tensor(y_idx, dtype=torch.int64, device=device)
        n_y_idx = list(y_idx.size())
        if n_y_idx:
            n_y_idx = n_y_idx[0]
        else:
            n_y_idx = 1
    # Repeat to match x_batch_size if defined
    if x_batch_size is not None:
        y_idx = y_idx.repeat(x_batch_size)
    return n_y_idx, y_idx


@torch.no_grad()
def _prepare_x_dtld(x, x_size, dtld_func, x_batch_size, x_seed, dtype, device,
                    dtld_kwargs):
    # x sampled from the dataloader (x is n_x, or None)
    if (x is None) or isinstance(x, int):
        # Check x_size
        assert x_size is not None, 'x_size must be defined if x is sampled.'
        if isinstance(x_size, int):
            x_size = (x_size,)
        # Init x dataloader
        assert dtld_func is not None, (
            'dtld_func must be defined if x is sampled.')
        if dtld_kwargs is None:
            dtld_kwargs = {}
        x_dtld = dtld_func(batch_size=x_batch_size, seed=x_seed, **dtld_kwargs)
        # Compute n_x and n_x_batch
        n_x_batch_max = max(
            1, int(np.floor(len(x_dtld.dataset) / x_batch_size)))
        n_x_max = n_x_batch_max * x_batch_size
        if x is None:
            n_x = n_x_max
        else:
            n_x = max(x_batch_size, min(x, n_x_max))
        n_x_batch = n_x // x_batch_size
        return x_dtld, n_x, n_x_batch, x_size
    # Predefined x
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=dtype, device=device)
    # Check x_batch_size
    n_x = x.size(0)
    assert not n_x % x_batch_size, 'n_x must be a multiple of x_batch_size.'
    # Compute n_x_batch
    n_x_batch = n_x // x_batch_size
    # Build x_dtld
    x_dtld = [
        (x[i*x_batch_size:(i+1)*x_batch_size], None) for i in range(n_x_batch)]
    return x_dtld, n_x, n_x_batch, x.size()[1:]


@torch.no_grad()
def _prepare_x_0_dtld(x_0, x_size, dtld_func, x_batch_size, x_0_batch_size,
                      x_0_seed, dtype, device, dtld_kwargs):
    # Random baselines (x_0 is n_x_0)
    if isinstance(x_0, int):
        assert dtld_func is not None, (
            'dtld_func must be defined if x_0 is sampled.')
        if x_0_batch_size is None:
            x_0_batch_size = x_0
        x_0_batch_size = min(x_0, max(1, x_0_batch_size))
        n_x_0_batch = max(1, int(np.ceil(x_0 / x_0_batch_size)))
        if dtld_kwargs is None:
            dtld_kwargs = {}
        x_0_dtld = dtld_func(
            batch_size=x_0_batch_size * x_batch_size, seed=x_0_seed,
            **dtld_kwargs)
        assert len(x_0_dtld) >= n_x_0_batch
        return x_0_dtld, n_x_0_batch, x_0_batch_size
    # Zero baseline
    if x_0 is None:
        x_0 = torch.zeros(x_batch_size, *x_size, dtype=dtype, device=device)
        return [(x_0, None)], 1, 1
    # Predefined baseline
    if not torch.is_tensor(x_0):
        x_0 = torch.as_tensor(x_0, dtype=dtype, device=device)
    if x_0.dim() == len(x_size):
        x_0 = x_0.unsqueeze(dim=0)
    x_0_batch_size = x_0.size(0)
    x_0 = x_0.repeat_interleave(x_batch_size, dim=0)
    assert x_0.size()[1:] == x_size, 'Incompatible x and x_0 shapes.'
    return [(x_0, None)], 1, x_0_batch_size


@torch.no_grad()
def _record_x_y(x, y, y_idx, x_batch_size):
    if y is not None:
        y = torch.gather(
            y.cpu(), dim=1, index=y_idx.view(x_batch_size, -1).cpu()).numpy()
    return x.cpu().numpy(), y


def _set_dtld_seed(dtld, seed):
    """
    Update dataloader seeds after initialization.
    """
    dtld.generator.manual_seed(seed)
    if dtld.dataset.rng is not None:
        dtld.dataset.rng.manual_seed(seed)


# Gradients


def grad(grad_func, x=None, y_idx=None, x_size=None, y_size=None,
         dtld_func=None, x_batch_size=1, x_seed=None, dtype=torch.float32,
         device='cpu', dtld_kwargs=None, grad_kwargs=None):
    """
    Compute gradients (grad:array [n_x, n_y_idx, ...]):
        - for all inputs (x:array [n_x, ...]), for n_x inputs sampled from
        the dataloader (x:int), or all inputs of the dataloader (x:None)
        - w.r.t. one specific output component index (y_idx:int), multiple
        component indices (y_idx:array), or all components (y_idx:None)
    """
    if grad_kwargs is None:
        grad_kwargs = {}
    # Prepare x dataloader
    x_dtld, n_x, n_x_batch, x_size = _prepare_x_dtld(
        x, x_size, dtld_func, x_batch_size, x_seed, dtype, device, dtld_kwargs)
    # Prepare y_idx
    n_y_idx, y_idx = _prepare_y_idx(y_idx, y_size, device, x_batch_size)
    # Prepare outputs
    x_np = np.zeros((n_x,) + x_size, dtype=np.float32)
    y_np = np.zeros((n_x, n_y_idx), dtype=np.float32)
    y_r = np.zeros((n_x, n_y_idx), dtype=np.float32)
    grad_ = np.zeros((n_x, n_y_idx) + x_size, dtype=np.float32)
    # Iterate over dataloader
    for i, (x_i, y_i) in enumerate(x_dtld):
        # Break when n_x_batch is reached
        if i == n_x_batch:
            break
        # Current slice
        slc = slice(i*x_batch_size, (i+1)*x_batch_size)
        # Record x, y
        x_i_np, y_i_np = _record_x_y(x_i, y_i, y_idx, x_batch_size)
        x_np[slc] += x_i_np
        if y_i_np is not None:
            y_np[slc] += y_i_np
        # Prepare x
        x_i = x_i.repeat_interleave(n_y_idx, dim=0).to(device)
        # Compute gradients
        y_r_i, grad_i = grad_func(x=x_i, y_idx=y_idx, **grad_kwargs)
         # Reshape
        y_r_i = y_r_i.reshape((x_batch_size, -1))
        grad_i = grad_i.reshape((x_batch_size, -1) + grad_i.shape[1:])
        # Record results
        y_r[slc] += y_r_i
        grad_[slc] += grad_i
    return x_np, y_np, y_r, grad_


# Integrated gradients


def _int_grad_1_y_idx(grad_func, x, y_idx, x_0_dtld, n_steps, x_batch_size,
                      n_x_0_batch, x_0_batch_size, dtype, device, grad_kwargs,
                      check_error):
    with torch.no_grad():
        # Prepare inputs
        repeat = (x_0_batch_size,) + (1,)*(x.dim()-1)
        x = x.repeat(*repeat).to(device)
        y_idx = y_idx.expand(x_batch_size*x_0_batch_size*n_steps).to(device)
        # Prepare interpolation coefficients
        w = torch.linspace(0.0, 1.0, n_steps, dtype=dtype, device=device)[
            (...,) + (None,) * x.dim()]
    # Prepare outputs
    y_0 = np.zeros(x_batch_size, dtype=np.float32)
    y_r = np.zeros(x_batch_size, dtype=np.float32)
    int_grad_ = np.zeros((x_batch_size,) + x.size()[1:], dtype=np.float32)
    # Iterate over baselines
    for i, (x_0_i, _) in enumerate(x_0_dtld):
        # Break when n_x_0_batch is reached
        if i == n_x_0_batch:
            break
        # Send baselines to the device
        x_0_i = x_0_i.to(device)
        # Generate inputs along a linear path between x_0 and x
        with torch.no_grad():
            x_s_i = (1.0 - w)*x_0_i.unsqueeze(dim=0) + w*x.unsqueeze(dim=0)
            x_s_i = x_s_i.flatten(0, 1)
        # Compute predictions and gradients
        y_r_i, grad_i = grad_func(x=x_s_i, y_idx=y_idx, **grad_kwargs)
        y_r_i = y_r_i.reshape((n_steps, x_0_batch_size, x_batch_size))
        grad_i = grad_i.reshape(
            (n_steps, x_0_batch_size, x_batch_size) + x.size()[1:])
        # Compute integrated gradients (Riemann sums, trapezoidal rule)
        y_0 += np.sum(y_r_i[0], axis=0)
        y_r += np.sum(y_r_i[-1], axis=0)
        int_grad_i = grad_i[:-1] + grad_i[1:]
        int_grad_i = 0.5 * np.mean(int_grad_i, axis=0)
        int_grad_i *= (x - x_0_i).cpu().numpy().reshape(
            (x_0_batch_size, x_batch_size) + x.size()[1:])
        int_grad_ += np.sum(int_grad_i, axis=0)
    # Average baselines
    n_x_0 = float(n_x_0_batch * x_0_batch_size)
    y_0 /= n_x_0
    y_r /= n_x_0
    int_grad_ /= n_x_0
    # Check integrated gradients error
    if check_error:
        int_grad_sum = np.sum(int_grad_.reshape((x_batch_size, -1)), axis=1)
        print(f'error: {int_grad_sum - y_r + y_0}')
    return y_0, y_r, int_grad_


def int_grad(grad_func, x=None, y_idx=None, x_0=None, n_steps=32, x_size=None,
             y_size=None, dtld_func=None, x_batch_size=1, x_0_batch_size=None,
             x_seed=None, x_0_seed=100, dtype=torch.float32, device='cpu',
             dtld_kwargs=None, grad_kwargs=None, check_error=False,
             description=''):
    """
    Compute integrated gradients (int_grad:array [n_x, n_y_idx, ...]):
        - for all inputs (x:array [n_x, ...]), for n_x inputs sampled from
        the dataloader (x:int), or all inputs of the dataloader (x:None)
        - w.r.t. one specific output component index (y_idx:int), multiple
        component indices (y_idx:array), or all components (y_idx:None)
        - using a predifined baseline (x_0:array), n baselines sampled from
        the dataloader (x_0:int), or initialized to a zero vector (x_0:None)
    """
    if grad_kwargs is None:
        grad_kwargs = {}
    # Prepare x dataloader
    x_dtld, n_x, n_x_batch, x_size = _prepare_x_dtld(
        x, x_size, dtld_func, x_batch_size, x_seed, dtype, device, dtld_kwargs)
    # Prepare y_idx
    n_y_idx, y_idx = _prepare_y_idx(y_idx, y_size, device, x_batch_size)
    # Prepare x_0 dataloader
    x_0_dtld, n_x_0_batch, x_0_batch_size = _prepare_x_0_dtld(
        x_0, x_size, dtld_func, x_batch_size, x_0_batch_size, x_0_seed, dtype,
        device, dtld_kwargs)
    # Prepare outputs
    x_np = np.zeros((n_x,) + x_size, dtype=np.float32)
    y_np = np.zeros((n_x, n_y_idx), dtype=np.float32)
    y_0 = np.zeros((n_x, n_y_idx), dtype=np.float32)
    y_r = np.zeros((n_x, n_y_idx), dtype=np.float32)
    int_grad_ = np.zeros((n_x, n_y_idx) + x_size, dtype=np.float32)
    # Iterate over dataloader
    if not description:
        description = 'ig'
    for i, (x_i, y_i) in enumerate(
            tqdm(x_dtld, total=n_x_batch, desc=description)):
        # Break when n_x_batch is reached
        if i == n_x_batch:
            break
        # Current slice
        slc = slice(i*x_batch_size, (i+1)*x_batch_size)
        # Record x, y
        x_i_np, y_i_np = _record_x_y(x_i, y_i, y_idx, x_batch_size)
        x_np[slc] += x_i_np
        if y_i_np is not None:
            y_np[slc] += y_i_np
        # Iterate over y_idx
        if not isinstance(x_0_dtld, list):
            _set_dtld_seed(x_0_dtld, int(x_0_seed+1e6*(i+1)))
        for j in range(n_y_idx):
            y_0_ij, y_r_ij, int_grad_ij = _int_grad_1_y_idx(
                grad_func, x_i, y_idx[j], x_0_dtld, n_steps, x_batch_size,
                n_x_0_batch, x_0_batch_size, dtype, device, grad_kwargs,
                check_error)
            y_0[slc, j] += y_0_ij
            y_r[slc, j] += y_r_ij
            int_grad_[slc, j] += int_grad_ij
    return x_np, y_np, y_0, y_r, int_grad_


# Integrated gradient correlation


def int_grad_corr(grad_func, dtld_func, x_size, y_size, y_idx=None, x_0=None,
                  n_steps=32, x_batch_size=1, x_0_batch_size=None,
                  x_0_seed=100, dtype=torch.float32, device='cpu',
                  dtld_kwargs=None, grad_kwargs=None, check_error=False):
    """
    Compute integrated gradient correlation (igc:array [n_y_idx, ...]):
        - for all inputs of the dataloader
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
    # Prepare outputs
    corr = np.zeros(n_y_idx, dtype=np.float32)
    int_grad_corr_ = np.zeros((n_y_idx,) + x_size, dtype=np.float32)
    # Iterate over y_idx
    for i, j in enumerate(y_idx):
        # Compute integrated gradients
        x, x_seed = None, None
        _, y, y_0, y_r, int_grad_ = int_grad(
            grad_func, x, j, x_0, n_steps, x_size, y_size, dtld_func,
            x_batch_size, x_0_batch_size, x_seed, x_0_seed, dtype, device,
            dtld_kwargs, grad_kwargs, check_error=False,
            description=f'igc {i+1}/{n_y_idx}')
        y, y_0, y_r, int_grad_ = y[:, 0], y_0[:, 0], y_r[:, 0], int_grad_[:, 0]
        # Compute output correlation
        if check_error:
            corr[i] += pearsonr(y_r, y)[0]
        # Compute integrated gradient correlation
        mu_y, std_y, std_y_r = np.mean(y), np.std(y), np.std(y_r)
        int_grad_corr_i = np.mean(
            int_grad_ * (y - mu_y)[(...,) + (None,)*len(x_size)], axis=0)
        int_grad_corr_i /= std_y
        int_grad_corr_i /= std_y_r
        int_grad_corr_[i] += int_grad_corr_i
    # Check integrated gradient correlation error
    if check_error:
        igc_sum = np.sum(np.reshape(int_grad_corr_, (n_y_idx, -1)), axis=1)
        print(f'error : {igc_sum - corr}')
    return int_grad_corr_
