"""
Integrated Gradient Correlation (IGC).

For optimization purposes, dataloader 'dtld_func', forward function
'forward_func', backward function 'backward_func', embedding function
'embed_func', and IG post function (ig_post_func) require specific patterns.
See below for generic examples.

Welford's algorith is used for variance and covariance/correlation estimations.


———————— dtld_func


import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset


class _Dataset(TorchDataset):
    def __init__(self, x_1, x_2, x_cat, y, seed=None):
        self.x_1 = x_1  # Input 1
        self.x_2 = x_2  # Input 2
        self.x_cat = x_cat  # Input 3 (categorical)
        self.y = y  # Output
        self.rng = None
        if seed is not None:
            self.rng = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.x_1)

    def __getitem__(self, idx):
        x_1_i = self.x_1[idx]
        x_2_i = self.x_2[idx]
        x_cat_i = self.x_cat[idx]
        y_i = self.x[idx]
        # Use self.rng for data augmentation. Caution, self.rng may be None.
        return (x_1_i, x_2_i, x_cat_i), y_i


class Dataset:
    def __init__(self, x, y):
        self.x_1 = x_1  # Input 1
        self.x_2 = x_2  # Input 2
        self.x_cat = x_cat  # Input 3 (categorical)
        self.y = y  # Output

    @torch.no_grad()
    def dtld_func(self, batch_size, seed=None, num_workers=0):
        if seed is None:
            return DataLoader(
                _Dataset(self.x_1, self.x_2, self.x_cat, self.y),
                batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return DataLoader(
            _Dataset(self.x_1, self.x_2, self.x_cat, self.y, seed),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            generator=torch.Generator().manual_seed(seed))


———————— forward_func, backward_func, embed_func and ig_post_func


import torch
from torch import nn


class Model:
    def __init__(self, ...):
        self.module = Module(...)
        # Inherited from nn.Module, and with the forward method dividable in
        # 'embed' and 'fw_after_emd' methods.

    def embed_func(self, x1_x2_xcat):
        # Expand inputs
        x_1, x_2, x_cat = x1_x2_xcat
        # Get embedding
        x_cat = self.module.embed(x_cat)
        return x_1, x_2, x_cat

    def forward_func(self, x1_x2_xcat):
        # Expand inputs
        x_1, x_2, x_cat = x1_x2_xcat
        # Prepare inputs
        x_1.requires_grad_(True)
        x_2.requires_grad_(True)
        x_cat.requires_grad_(True)
        # Prepare model
        self.module.eval()
        # Eval
        y_r = self.module.fw_after_emd(x_1, x_2, x_cat)
        return y_r

    def backward_func(self, x1_x2_xcat, y_r, y_idx):
        # Expand inputs
        x_1, x_2, x_cat = x1_x2_xcat
        # Reset gradients
        if x_1.grad is not None:
            x_1.grad = None
        if x_2.grad is not None:
            x_2.grad = None
        if x_cat.grad is not None:
            x_cat.grad = None
        self.module.zero_grad(set_to_none=True)
        # Compute gradients
        y_r = torch.gather(y_r, dim=1, index=y_idx.unsqueeze(dim=1))
        y_r.backward(gradient=torch.ones_like(y_r), retain_graph=True)
        return y_r.squeeze(dim=1).detach().cpu().numpy(), (
            x_1.grad.cpu().numpy(), x_2.grad.cpu().numpy(),
            x_cat.grad.cpu().numpy())

    def ig_post_func(self, ig1_ig2_igcat, x1_x2_xcat_ori):
        # Expand integrated gradients and non-embedded inputs
        ig_x1, ig_x2, ig_cat = ig1_ig2_igcat
        _, _, x_cat = x1_x2_xcat_ori
        # Modify integrated gradients here
        return ig_x1, ig_x2, ig_cat


————————


Author: Pierre Lelievre
"""

import torch
import numpy as np
from tqdm import tqdm


# Utils


def _check_x_size(x_size):
    if isinstance(x_size, int):
        x_size = (x_size,)
    if not isinstance(x_size[0], tuple):
        x_size = (x_size,)
    return x_size


def _check_embed_size(embed_size, x_size):
    if embed_size is None:
        return x_size
    embed_size = _check_x_size(embed_size)
    assert len(embed_size) == len(x_size), (
        'Length of embed_size is different from x_size.')
    return embed_size


def _check_output_size(output_size, embed_size):
    if output_size is None:
        return embed_size
    output_size = _check_x_size(output_size)
    assert len(output_size) == len(embed_size), (
        'Length of output_size is different from x_size.')
    return output_size


def _check_dtype(dtype, x_size):
    if not isinstance(dtype, tuple):
        dtype = (dtype,)
    if (len(dtype) == 1) and len(dtype) != len(x_size):
        dtype = dtype * len(x_size)
    assert len(dtype) == len(x_size), (
        'Length of dtype is different from x_size.')
    # Define dtype for numpy arrays
    dtype_np = np.float64
    if (torch.float64 not in dtype) and (torch.float32 in dtype):
        dtype_np = np.float32
    return dtype, dtype_np


def _fwd_func_multi_x(func):
    def wrapper(*args, **kwargs):
        args = (args[0][0],) + args[1:]
        y_r = func(*args, **kwargs)
        return y_r
    return wrapper


def _bwd_func_multi_x(func):
    def wrapper(*args, **kwargs):
        args = (args[0][0],) + args[1:]
        y_r, grad_ = func(*args, **kwargs)
        return y_r, (grad_,)
    return wrapper


def _embed_func_multi_x(func):
    def wrapper(*args, **kwargs):
        args = (args[0][0],) + args[1:]
        x = func(*args, **kwargs)
        return (x,)
    return wrapper


def _ig_post_func_multi_x(func):
    def wrapper(*args, **kwargs):
        args = (args[0][0], args[1][0]) + args[1:]
        i_g = func(*args, **kwargs)
        return (i_g,)
    return wrapper


@torch.no_grad()
def _prepare_x_dtld(x, x_size, dtld_func, x_batch_size, x_seed, dtype, device,
                    dtld_kwargs):
    # x sampled from the dataloader (x is n_x, or None) #######################
    if (x is None) or isinstance(x, int):
        # Check x_size
        assert x_size is not None, 'x_size must be defined if x is sampled.'
        x_size = _check_x_size(x_size)
        # Define multi_x
        multi_x = len(x_size) > 1
        # Check dtype
        dtype, dtype_np = _check_dtype(dtype, x_size)
        # Init x dataloader
        assert dtld_func is not None, (
            'dtld_func must be defined if x is sampled.')
        if dtld_kwargs is None:
            dtld_kwargs = {}
        x_dtld = dtld_func(batch_size=x_batch_size, seed=x_seed, **dtld_kwargs)
        # Compute n_x and x_n_batch
        x_n_batch_max = max(
            1, int(np.floor(len(x_dtld.dataset) / x_batch_size)))
        n_x_max = x_n_batch_max * x_batch_size
        if x is None:
            n_x = n_x_max
        else:
            n_x = max(x_batch_size, min(x, n_x_max))
        x_n_batch = n_x // x_batch_size
        return x_dtld, n_x, x_n_batch, x_size, multi_x, dtype, dtype_np
    # Predefined x ############################################################
    # Define multi_x
    multi_x = True
    if not isinstance(x, tuple):
        multi_x = False
        x = (x,)
    # Build x_size
    x_size = tuple(x_i.shape[1:] for x_i in x)
    # Check dtype
    dtype, dtype_np = _check_dtype(dtype, x_size)
    # Send x to the device
    x = tuple(
        torch.as_tensor(x_i, dtype=dtype_i, device=device)\
            for x_i, dtype_i in zip(x, dtype))
    # Check x_batch_size
    n_x = x[0].size(0)
    assert not n_x % x_batch_size, 'n_x must be a multiple of x_batch_size.'
    # Compute x_n_batch
    x_n_batch = n_x // x_batch_size
    # Build x_dtld
    if multi_x:
        x_dtld = tuple(
            (tuple(x_i[i*x_batch_size:(i+1)*x_batch_size] for x_i in x), None)\
                for i in range(x_n_batch))
    else:
        x_dtld = tuple(
            (x[0][i*x_batch_size:(i+1)*x_batch_size], None)\
                for i in range(x_n_batch))
    return x_dtld, n_x, x_n_batch, x_size, multi_x, dtype, dtype_np


@torch.no_grad()
def _prepare_x_0_dtld(x_0, x_size, dtld_func, x_batch_size, x_0_batch_size,
                      x_0_seed, multi_x, dtype, device, dtld_kwargs):
    # Random baselines (x_0 is n_x_0) #########################################
    if isinstance(x_0, int):
        assert dtld_func is not None, (
            'dtld_func must be defined if x_0 is sampled.')
        x_0_batch_size = min(x_0, max(1, x_0_batch_size))
        x_0_n_batch = max(1, int(np.ceil(x_0 / x_0_batch_size)))
        if dtld_kwargs is None:
            dtld_kwargs = {}
        x_0_dtld = dtld_func(
            batch_size=x_0_batch_size * x_batch_size, seed=x_0_seed,
            **dtld_kwargs)
        assert len(x_0_dtld) >= x_0_n_batch
        n_x_0 = x_0_n_batch * x_0_batch_size
        return x_0_dtld, n_x_0, x_0_n_batch, x_0_batch_size
    # Zero baselines ##########################################################
    if x_0 is None and multi_x:
        x_0_dtld = ((tuple(
            torch.zeros(x_batch_size, *x_size_i, dtype=dtype_i, device=device)\
                for x_size_i, dtype_i in zip(x_size, dtype)), None),)
        return x_0_dtld, 1, 1, 1
    if x_0 is None:
        x_0_dtld = ((torch.zeros(
            x_batch_size, *x_size[0], dtype=dtype[0], device=device), None),)
        return x_0_dtld, 1, 1, 1
    # Uniform baselines #######################################################
    if isinstance(x_0, float) and multi_x:
        x_0_dtld = ((tuple(torch.full(
            (x_batch_size,) + x_size_i, x_0, dtype=dtype_i, device=device)\
                for x_size_i, dtype_i in zip(x_size, dtype)), None),)
        return x_0_dtld, 1, 1, 1
    if isinstance(x_0, float):
        x_0_dtld = ((torch.full(
            (x_batch_size,) + x_size[0], x_0, dtype=dtype[0],
            device=device), None),)
        return x_0_dtld, 1, 1, 1
    # Predefined baselines ####################################################
    # Comply with multi_x
    if not multi_x:
        x_0 = (x_0,)
    assert len(x_0) == len(x_size), 'x_0 and x number of items mismatch.'
    # Send baselines to the device
    x_0 = tuple(
        torch.as_tensor(x_0_i, dtype=dtype_i, device=device)\
            for x_0_i, dtype_i in zip(x_0, dtype))
    # Expand batch (first) dimension if necessary
    x_0 = tuple(
        x_0_i.unsqueeze(dim=0) if x_0_i.dim() == len(x_size_i) else x_0_i\
            for x_0_i, x_size_i in zip(x_0, x_size))
    # Check baseline sizes
    for x_0_i, x_size_i in zip(x_0, x_size):
        assert x_0_i.size()[1:] == x_size_i, 'Incompatible x and x_0 shapes.'
    # Define n_x_0
    n_x_0 = x_0[0].size(0)
    assert not n_x_0 % x_0_batch_size, (
        'n_x_0 must be a multiple of x_0_batch_size.')
    # Compute x_0_n_batch
    x_0_n_batch = n_x_0 // x_0_batch_size
    # Repeat along batch dimension
    x_0 = tuple(x_0_i.repeat_interleave(x_batch_size, dim=0) for x_0_i in x_0)
    # Build x_0_dtld
    b_sz = x_batch_size * x_0_batch_size
    if multi_x:
        x_0_dtld = tuple(
            (tuple(x_0_i[i*b_sz:(i+1)*b_sz] for x_0_i in x_0), None)\
                for i in range(x_0_n_batch))
    else:
        x_0_dtld = tuple(
            (x_0[0][i*b_sz:(i+1)*b_sz], None) for i in range(x_0_n_batch))
    return x_0_dtld, n_x_0, x_0_n_batch, x_0_batch_size


@torch.no_grad()
def _prepare_y_idx(y_idx, y_size, x_batch_size, y_batch_size, x_0_batch_size,
                   n_steps, device):
    # All components
    if y_idx is None:
        assert y_size is not None, 'y_size must be defined if y_idx is None.'
        n_y = y_size
        y_idx = torch.arange(y_size, dtype=torch.int64, device=device)
    # One specific component
    elif isinstance(y_idx, int):
        n_y = 1
        y_idx = torch.full((1,), y_idx, dtype=torch.int64, device=device)
    # Multiple components
    else:
        y_idx = torch.as_tensor(y_idx, dtype=torch.int64, device=device)
        n_y = list(y_idx.size())
        if n_y:
            n_y = n_y[0]
        else:
            n_y = 1
    # Check y_batch_size
    if y_batch_size is None:
        y_batch_size = n_y
    y_batch_size = min(n_y, max(1, y_batch_size))
    # Compute y_n_batch
    y_n_batch = max(1, int(np.ceil(n_y / y_batch_size)))
    # Fill y_idx with zeros to ensure even batchsizes
    y_idx_ = torch.concat((y_idx, torch.zeros(y_n_batch * y_batch_size\
        - y_idx.size(0), dtype=torch.int64, device=device)))
    # Repeat along batch dimension
    y_idx_ = y_idx_.repeat_interleave(x_batch_size * x_0_batch_size)
    # Build y_dtld
    b_sz = x_batch_size * x_0_batch_size * y_batch_size
    y_dtld = tuple(
        y_idx_[i*b_sz:(i+1)*b_sz].repeat(n_steps) for i in range(y_n_batch))
    return y_idx, y_dtld, n_y, y_batch_size


def _prepare_output(shape_prefix, x_size, dtype):
    return tuple(np.zeros(
        shape_prefix + x_size_i, dtype=dtype) for x_size_i in x_size)


@torch.no_grad()
def _record_y(y, y_idx, x_batch_size, dtype):
    if y is None:
        return None
    y_idx_ = y_idx.cpu().unsqueeze(dim=0).expand(x_batch_size, -1)
    return torch.gather(y.cpu(), dim=1, index=y_idx_).numpy().astype(dtype)


def _set_dtld_seed(dtld, seed):
    """
    Update dataloader seeds after initialization.
    """
    dtld.generator.manual_seed(seed)
    if dtld.dataset.rng is not None:
        dtld.dataset.rng.manual_seed(seed)


# Gradients


def grad(forward_func, backward_func, x=None, y_idx=None, x_size=None,
         y_size=None, embed_size=None, dtld_func=None, x_batch_size=1,
         y_batch_size=None, x_seed=None, embed_func=None, dtype=torch.float32,
         device='cpu', dtld_kwargs=None, fwd_kwargs=None, bwd_kwargs=None,
         embed_kwargs=None):
    """
    Compute gradients.
    """
    if fwd_kwargs is None:
        fwd_kwargs = {}
    if bwd_kwargs is None:
        bwd_kwargs = {}
    if embed_kwargs is None:
        embed_kwargs = {}
    # Prepare x dataloader
    x_dtld, n_x, x_n_batch, x_size, multi_x, dtype, dtype_np = _prepare_x_dtld(
        x, x_size, dtld_func, x_batch_size, x_seed, dtype, device, dtld_kwargs)
    # Prepare y_idx
    y_idx, y_dtld, n_y, y_batch_size = _prepare_y_idx(
        y_idx, y_size, x_batch_size, y_batch_size, x_0_batch_size=1, n_steps=1,
        device=device)
    # Prepare embed_size
    embed_size = _check_embed_size(embed_size, x_size)
    # Wrap functions for multi x
    if not multi_x:
        forward_func = _fwd_func_multi_x(forward_func)
        backward_func = _bwd_func_multi_x(backward_func)
        if embed_func is not None:
            embed_func = _embed_func_multi_x(embed_func)
    # Prepare outputs
    x_np = _prepare_output((n_x,), x_size, dtype_np)
    y_np = np.zeros((n_x, n_y), dtype=dtype_np)
    y_r = np.zeros((n_x, n_y), dtype=dtype_np)
    grad_ = _prepare_output((n_x, n_y), embed_size, dtype_np)
    # Iterate over x ##########################################################
    for i, (x_i, y_i) in enumerate(tqdm(x_dtld, total=x_n_batch, desc='grad')):
        # Break when x_n_batch is reached
        if i == x_n_batch:
            break
        # Multi x
        if not multi_x:
            x_i = (x_i,)
        # Current x slice
        x_slc = slice(i*x_batch_size, (i+1)*x_batch_size)
        # Record x, y
        for k, x_i_k in enumerate(x_i):
            x_np[k][x_slc] += x_i_k.cpu().numpy()
        y_i_np = _record_y(y_i, y_idx, x_batch_size, dtype_np)
        if y_i_np is not None:
            y_np[x_slc] += y_i_np
        # Prepare x
        with torch.no_grad():
            # Send x to the device
            x_i = tuple(x_i_k.to(device) for x_i_k in x_i)
            # Embed discrete inputs
            if embed_func is not None:
                x_i = embed_func(x_i, **embed_kwargs)
            # Repeat x along batch dimension
            x_i = tuple(x_i_k.repeat(*((y_batch_size,) + (1,)*(
                x_i_k.dim()-1))) for x_i_k in x_i)
        # Forward pass
        y_f = forward_func(x_i, **fwd_kwargs)
        # Iterate over output features y_idx ##################################
        for j, y_idx_j in enumerate(y_dtld):
            # Current slice and batchsize
            y_slc = slice(j*y_batch_size, min(n_y, (j+1)*y_batch_size))
            batch_size = y_slc.stop - y_slc.start
            # Compute predictions and gradients
            y_r_i_j, grad_i_j = backward_func(x_i, y_f, y_idx_j, **bwd_kwargs)
            y_r_i_j = y_r_i_j.reshape(
                (y_batch_size, x_batch_size))[:batch_size]
            grad_i_j = tuple(grad_i_j_k.reshape(
                (y_batch_size, x_batch_size) + grad_i_j_k.shape[1:])[
                    :batch_size] for grad_i_j_k in grad_i_j)
            # Record results
            y_r[x_slc, y_slc] += y_r_i_j.T
            for k, grad_i_j_k in enumerate(grad_i_j):
                grad_[k][x_slc, y_slc] += grad_i_j_k.swapaxes(0, 1)
    # Return results
    if multi_x:
        return x_np, y_np, y_r, grad_
    return x_np[0], y_np, y_r, grad_[0]


# Integrated gradients


def _int_grad_per_x_0(forward_func, backward_func, x, x_0, y_dtld, n_y,
                      n_steps, w, embed_size, x_batch_size, x_0_batch_size,
                      y_batch_size, dtype_np, fwd_kwargs, bwd_kwargs):
    # Prepare outputs
    y_0 = np.zeros((n_y, x_0_batch_size, x_batch_size), dtype=dtype_np)
    y_r = np.zeros((n_y, x_0_batch_size, x_batch_size), dtype=dtype_np)
    int_grad_ = _prepare_output(
        (n_y, x_0_batch_size, x_batch_size), embed_size, dtype_np)
    # Generate inputs along a linear path between x_0 and x
    with torch.no_grad():
        x_s = tuple()
        for x_0_i, x_i, w_i in zip(x_0, x, w):
            x_s_i = (1.0 - w_i)*x_0_i.unsqueeze(
                dim=0) + w_i*x_i.unsqueeze(dim=0)
            x_s += (x_s_i.flatten(0, 1),)
    # Compute input/baseline differences
    x_diff = tuple((x_i - x_0_i).cpu().numpy().reshape(
        (y_batch_size, x_0_batch_size, x_batch_size) + sz_i)\
            for x_i, x_0_i, sz_i in zip(x, x_0, embed_size))
    # Forward pass
    y_f = forward_func(x_s, **fwd_kwargs)
    # Iterate over output features y_idx
    for i, y_idx_i in enumerate(y_dtld):
        # Current slice and batchsize
        y_slc = slice(i*y_batch_size, min(n_y, (i+1)*y_batch_size))
        batch_size = y_slc.stop - y_slc.start
        # Compute predictions and gradients
        y_r_i, grad_i = backward_func(x_s, y_f, y_idx_i, **bwd_kwargs)
        y_r_i = y_r_i.reshape(
            (n_steps, y_batch_size, x_0_batch_size, x_batch_size))[
                :, :batch_size]
        grad_i = tuple(grad_i_j.reshape(
            (n_steps, y_batch_size, x_0_batch_size, x_batch_size) + sz_j)[
                :, :batch_size] for grad_i_j, sz_j in zip(grad_i, embed_size))
        # Record y_0 and y_r
        y_0[y_slc] += y_r_i[0]
        y_r[y_slc] += y_r_i[-1]
        # Compute integrated gradients (Riemann sums, trapezoidal rule)
        for j, (grad_i_j, x_diff_j) in enumerate(zip(grad_i, x_diff)):
            int_grad_i_j = grad_i_j[:-1] + grad_i_j[1:]
            int_grad_i_j = 0.5 * np.mean(int_grad_i_j, axis=0)
            int_grad_i_j *= x_diff_j[:batch_size]
            int_grad_[j][y_slc] += int_grad_i_j
    return y_0, y_r, int_grad_


def _int_grad_per_x(forward_func, backward_func, x, multi_x, x_0_dtld, n_x_0,
                    x_0_n_batch, y_dtld, n_y, n_steps, w, embed_size,
                    x_batch_size, x_0_batch_size, y_batch_size, x_0_seed,
                    embed_func, dtype_np, device, fwd_kwargs, bwd_kwargs,
                    embed_kwargs):
    # Prepare outputs
    y_0 = np.zeros((x_batch_size, n_y), dtype=dtype_np)
    y_r = np.zeros((x_batch_size, n_y), dtype=dtype_np)
    int_grad_ = _prepare_output((x_batch_size, n_y), embed_size, dtype_np)
    # Update x_0_dtld seed
    if not isinstance(x_0_dtld, tuple):
        _set_dtld_seed(x_0_dtld, x_0_seed)
    # Iterate over baselines
    for i, (x_0_i, _) in enumerate(x_0_dtld):
        # Break when x_0_n_batch is reached
        if i == x_0_n_batch:
            break
        # Multi x
        if not multi_x:
            x_0_i = (x_0_i,)
        # Prepare x_0
        with torch.no_grad():
            # Send x_0 to the device
            x_0_i = tuple(x_0_i_j.to(device) for x_0_i_j in x_0_i)
            # Embed discrete inputs
            if embed_func is not None:
                x_0_i = embed_func(x_0_i, **embed_kwargs)
            # Repeat x_0 along batch dimension
            x_0_i = tuple(x_0_i_j.repeat(*((y_batch_size,) + (1,)*(
                x_0_i_j.dim()-1))) for x_0_i_j in x_0_i)
        # Compute integrated gradients
        y_0_i, y_r_i, int_grad_i = _int_grad_per_x_0(
            forward_func, backward_func, x, x_0_i, y_dtld, n_y, n_steps, w,
            embed_size, x_batch_size, x_0_batch_size, y_batch_size, dtype_np,
            fwd_kwargs, bwd_kwargs)
        # Record y_0 and y_r
        y_0 += np.sum(y_0_i, axis=1).T
        y_r += np.sum(y_r_i, axis=1).T
        # Record integrated gradients
        for j, int_grad_i_j in enumerate(int_grad_i):
            int_grad_[j][...] += np.sum(int_grad_i_j, axis=1).swapaxes(0, 1)
    # Average across baselines
    y_0 /= n_x_0
    y_r /= n_x_0
    for int_grad_i in int_grad_:
        int_grad_i /= n_x_0
    return y_0, y_r, int_grad_


def int_grad(forward_func, backward_func, x=None, x_0=None, y_idx=None,
             n_steps=64, x_size=None, y_size=None, embed_size=None,
             output_size=None, dtld_func=None, x_batch_size=1,
             x_0_batch_size=1, y_batch_size=None, x_seed=None, x_0_seed=100,
             embed_func=None, ig_post_func=None, dtype=torch.float32,
             device='cpu', dtld_kwargs=None, fwd_kwargs=None, bwd_kwargs=None,
             embed_kwargs=None, ig_post_kwargs=None, check_error=False):
    """
    Compute integrated gradients.
    """
    if fwd_kwargs is None:
        fwd_kwargs = {}
    if bwd_kwargs is None:
        bwd_kwargs = {}
    if embed_kwargs is None:
        embed_kwargs = {}
    if ig_post_kwargs is None:
        ig_post_kwargs = {}
    # Prepare x dataloader
    x_dtld, n_x, x_n_batch, x_size, multi_x, dtype, dtype_np = _prepare_x_dtld(
        x, x_size, dtld_func, x_batch_size, x_seed, dtype, device, dtld_kwargs)
    # Prepare x_0 dataloader
    x_0_dtld, n_x_0, x_0_n_batch, x_0_batch_size = _prepare_x_0_dtld(
        x_0, x_size, dtld_func, x_batch_size, x_0_batch_size, x_0_seed,
        multi_x, dtype, device, dtld_kwargs)
    # Prepare y_idx
    y_idx, y_dtld, n_y, y_batch_size = _prepare_y_idx(
        y_idx, y_size, x_batch_size, y_batch_size, x_0_batch_size, n_steps,
        device)
    # Prepare sizes
    embed_size = _check_embed_size(embed_size, x_size)
    output_size = _check_output_size(output_size, embed_size)
    # Prepare interpolation coefficients w
    w = tuple(torch.linspace(
        0.0, 1.0, n_steps, dtype=torch.float32, device=device)[
            (...,) + (None,) * (1+len(sz_i))] for sz_i in embed_size)
    # Wrap functions for multi x
    if not multi_x:
        forward_func = _fwd_func_multi_x(forward_func)
        backward_func = _bwd_func_multi_x(backward_func)
        if embed_func is not None:
            embed_func = _embed_func_multi_x(embed_func)
        if ig_post_func is not None:
            ig_post_func = _ig_post_func_multi_x(ig_post_func)
    # Prepare outputs
    x_np = _prepare_output((n_x,), x_size, dtype_np)
    y_np = np.zeros((n_x, n_y), dtype=dtype_np)
    y_0 = np.zeros((n_x, n_y), dtype=dtype_np)
    y_r = np.zeros((n_x, n_y), dtype=dtype_np)
    int_grad_ = _prepare_output((n_x, n_y), output_size, dtype_np)
    # Iterate over x
    for i, (x_i, y_i) in enumerate(tqdm(x_dtld, total=x_n_batch, desc='ig')):
        # Break when x_n_batch is reached
        if i == x_n_batch:
            break
        # Multi x
        if not multi_x:
            x_i = (x_i,)
        # Current slice
        slc = slice(i*x_batch_size, (i+1)*x_batch_size)
        # Record x, y
        for j, x_i_j in enumerate(x_i):
            x_np[j][slc] += x_i_j.cpu().numpy()
        y_i_np = _record_y(y_i, y_idx, x_batch_size, dtype_np)
        if y_i_np is not None:
            y_np[slc] += y_i_np
        # Prepare x
        with torch.no_grad():
            # Send x to the device
            x_i = tuple(x_i_j.to(device) for x_i_j in x_i)
            # Embed discrete inputs
            if ig_post_func is not None:
                x_i_ori = x_i
            if embed_func is not None:
                x_i = embed_func(x_i, **embed_kwargs)
            # Repeat x along batch dimension
            x_i = tuple(x_i_j.repeat(*((x_0_batch_size*y_batch_size,) + (1,)*(
                x_i_j.dim()-1))) for x_i_j in x_i)
        # Compute current x_0_seed seed
        x_0_seed_i = int(x_0_seed+1e6*(i+1))
        # Compute integrated gradients
        y_0_i, y_r_i, int_grad_i = _int_grad_per_x(
            forward_func, backward_func, x_i, multi_x, x_0_dtld, n_x_0,
            x_0_n_batch, y_dtld, n_y, n_steps, w, embed_size, x_batch_size,
            x_0_batch_size, y_batch_size, x_0_seed_i, embed_func, dtype_np,
            device, fwd_kwargs, bwd_kwargs, embed_kwargs)
        # Record y_0 and y_r
        y_0[slc] += y_0_i
        y_r[slc] += y_r_i
        # Apply integrated gradients post-function
        if ig_post_func is not None:
            int_grad_i = ig_post_func(int_grad_i, x_i_ori, **ig_post_kwargs)
        # Record integrated gradients
        for j, int_grad_i_j in enumerate(int_grad_i):
            int_grad_[j][slc] += int_grad_i_j
    # Check error
    if check_error:
        int_grad_sum = np.zeros((n_x, n_y), dtype=dtype_np)
        for int_grad_i in int_grad_:
            int_grad_sum += np.sum(int_grad_i.reshape((n_x, n_y, -1)), axis=2)
        print(f'ig err: {np.mean(np.abs(int_grad_sum - y_r + y_0)):>9.6f}')
    # Return results
    if multi_x:
        return x_np, y_np, y_0, y_r, int_grad_
    return x_np[0], y_np, y_0, y_r, int_grad_[0]


# Integrated gradient correlation


def int_grad_corr(forward_func, backward_func, dtld_func, x_0=None, y_idx=None,
                  n_steps=64, x_size=None, y_size=None, embed_size=None,
                  output_size=None, x_batch_size=1, x_0_batch_size=1,
                  y_batch_size=None, x_0_seed=100, embed_func=None,
                  ig_post_func=None, dtype=torch.float32, device='cpu',
                  dtld_kwargs=None, fwd_kwargs=None, bwd_kwargs=None,
                  embed_kwargs=None, ig_post_kwargs=None, check_error=False,
                  n_x=None):
    """
    Compute integrated gradient correlation.
    """
    if fwd_kwargs is None:
        fwd_kwargs = {}
    if bwd_kwargs is None:
        bwd_kwargs = {}
    if embed_kwargs is None:
        embed_kwargs = {}
    if ig_post_kwargs is None:
        ig_post_kwargs = {}
    # Prepare x dataloader
    x_seed = None
    if n_x is not None:
        x_seed = int(1e9 + x_0_seed)
    x_dtld, n_x, x_n_batch, x_size, multi_x, dtype, dtype_np = _prepare_x_dtld(
        n_x, x_size, dtld_func, x_batch_size, x_seed, dtype, device,
        dtld_kwargs)
    # Prepare x_0 dataloader
    x_0_dtld, n_x_0, x_0_n_batch, x_0_batch_size = _prepare_x_0_dtld(
        x_0, x_size, dtld_func, x_batch_size, x_0_batch_size, x_0_seed,
        multi_x, dtype, device, dtld_kwargs)
    # Prepare y_idx
    y_idx, y_dtld, n_y, y_batch_size = _prepare_y_idx(
        y_idx, y_size, x_batch_size, y_batch_size, x_0_batch_size, n_steps,
        device)
    # Prepare sizes
    embed_size = _check_embed_size(embed_size, x_size)
    output_size = _check_output_size(output_size, embed_size)
    # Prepare interpolation coefficients w
    w = tuple(torch.linspace(
        0.0, 1.0, n_steps, dtype=torch.float32, device=device)[
            (...,) + (None,) * (1+len(sz_i))] for sz_i in embed_size)
    # Wrap functions for multi x
    if not multi_x:
        forward_func = _fwd_func_multi_x(forward_func)
        backward_func = _bwd_func_multi_x(backward_func)
        if embed_func is not None:
            embed_func = _embed_func_multi_x(embed_func)
        if ig_post_func is not None:
            ig_post_func = _ig_post_func_multi_x(ig_post_func)
    # Prepare outputs
    ig_error = 0.0
    y_mean = np.zeros(n_y, dtype=dtype_np)
    y_std = np.zeros(n_y, dtype=dtype_np)
    y_r_mean = np.zeros(n_y, dtype=dtype_np)
    y_r_std = np.zeros(n_y, dtype=dtype_np)
    corr = np.zeros(n_y, dtype=dtype_np)
    igc = _prepare_output((n_y,), output_size, dtype_np)
    igc_mean = _prepare_output((n_y,), output_size, dtype_np)
    # Iterate over x
    postfix = None
    if check_error:
        postfix = 'ig err: ?'
    tqdm_iterator = tqdm(x_dtld, total=x_n_batch, desc='igc', postfix=postfix)
    for i, (x_i, y_i) in enumerate(tqdm_iterator):
        n_x_count = (i+1) * x_batch_size
        # Break when x_n_batch is reached
        if i == x_n_batch:
            break
        # Multi x
        if not multi_x:
            x_i = (x_i,)
        # Update y_mean and y_std
        y_i_np = _record_y(y_i, y_idx, x_batch_size, dtype_np)
        y_delta = y_i_np - y_mean
        y_mean += np.sum(y_delta, axis=0) / n_x_count
        y_delta_2 = y_i_np - y_mean
        y_std += np.sum(y_delta * y_delta_2, axis=0)
        # Prepare x
        with torch.no_grad():
            # Send x to the device
            x_i = tuple(x_i_j.to(device) for x_i_j in x_i)
            # Embed discrete inputs
            if ig_post_func is not None:
                x_i_ori = x_i
            if embed_func is not None:
                x_i = embed_func(x_i, **embed_kwargs)
            # Repeat x along batch dimension
            x_i = tuple(x_i_j.repeat(*((x_0_batch_size*y_batch_size,) + (1,)*(
                x_i_j.dim()-1))) for x_i_j in x_i)
        # Compute current x_0_seed seed
        x_0_seed_i = int(x_0_seed+1e6*(i+1))
        # Compute IG
        y_0_i, y_r_i, ig_i = _int_grad_per_x(
            forward_func, backward_func, x_i, multi_x, x_0_dtld, n_x_0,
            x_0_n_batch, y_dtld, n_y, n_steps, w, embed_size, x_batch_size,
            x_0_batch_size, y_batch_size, x_0_seed_i, embed_func, dtype_np,
            device, fwd_kwargs, bwd_kwargs, embed_kwargs)
        # Update y_r_mean and y_r_std
        y_r_delta = y_r_i - y_r_mean
        y_r_mean += np.sum(y_r_delta, axis=0) / n_x_count
        y_r_std += np.sum(y_r_delta * (y_r_i - y_r_mean), axis=0)
        # Apply IG post-function
        if ig_post_func is not None:
            ig_i = ig_post_func(ig_i, x_i_ori, **ig_post_kwargs)
        # Update correlation
        corr += np.sum(y_r_delta * y_delta_2, axis=0)
        # Update IGC
        for j, (ig_i_j, sz_j) in enumerate(zip(ig_i, output_size)):
            igc_delta = ig_i_j - igc_mean[j]
            igc_mean[j][...] += np.sum(igc_delta, axis=0) / n_x_count
            igc[j][...] += np.sum(
                igc_delta * y_delta_2[(...,) + (None,)*len(sz_j)], axis=0)
        # Check IG error and display incremental value in tqdm
        if check_error:
            ig_sum_i = np.zeros((x_batch_size, n_y), dtype=dtype_np)
            for ig_i_j in ig_i:
                ig_sum_i += np.sum(
                    ig_i_j.reshape((x_batch_size, n_y, -1)), axis=2)
            ig_error_i = np.mean(np.abs(ig_sum_i - y_r_i + y_0_i), axis=1)
            ig_error += np.sum(ig_error_i - ig_error) / n_x_count
            tqdm_iterator.set_postfix_str(
                f'ig err: {ig_error:>9.6f}', refresh=False)
    # Finalize y_std and y_r_std
    y_std /= n_x
    y_r_std /= n_x
    y_y_r_std = np.sqrt(y_std * y_r_std)
    # Finalize IGC
    for igc_i in igc:
        igc_i /= n_x
        igc_i /= y_y_r_std[(...,) + (None,)*(igc_i.ndim-1)]
    # Check IGC error
    if check_error:
        igc_sum = np.zeros(n_y, dtype=dtype_np)
        for igc_i in igc:
            igc_sum += np.sum(np.reshape(igc_i, (n_y, -1)), axis=1)
        corr /= n_x
        corr /= y_y_r_std
        print(f'igc err: {np.mean(np.abs(igc_sum - corr)):>9.6f}')
    # Return results
    if multi_x:
        return igc
    return igc[0]


@torch.no_grad()
def igc_error(igc, forward_func, dtld_func, y_idx=None, x_size=None,
              y_size=None, x_batch_size=1, x_seed=None, embed_func=None,
              dtype=torch.float32, device='cpu', dtld_kwargs=None,
              fwd_kwargs=None, embed_kwargs=None, n_x=None):
    """
    Compute IGC error.
    """
    if fwd_kwargs is None:
        fwd_kwargs = {}
    if embed_kwargs is None:
        embed_kwargs = {}
    # Prepare x dataloader
    x_dtld, n_x, x_n_batch, x_size, multi_x, dtype, dtype_np = _prepare_x_dtld(
        n_x, x_size, dtld_func, x_batch_size, x_seed, dtype, device,
        dtld_kwargs)
    # Prepare y_idx
    x_batch_size_tmp, y_batch_size, x_0_batch_size, n_steps = 1, None, 1, 1
    y_idx, _, n_y, _ = _prepare_y_idx(
        y_idx, y_size, x_batch_size_tmp, y_batch_size, x_0_batch_size, n_steps,
        device)
    # Wrap functions for multi x
    if not multi_x:
        forward_func = _fwd_func_multi_x(forward_func)
        if embed_func is not None:
            embed_func = _embed_func_multi_x(embed_func)
    # Prepare outputs
    y_mean = np.zeros(n_y, dtype=dtype_np)
    y_std = np.zeros(n_y, dtype=dtype_np)
    y_r_mean = np.zeros(n_y, dtype=dtype_np)
    y_r_std = np.zeros(n_y, dtype=dtype_np)
    corr = np.zeros(n_y, dtype=dtype_np)
    # Iterate over x
    for i, (x_i, y_i) in enumerate(
            tqdm(x_dtld, total=x_n_batch, desc='igc err')):
        n_x_count = (i+1) * x_batch_size
        # Break when x_n_batch is reached
        if i == x_n_batch:
            break
        # Multi x
        if not multi_x:
            x_i = (x_i,)
        # Update y_mean and y_std
        y_i_np = _record_y(y_i, y_idx, x_batch_size, dtype_np)
        y_delta = y_i_np - y_mean
        y_mean += np.sum(y_delta, axis=0) / n_x_count
        y_delta_2 = y_i_np - y_mean
        y_std += np.sum(y_delta * y_delta_2, axis=0)
        # Send inputs to the device
        x_i = tuple(x_i_j.to(device) for x_i_j in x_i)
        y_i = y_i.to(device)
        # Embed discrete inputs
        if embed_func is not None:
            x_i = embed_func(x_i, **embed_kwargs)
        # Compute predictions
        y_r_i = forward_func(x_i, **fwd_kwargs)
        # Update y_r_mean and y_r_std
        y_r_i_np = _record_y(y_r_i, y_idx, x_batch_size, dtype_np)
        y_r_delta = y_r_i_np - y_r_mean
        y_r_mean += np.sum(y_r_delta, axis=0) / n_x_count
        y_r_std += np.sum(y_r_delta * (y_r_i_np - y_r_mean), axis=0)
        # Update correlation
        corr += np.sum(y_r_delta * y_delta_2, axis=0)
    # Finalize y_std and y_r_std
    y_std /= n_x
    y_r_std /= n_x
    y_y_r_std = np.sqrt(y_std * y_r_std)
    # Finalize correlation
    corr /= n_x
    corr /= y_y_r_std
    # Multi x
    if not multi_x:
        igc = (igc,)
    # Check IGC error
    igc_sum = np.zeros(n_y, dtype=dtype_np)
    for igc_i in igc:
        igc_sum += np.sum(np.reshape(igc_i, (n_y, -1)), axis=1)
    error = np.abs(igc_sum - corr)
    print(f'igc err: {np.mean(error):>9.6f}')
    return error


# Integrated gradient auto-correlation


def int_grad_auto_corr(forward_func, backward_func, dtld_func, x_0=None,
                       y_idx=None, n_steps=64, x_size=None, y_size=None,
                       embed_size=None, output_size=None, x_batch_size=1,
                       x_0_batch_size=1, y_batch_size=None, x_0_seed=100,
                       embed_func=None, ig_post_func=None, dtype=torch.float32,
                       device='cpu', dtld_kwargs=None, fwd_kwargs=None,
                       bwd_kwargs=None, embed_kwargs=None, ig_post_kwargs=None,
                       check_error=False, n_x=None):
    """
    Compute integrated gradient auto-correlation.
    """
    if fwd_kwargs is None:
        fwd_kwargs = {}
    if bwd_kwargs is None:
        bwd_kwargs = {}
    if embed_kwargs is None:
        embed_kwargs = {}
    if ig_post_kwargs is None:
        ig_post_kwargs = {}
    # Prepare x dataloader
    x_seed = None
    if n_x is not None:
        x_seed = int(1e9 + x_0_seed)
    x_dtld, n_x, x_n_batch, x_size, multi_x, dtype, dtype_np = _prepare_x_dtld(
        n_x, x_size, dtld_func, x_batch_size, x_seed, dtype, device,
        dtld_kwargs)
    # Prepare x_0 dataloader
    x_0_dtld, n_x_0, x_0_n_batch, x_0_batch_size = _prepare_x_0_dtld(
        x_0, x_size, dtld_func, x_batch_size, x_0_batch_size, x_0_seed,
        multi_x, dtype, device, dtld_kwargs)
    # Prepare y_idx
    y_idx, y_dtld, n_y, y_batch_size = _prepare_y_idx(
        y_idx, y_size, x_batch_size, y_batch_size, x_0_batch_size, n_steps,
        device)
    # Prepare sizes
    embed_size = _check_embed_size(embed_size, x_size)
    output_size = _check_output_size(output_size, embed_size)
    # Prepare interpolation coefficients w
    w = tuple(torch.linspace(
        0.0, 1.0, n_steps, dtype=torch.float32, device=device)[
            (...,) + (None,) * (1+len(sz_i))] for sz_i in embed_size)
    # Wrap functions for multi x
    if not multi_x:
        forward_func = _fwd_func_multi_x(forward_func)
        backward_func = _bwd_func_multi_x(backward_func)
        if embed_func is not None:
            embed_func = _embed_func_multi_x(embed_func)
        if ig_post_func is not None:
            ig_post_func = _ig_post_func_multi_x(ig_post_func)
    # Prepare outputs
    ig_error = 0.0
    y_r_mean = np.zeros(n_y, dtype=dtype_np)
    y_r_var = np.zeros(n_y, dtype=dtype_np)
    igac = _prepare_output((n_y,), output_size, dtype_np)
    igac_mean = _prepare_output((n_y,), output_size, dtype_np)
    # Iterate over x
    postfix = None
    if check_error:
        postfix = 'ig err: ?'
    tqdm_iterator = tqdm(x_dtld, total=x_n_batch, desc='igac', postfix=postfix)
    for i, (x_i, _) in enumerate(tqdm_iterator):
        n_x_count = (i+1) * x_batch_size
        # Break when x_n_batch is reached
        if i == x_n_batch:
            break
        # Multi x
        if not multi_x:
            x_i = (x_i,)
        # Prepare x
        with torch.no_grad():
            # Send x to the device
            x_i = tuple(x_i_j.to(device) for x_i_j in x_i)
            # Embed discrete inputs
            if ig_post_func is not None:
                x_i_ori = x_i
            if embed_func is not None:
                x_i = embed_func(x_i, **embed_kwargs)
            # Repeat x along batch dimension
            x_i = tuple(x_i_j.repeat(*((x_0_batch_size*y_batch_size,) + (1,)*(
                x_i_j.dim()-1))) for x_i_j in x_i)
        # Compute current x_0_seed seed
        x_0_seed_i = int(x_0_seed+1e6*(i+1))
        # Compute IG
        y_0_i, y_r_i, ig_i = _int_grad_per_x(
            forward_func, backward_func, x_i, multi_x, x_0_dtld, n_x_0,
            x_0_n_batch, y_dtld, n_y, n_steps, w, embed_size, x_batch_size,
            x_0_batch_size, y_batch_size, x_0_seed_i, embed_func, dtype_np,
            device, fwd_kwargs, bwd_kwargs, embed_kwargs)
        # Update y_r_mean and y_r_var
        y_r_delta = y_r_i - y_r_mean
        y_r_mean += np.sum(y_r_delta, axis=0) / n_x_count
        y_r_delta_2 = y_r_i - y_r_mean
        y_r_var += np.sum(y_r_delta * y_r_delta_2, axis=0)
        # Apply IG post-function
        if ig_post_func is not None:
            ig_i = ig_post_func(ig_i, x_i_ori, **ig_post_kwargs)
        # Update IGaC
        for j, (ig_i_j, sz_j) in enumerate(zip(ig_i, output_size)):
            igac_delta = ig_i_j - igac_mean[j]
            igac_mean[j][...] += np.sum(igac_delta, axis=0) / n_x_count
            igac[j][...] += np.sum(
                igac_delta * y_r_delta_2[(...,) + (None,)*len(sz_j)], axis=0)
        # Check IG error and display incremental value in tqdm
        if check_error:
            ig_sum_i = np.zeros((x_batch_size, n_y), dtype=dtype_np)
            for ig_i_j in ig_i:
                ig_sum_i += np.sum(
                    ig_i_j.reshape((x_batch_size, n_y, -1)), axis=2)
            ig_error_i = np.mean(np.abs(ig_sum_i - y_r_i + y_0_i), axis=1)
            ig_error += np.sum(ig_error_i - ig_error) / n_x_count
            tqdm_iterator.set_postfix_str(
                f'ig err: {ig_error:>9.6f}', refresh=False)
    # Finalize y_r_var
    y_r_var /= n_x
    # Finalize IGaC
    for igac_i in igac:
        igac_i /= n_x
        igac_i /= y_r_var[(...,) + (None,)*(igac_i.ndim-1)]
    # Check IGaC error
    if check_error:
        igac_sum = np.zeros(n_y, dtype=dtype_np)
        for igac_i in igac:
            igac_sum += np.sum(np.reshape(igac_i, (n_y, -1)), axis=1)
        print(f'igac err: {np.mean(np.abs(igac_sum - 1.0)):>9.6f}')
    # Return results
    if multi_x:
        return igac
    return igac[0]


@torch.no_grad()
def igac_error(igac):
    """
    Compute IGaC error.
    """
    # Multi x
    if not isinstance(igac, tuple):
        igac = (igac,)
    # Check IGaC error
    n_y = igac[0].shape[0]
    igac_sum = np.zeros(n_y, dtype=igac[0].dtype)
    for igac_i in igac:
        igac_sum += np.sum(np.reshape(igac_i, (n_y, -1)), axis=1)
    error = np.abs(igac_sum - 1.0)
    print(f'igac err: {np.mean(error):>9.6f}')
    return error
