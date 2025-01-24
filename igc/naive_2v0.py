"""
Naive alternatives to Integrated Gradient Correlation (IGC).

Author: Pierre Lelievre
"""

import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_ind_from_stats

from .igc_2v0 import (
    _prepare_x_dtld, _prepare_x_0_dtld, _prepare_y_idx, _check_embed_size,
    _check_output_size, _fwd_func_multi_x, _bwd_func_multi_x,
    _embed_func_multi_x, _ig_post_func_multi_x, _prepare_output, _record_y,
    _int_grad_per_x)


# Integrated gradient mean and std


def int_grad_naive(forward_func, backward_func, dtld_func, x_0=None,
                   y_idx=None, n_steps=64, x_size=None, y_size=None,
                   embed_size=None, output_size=None, x_batch_size=1,
                   x_0_batch_size=1, y_batch_size=None, x_0_seed=100,
                   embed_func=None, ig_post_func=None, dtype=torch.float32,
                   device='cpu', dtld_kwargs=None, fwd_kwargs=None,
                   bwd_kwargs=None, embed_kwargs=None, ig_post_kwargs=None,
                   check_error=False, n_x=None):
    """
    Compute integrated gradients mean and std.
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
    ig_mean = _prepare_output((n_y,), output_size, dtype_np)
    ig_std = _prepare_output((n_y,), output_size, dtype_np)
    # Iterate over x
    postfix = None
    if check_error:
        postfix = 'ig err: ?'
    tqdm_iterator = tqdm(x_dtld, total=x_n_batch, desc='igm', postfix=postfix)
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
        # Apply IG post-function
        if ig_post_func is not None:
            ig_i = ig_post_func(ig_i, x_i_ori, **ig_post_kwargs)
        # Update IG mean and std
        for j, ig_i_j in enumerate(ig_i):
            ig_i_j_delta = ig_i_j - ig_mean[j]
            ig_mean[j][...] += np.sum(ig_i_j_delta, axis=0) / n_x_count
            ig_std[j][...] += np.sum(
                ig_i_j_delta * (ig_i_j - ig_mean[j]), axis=0)
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
    # Finalize IG std
    for ig_std_i in ig_std:
        ig_std_i /= n_x
        ig_std_i = np.sqrt(ig_std_i)
    # Return results
    if multi_x:
        return ig_mean, ig_std
    return ig_mean[0], ig_std[0]


# Naive correlation


@torch.no_grad()
def correlation_naive(dtld_func, y_idx=None, x_size=None, y_size=None,
                      embed_size=None, x_batch_size=1, y_batch_size=None,
                      x_seed=None, embed_func=None, dtype=torch.float32,
                      device='cpu', dtld_kwargs=None, embed_kwargs=None,
                      n_x=None):
    """
    Compute naive correlation.
    """
    if embed_kwargs is None:
        embed_kwargs = {}
    # Prepare x dataloader
    x_dtld, n_x, x_n_batch, x_size, multi_x, dtype, dtype_np = _prepare_x_dtld(
        n_x, x_size, dtld_func, x_batch_size, x_seed, dtype, device,
        dtld_kwargs)
    # Prepare y_idx
    x_batch_size_tmp, x_0_batch_size, n_steps = 1, 1, 1
    y_idx, y_dtld, n_y, y_batch_size = _prepare_y_idx(
        y_idx, y_size, x_batch_size_tmp, y_batch_size, x_0_batch_size, n_steps,
        device)
    # Prepare sizes
    embed_size = _check_embed_size(embed_size, x_size)
    # Wrap functions for multi x
    if (not multi_x) and (embed_func is not None):
        embed_func = _embed_func_multi_x(embed_func)
    # Prepare outputs
    y_mean = np.zeros(n_y, dtype=dtype_np)
    y_std = np.zeros(n_y, dtype=dtype_np)
    x_mean = _prepare_output((n_y,), embed_size, dtype_np)
    x_std = _prepare_output((n_y,), embed_size, dtype_np)
    corr = _prepare_output((n_y,), embed_size, dtype_np)
    # Iterate over x
    for i, (x_i, y_i) in enumerate(tqdm(x_dtld, total=x_n_batch, desc='corr')):
        n_x_count = (i+1) * x_batch_size
        # Break when x_n_batch is reached
        if i == x_n_batch:
            break
        # Multi x
        if not multi_x:
            x_i = (x_i,)
        # Record y_mean and y_std
        y_i_np = _record_y(y_i, y_idx, x_batch_size, dtype_np)
        y_delta = y_i_np - y_mean
        y_mean += np.sum(y_delta, axis=0) / n_x_count
        y_delta_2 = y_i_np - y_mean
        y_std += np.sum(y_delta * y_delta_2, axis=0)
        # Send data to the device
        x_i = tuple(x_i_j.to(device) for x_i_j in x_i)
        y_delta_2 = torch.as_tensor(y_delta_2, device=device)
        # Embed discrete inputs
        if embed_func is not None:
            x_i = embed_func(x_i, **embed_kwargs)
        # Record x_mean and x_std
        x_i_delta = []
        for j, x_i_j in enumerate(x_i):
            x_i_j_np = x_i_j.unsqueeze(dim=1).repeat(
                1, *((n_y,) + (1,)*(x_i_j.dim()-1))).cpu().numpy()
            x_i_j_delta = x_i_j_np - x_mean[j]
            x_i_delta.append(x_i_j_delta)
            x_mean[j][...] += np.sum(x_i_j_delta, axis=0) / n_x_count
            x_std[j][...] += np.sum(
                x_i_j_delta * (x_i_j_np - x_mean[j]), axis=0)
        # Prepare x_i_delta
        x_i_delta = tuple(torch.as_tensor(
            x_i_j_delta, dtype=dtype_j, device=device)\
                for x_i_j_delta, dtype_j in zip(x_i_delta, dtype))
        # Iterate over output features y_idx
        for j, y_idx_j in enumerate(y_dtld):
            # Current slice and batchsize
            y_slc = slice(j*y_batch_size, min(n_y, (j+1)*y_batch_size))
            batch_size = y_slc.stop - y_slc.start
            # Compute correlation
            y_i_j_d_2 = torch.gather(y_delta_2, dim=1, index=y_idx_j.unsqueeze(
                dim=0).expand(x_batch_size, -1))
            for k, (x_i_k_d, sz_k) in enumerate(zip(x_i_delta, embed_size)):
                x_i_j_k_d = torch.gather(x_i_k_d, dim=1, index=y_idx_j[
                    (None, ...) + (None,) * len(sz_k)].expand(
                        x_batch_size, -1, *sz_k))
                corr_i_j_k = x_i_j_k_d * y_i_j_d_2[(...,) + (None,)*len(sz_k)]
                corr[k][y_slc] += torch.sum(
                    corr_i_j_k, dim=0)[:batch_size].cpu().numpy()
    # Finalize x_std and y_std
    for x_std_i in x_std:
        x_std_i /= n_x
        x_std_i = np.sqrt(x_std_i)
    y_std /= n_x
    y_std = np.sqrt(y_std)
    # Compute correlation
    for corr_i, x_std_i in zip(corr, x_std):
        corr_i /= n_x
        corr_i /= x_std_i
        corr_i /= y_std[(...,) + (None,)*(corr_i.ndim-1)]
    # Return results
    if multi_x:
        return corr
    return corr[0]


# Naive t-test


@torch.no_grad()
def ttest_naive(dtld_func, y_idx=None, cat_ranges=None, x_size=None,
                y_size=None, embed_size=None, x_batch_size=1,
                y_batch_size=None, x_seed=None, embed_func=None,
                dtype=torch.float32, device='cpu', dtld_kwargs=None,
                embed_kwargs=None, n_x=None):
    """
    Compute naive ttest.
    """
    if embed_kwargs is None:
        embed_kwargs = {}
    # Prepare x dataloader
    x_dtld, n_x, x_n_batch, x_size, multi_x, dtype, dtype_np = _prepare_x_dtld(
        n_x, x_size, dtld_func, x_batch_size, x_seed, dtype, device,
        dtld_kwargs)
    # Prepare y_idx
    x_batch_size_tmp, x_0_batch_size, n_steps = 1, 1, 1
    y_idx, y_dtld, n_y, y_batch_size = _prepare_y_idx(
        y_idx, y_size, x_batch_size_tmp, y_batch_size, x_0_batch_size, n_steps,
        device)
    # Prepare sizes
    embed_size = _check_embed_size(embed_size, x_size)
    # Wrap functions for multi x
    if (not multi_x) and (embed_func is not None):
        embed_func = _embed_func_multi_x(embed_func)
    # Prepares cat_ranges
    if cat_ranges is None:
        cat_ranges = (0.0, 0.0)
    assert isinstance(cat_ranges, tuple)
    if isinstance(cat_ranges[0], (float, int)):
        cat_ranges = (cat_ranges,) * n_y
    assert len(cat_ranges) == n_y
    cat_ranges = torch.tensor(
        cat_ranges, dtype=torch.float32,
        device=device).transpose(0, 1).unsqueeze(dim=1)
    # Prepare outputs
    n_x_a = np.zeros(n_y, dtype=np.int64)
    x_mean_a = _prepare_output((n_y,), embed_size, dtype_np)
    x_std_a = _prepare_output((n_y,), embed_size, dtype_np)
    n_x_b = np.zeros(n_y, dtype=np.int64)
    x_mean_b = _prepare_output((n_y,), embed_size, dtype_np)
    x_std_b = _prepare_output((n_y,), embed_size, dtype_np)
    ttest = _prepare_output((n_y,), embed_size, dtype_np)
    # Iterate over x
    for i, (x_i, y_i) in enumerate(
            tqdm(x_dtld, total=x_n_batch, desc='ttest')):
        # Break when x_n_batch is reached
        if i == x_n_batch:
            break
        # Multi x
        if not multi_x:
            x_i = (x_i,)
        # Send inputs to the device
        x_i = tuple(x_i_j.to(device) for x_i_j in x_i)
        y_i = y_i.to(device)
        # Embed discrete inputs
        if embed_func is not None:
            x_i = embed_func(x_i, **embed_kwargs)
        # Iterate over output features y_idx
        for j, y_idx_j in enumerate(y_dtld):
            # Current slice and batchsize
            y_slc = slice(j*y_batch_size, min(n_y, (j+1)*y_batch_size))
            batch_size = y_slc.stop - y_slc.start
            # Extract a and b data
            y_i_j = torch.gather(y_i, dim=1, index=y_idx_j.unsqueeze(
                dim=0).expand(x_batch_size, -1))
            mask_a = torch.le(y_i_j, cat_ranges[0]).cpu().numpy()
            n_x_a[y_slc] += np.sum(
                mask_a, axis=0)[:batch_size].astype(np.int64)
            mask_b = torch.ge(y_i_j, cat_ranges[1]).cpu().numpy()
            n_x_b[y_slc] += np.sum(
                mask_b, axis=0)[:batch_size].astype(np.int64)
            for k, x_i_k in enumerate(x_i):
                x_i_k_np = x_i_k.unsqueeze(dim=1).repeat(
                    1, *((batch_size,) + (1,)*(x_i_k.dim()-1))).cpu().numpy()
                for m in range(y_slc.start, y_slc.stop):
                    if np.sum(mask_a[:, m]):
                        selected_a = x_i_k_np[:, m][mask_a[:, m]]
                        x_delta_a = selected_a - x_mean_a[k][m]
                        x_mean_a[k][m] += np.sum(x_delta_a, axis=0) / n_x_a[m]
                        x_delta_2_a = selected_a - x_mean_a[k][m]
                        x_std_a[k][m] += np.sum(x_delta_a*x_delta_2_a, axis=0)
                    if np.sum(mask_b[:, m]):
                        selected_b = x_i_k_np[:, m][mask_b[:, m]]
                        x_delta_b = selected_b - x_mean_b[k][m]
                        x_mean_b[k][m] += np.sum(x_delta_b, axis=0) / n_x_b[m]
                        x_delta_2_b = selected_b - x_mean_b[k][m]
                        x_std_b[k][m] += np.sum(x_delta_b*x_delta_2_b, axis=0)
    # Finalize x_std_a and x_std_b (bias corrected)
    for x_std_a_i in x_std_a:
        x_std_a_i /= n_x_a[(...,) + (None,)*(x_std_a_i.ndim-1)] - 1
        x_std_a_i = np.sqrt(x_std_a_i)
    for x_std_b_i in x_std_b:
        x_std_b_i /= n_x_b[(...,) + (None,)*(x_std_b_i.ndim-1)] - 1
        x_std_b_i = np.sqrt(x_std_b_i)
    # Compute t-test
    for i, (m_a_i, s_a_i, m_b_i, s_b_i, sz_i) in enumerate(zip(
            x_mean_a, x_std_a, x_mean_b, x_std_b, embed_size)):
        for j, (m_a_ij, s_a_ij, n_a_j, m_b_ij, s_b_ij, n_b_j) in enumerate(zip(
                m_a_i, s_a_i, n_x_a, m_b_i, s_b_i, n_x_b)):
            ttest_j = np.zeros(m_a_ij.size, dtype=dtype_np)
            for k, (m_a_ijk, s_a_ijk, m_b_ijk, s_b_ijk) in enumerate(zip(
                    m_a_ij.ravel(), s_a_ij.ravel(), m_b_ij.ravel(),
                    s_b_ij.ravel())):
                ttest_j[k] += ttest_ind_from_stats(
                    m_a_ijk, s_a_ijk, n_a_j, m_b_ijk, s_b_ijk, n_b_j)[1]
            ttest[i][j] += ttest_j.reshape(sz_i)
    # Return results
    if multi_x:
        return ttest
    return ttest[0]
