"""
Predict image statistics from images.

0v0 : True image statistic functions
1v0 : ConvNeXt and linear blocks

Author: Pierre Lelievre
"""

import os
import numpy as np
from scipy.stats import norm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset as TorchDataset

from torchutils import AbstractModel, fix_cpu_affinity
from igc import (
    grad, int_grad, int_grad_corr, igc_error, int_grad_auto_corr, igac_error)
from igc.naive_2v0 import int_grad_naive, correlation_naive, ttest_naive
from igc.bsc_1v0 import bsl_shap, bsl_shap_corr

from .imgstat_1v0 import ImgStatSet


# Dataset


class _Dataset(TorchDataset):
    def __init__(self, img_set, imst_set, seed=None):
        self.img_set = img_set
        self.imst_set = imst_set
        self.rng = None
        if seed is not None:
            self.rng = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.img_set.n

    def __getitem__(self, idx):
        img = self.img_set.torch(idx, self.rng)
        if self.imst_set.categorical:
            imst_set = self.imst_set.torch_onehot(idx, self.rng)
        else:
            imst_set = self.imst_set.torch(idx, self.rng)
        return img, imst_set


class Dataset:
    def __init__(self, imgstat_set_name, img_size=64):
        self.img_size = img_size
        # Load data
        self.imst_set = ImgStatSet(imgstat_set_name).load()
        self.masks = self.imst_set.get_masks(self.img_size)
        self.img_set = self.imst_set.get_img_set(self.img_size)
        # Standardize data
        self.img_set.standardize()
        self.imst_set.standardize()
        # Print info
        print(f'n : {self.img_set.n}')

    @torch.no_grad()
    def dtld(self, batch_size, seed=None, num_workers=0):
        if seed is None:
            return DataLoader(
                _Dataset(self.img_set, self.imst_set),
                batch_size=batch_size, shuffle=False, num_workers=num_workers,
                worker_init_fn=fix_cpu_affinity)
        return DataLoader(
            _Dataset(self.img_set, self.imst_set),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            worker_init_fn=fix_cpu_affinity,
            generator=torch.Generator().manual_seed(seed))


# Modules


class Encoder(nn.Module):
    def __init__(self, img_set, imst_set, device, seed=100):
        super().__init__()
        self.x_mean = torch.as_tensor(img_set.mean).to(device)
        self.x_std = torch.as_tensor(img_set.std).to(device)
        self.categorical = imst_set.categorical
        self.n_imst = imst_set.n_imst
        if self.categorical:
            self.n_imst = imst_set.n_cat
        self.funcs = tuple(imst.extract for imst in imst_set.imsts)
        self.masks = imst_set.get_masks(img_set.img_size)
        self.y_means, self.y_sts = None, None
        if not self.categorical:
            self.y_means = torch.concatenate(tuple(torch.as_tensor(
                imst.mean).to(device) for imst in imst_set.imsts), dim=1)
            self.y_stds = torch.concatenate(tuple(torch.as_tensor(
                imst.std).to(device) for imst in imst_set.imsts), dim=1)
        self.rng = torch.Generator().manual_seed(seed)
        self.kwargs = None
        if self.categorical:
            self.kwargs = {'beta': 100.0}  # harder softmax

    def forward(self, x):
        batch_size = x.size(0)
        if self.x_mean is not None:
            x = (x * self.x_std) + self.x_mean
        if self.categorical:
            return self.funcs[0](x, self.masks[0], self.rng, self.kwargs)
        y = torch.zeros(
            (batch_size, self.n_imst), dtype=x.dtype, device=x.device)
        for j, (func, mask) in enumerate(zip(self.funcs, self.masks)):
            y[:, j] += func(x, mask, self.rng, self.kwargs)
        if self.y_means is not None:
            y = (y - self.y_means) / self.y_stds
        return y


# Model


class Model(AbstractModel):
    # Default parameters
    seed = 100
    dtype = torch.float32
    project_path = os.path.dirname(__file__)
    def __init__(self, dataset, model_name='msk_stat_0v0', device=None):
        self.dtst = dataset
        super().__init__(model_name, device)
        # Network
        self.network = Encoder(
            self.dtst.img_set, self.dtst.imst_set, self.device, self.seed)

    def _fwd(self, x):
        # Prepare inputs
        x.requires_grad_(True)
        # Prepare model
        self.network.eval()
        # Eval
        y_r = self.network(x)
        return y_r

    def _bwd(self, x, y_r, y_idx):
        # Reset gradients
        if x.grad is not None:
            x.grad = None
        self.network.zero_grad(set_to_none=True)
        # Compute gradients
        y_r_ = torch.gather(y_r, dim=1, index=y_idx.unsqueeze(dim=1))
        y_r_.backward(gradient=torch.ones_like(y_r_), retain_graph=True)
        return y_r_.squeeze(dim=1).detach().cpu().numpy(), x.grad.cpu().numpy()

    def _igc_params(self, num_workers=0):
        x_size = (1, self.dtst.img_size, self.dtst.img_size)
        y_size = self.dtst.imst_set.n_imst
        if self.dtst.imst_set.categorical:
            y_size = self.dtst.imst_set.n_cat
        dtld_func = self.dtst.dtld
        dtld_kwargs = {'num_workers': num_workers}
        return x_size, y_size, dtld_func, dtld_kwargs

    def _igc_format(self, igc_, batched=False):
        if batched:
            return igc_[:, :, 0]
        return igc_[:, 0]

    def grad(self, n_x=None, y_idx=None, x_batch_size=1, y_batch_size=None,
             x_seed=None, num_workers=0):
        x_size, y_size, dtld_func, dtld_kw = self._igc_params(num_workers)
        x, y, y_r, grad_ = grad(
            self._fwd, self._bwd, n_x, y_idx, x_size, y_size,
            dtld_func=dtld_func, x_batch_size=x_batch_size,
            y_batch_size=y_batch_size, x_seed=x_seed, dtype=self.dtype,
            device=self.device, dtld_kwargs=dtld_kw)
        return x, y, y_r, self._igc_format(grad_, batched=True)

    def int_grad(self, n_x=None, x_0=8, y_idx=None, n_steps=64, x_batch_size=1,
                 x_0_batch_size=1, y_batch_size=None, x_seed=None,
                 x_0_seed=100, check_error=False, num_workers=0):
        x_size, y_size, dtld_func, dtld_kw = self._igc_params(num_workers)
        x, y, y_0, y_r, int_grad_ = int_grad(
            self._fwd, self._bwd, n_x, x_0, y_idx, n_steps, x_size, y_size,
            dtld_func=dtld_func, x_batch_size=x_batch_size,
            x_0_batch_size=x_0_batch_size, y_batch_size=y_batch_size,
            x_seed=x_seed, x_0_seed=x_0_seed, dtype=self.dtype,
            device=self.device, dtld_kwargs=dtld_kw,  check_error=check_error)
        return x, y, y_0, y_r, self._igc_format(int_grad_, batched=True)

    def int_grad_corr(self, x_0=8, y_idx=None, n_steps=64, x_batch_size=1,
                      x_0_batch_size=1, y_batch_size=None, x_0_seed=100,
                      save_results=True, check_error=False, suffix='',
                      num_workers=0, n_x=None):
        if suffix:
            suffix = '_' + suffix
        x_size, y_size, dtld_func, dtld_kw = self._igc_params(num_workers)
        int_grad_corr_ = int_grad_corr(
            self._fwd, self._bwd, dtld_func, x_0, y_idx, n_steps, x_size,
            y_size, x_batch_size=x_batch_size, x_0_batch_size=x_0_batch_size,
            y_batch_size=y_batch_size, x_0_seed=x_0_seed, dtype=self.dtype,
            device=self.device, dtld_kwargs=dtld_kw, check_error=check_error,
            n_x=n_x)
        int_grad_corr_ = self._igc_format(int_grad_corr_)
        # Save results
        if save_results:
            np.savez(self.get_result_path(
                f'int_grad_corr{suffix}.npz'), data=int_grad_corr_)
        # Return results
        return int_grad_corr_

    def int_grad_auto(self, x_0=8, y_idx=None, n_steps=64, x_batch_size=1,
                      x_0_batch_size=1, y_batch_size=None, x_0_seed=100,
                      save_results=True, check_error=False, suffix='',
                      num_workers=0, n_x=None):
        if suffix:
            suffix = '_' + suffix
        x_size, y_size, dtld_func, dtld_kw = self._igc_params(num_workers)
        int_grad_auto_corr_ = int_grad_auto_corr(
            self._fwd, self._bwd, dtld_func, x_0, y_idx, n_steps, x_size,
            y_size, x_batch_size=x_batch_size, x_0_batch_size=x_0_batch_size,
            y_batch_size=y_batch_size, x_0_seed=x_0_seed, dtype=self.dtype,
            device=self.device, dtld_kwargs=dtld_kw, check_error=check_error,
            n_x=n_x)
        int_grad_auto_corr_ = self._igc_format(int_grad_auto_corr_)
        # Save results
        if save_results:
            np.savez(self.get_result_path(
                f'int_grad_auto{suffix}.npz'), data=int_grad_auto_corr_)
        # Return results
        return int_grad_auto_corr_

    def int_grad_naive(self, x_0=8, y_idx=None, n_steps=64, x_batch_size=1,
                       x_0_batch_size=1, y_batch_size=None, x_0_seed=100,
                       save_results=True, check_error=False, suffix='',
                       num_workers=0, n_x=None):
        if suffix:
            suffix = '_' + suffix
        x_size, y_size, dtld_func, dtld_kw = self._igc_params(num_workers)
        int_grad_mean, int_grad_std = int_grad_naive(
            self._fwd, self._bwd, dtld_func, x_0, y_idx, n_steps, x_size,
            y_size, x_batch_size=x_batch_size, x_0_batch_size=x_0_batch_size,
            y_batch_size=y_batch_size, x_0_seed=x_0_seed, dtype=self.dtype,
            device=self.device, dtld_kwargs=dtld_kw, check_error=check_error,
            n_x=n_x)
        int_grad_mean = self._igc_format(int_grad_mean)
        int_grad_std = self._igc_format(int_grad_std)
        # Save results
        if save_results:
            np.savez(self.get_result_path(
                f'int_grad_mean{suffix}.npz'), data=int_grad_mean)
            np.savez(self.get_result_path(
                f'int_grad_std{suffix}.npz'), data=int_grad_std)
        # Return results
        return int_grad_mean, int_grad_std

    @torch.no_grad()
    def corr_naive(self, y_idx=None, x_batch_size=1, y_batch_size=None,
                   x_seed=None, save_results=True, suffix='', num_workers=0,
                   n_x=None):
        if suffix:
            suffix = '_' + suffix
        x_size, y_size, dtld_func, dtld_kw = self._igc_params(num_workers)
        corr = correlation_naive(
            dtld_func, y_idx, x_size, y_size, x_batch_size=x_batch_size,
            y_batch_size=y_batch_size, x_seed=x_seed, dtype=self.dtype,
            device=self.device, dtld_kwargs=dtld_kw, n_x=n_x)
        corr = self._igc_format(corr)
        # Save results
        if save_results:
            np.savez(self.get_result_path(f'corr{suffix}.npz'), data=corr)
        # Return results
        return corr

    @torch.no_grad()
    def ttest_naive(self, y_idx=None, x_batch_size=1, y_batch_size=None,
                    x_seed=None, save_results=True, suffix='', num_workers=0,
                    n_x=None):
        if suffix:
            suffix = '_' + suffix
        cat_ranges = (norm.ppf(0.1), norm.ppf(0.9))
        if self.dtst.imst_set.categorical:
            cat_ranges = (0.5, 0.5)
        x_size, y_size, dtld_func, dtld_kw = self._igc_params(num_workers)
        ttest = ttest_naive(
            dtld_func, y_idx, cat_ranges, x_size, y_size,
            x_batch_size=x_batch_size, y_batch_size=y_batch_size,
            x_seed=x_seed, dtype=self.dtype, device=self.device,
            dtld_kwargs=dtld_kw, n_x=n_x)
        ttest = self._igc_format(ttest)
        # Save results
        if save_results:
            np.savez(self.get_result_path(f'ttest{suffix}.npz'), data=ttest)
        # Return results
        return ttest

    @torch.no_grad()
    def igc_error(self, igc_name, y_idx=None, x_batch_size=1, num_workers=0):
        # Load IGC data
        igc = np.load(self.get_result_path(igc_name))['data']
        # Compute IGC error
        x_size, y_size, dtld_func, dtld_kw = self._igc_params(num_workers)
        error = igc_error(
            igc, self._fwd, dtld_func, y_idx, x_size, y_size,
            x_batch_size=x_batch_size, dtype=self.dtype, device=self.device,
            dtld_kwargs=dtld_kw)
        return error

    @torch.no_grad()
    def igac_error(self, igac_name):
        # Load IGC data
        igac = np.load(self.get_result_path(igac_name))['data']
        return igac_error(igac)

    @torch.no_grad()
    def _forward(self, x, y_idx):
        # Prepare model
        self.network.eval()
        # Eval
        y_r = self.network(x)
        y_r = torch.gather(y_r, dim=1, index=y_idx.unsqueeze(dim=1))
        return y_r.squeeze(dim=1).cpu().numpy()

    @torch.no_grad()
    def bsl_shap(self, n_x=None, x_0=8, y_idx=None, n_iter=8,
                 x_0_batch_size=None, x_seed=None, x_0_seed=100,
                 check_error=False, num_workers=0):
        x_size, y_size, dtld_func, dtld_kw = self._igc_params(num_workers)
        x, y, y_0, y_r, bsl_shap_ = bsl_shap(
            self._forward, n_x, y_idx, x_0, n_iter, x_size, y_size, dtld_func,
            x_0_batch_size, x_seed, x_0_seed, self.dtype, self.device, dtld_kw,
            check_error=check_error)
        return x, y, y_0, y_r, self._igc_format(bsl_shap_, batched=True)

    @torch.no_grad()
    def bsl_shap_corr(self, x_0=8, y_idx=None, n_iter=8, x_0_batch_size=None,
                      x_0_seed=100, save_results=True, check_error=False,
                      suffix='', num_workers=0, n_x=None):
        if suffix:
            suffix = '_' + suffix
        x_size, y_size, dtld_func, dtld_kw = self._igc_params(num_workers)
        bsl_shap_corr_ = bsl_shap_corr(
            self._forward, dtld_func, x_size, y_size, y_idx, x_0, n_iter,
            x_0_batch_size, x_0_seed, self.dtype, self.device, dtld_kw,
            check_error=check_error, n_x=n_x)
        bsl_shap_corr_ = self._igc_format(bsl_shap_corr_)
        # Save results
        if save_results:
            np.savez(self.get_result_path(
                f'bsl_shap_corr{suffix}.npz'), data=bsl_shap_corr_)
        # Return results
        return bsl_shap_corr_
