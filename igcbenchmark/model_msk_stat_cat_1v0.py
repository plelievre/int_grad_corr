"""
Predict categorical image properties from images.

0v0 : True image statistic functions
1v0 : ConvNeXt and linear blocks

Author: Pierre Lelievre
"""

import os
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from torchutils import (
    AbstractModel, LinearBlock, ConvNeXtStem, ConvNeXtBlock, freeze_network,
    init_linear, set_dtld_seed)
from igc import (
    grad, int_grad, int_grad_corr, igc_error, int_grad_auto_corr, igac_error)
from igc.naive_2v0 import int_grad_naive, correlation_naive, ttest_naive
from igc.bsc_1v0 import bsl_shap, bsl_shap_corr

from .model_msk_stat_1v0 import Dataset  # pylint: disable=W0611


# Modules


class Encoder(nn.Module):
    def __init__(self, img_size, n_cat, conv_stem_kernel, conv_sizes,
                 lin_sizes, stochastic_depth_prob=0.4, dropout=0.2,
                 dropout_min_size=16, act_type='mish'):
        super().__init__()
        self.n_cat = n_cat
        # Convolution stem
        if conv_sizes is None:
            conv_sizes = (1,)
        conv_out_hw = img_size
        self.stem = None
        if conv_stem_kernel is not None:
            self.stem = ConvNeXtStem(1, conv_sizes[0], conv_stem_kernel)
            conv_out_hw = self.stem.get_output_spatial_size(conv_out_hw)
        # Convolution block
        self.conv = None
        if (conv_stem_kernel is not None) and (len(conv_sizes) > 1):
            self.conv = ConvNeXtBlock(
                conv_sizes, stochastic_depth_prob=stochastic_depth_prob)
            conv_out_hw = self.conv.get_output_spatial_size(conv_out_hw)
        # Linear block
        if lin_sizes is None:
            lin_sizes = tuple()
        lin_sizes = (conv_out_hw**2 * conv_sizes[-1],) + lin_sizes
        self.lin = None
        if len(lin_sizes) > 1:
            self.lin = LinearBlock(
                lin_sizes, dropout, dropout_min_size, act_type)
        # Output layer
        self.lin_out = nn.Linear(lin_sizes[-1], self.n_cat, bias=False)
        init_linear(self.lin_out, init_gain_act='linear')

    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad_(True)

    def forward(self, x):
        # Convolution stem
        if self.stem is not None:
            x = self.stem(x)
        # Convolution block
        if self.conv is not None:
            x = self.conv(x)
        # Linear block
        x = x.flatten(start_dim=1)
        if self.lin is not None:
            x = self.lin(x)
        # Output layer
        x = self.lin_out(x)
        return nn.functional.log_softmax(x, dim=1)

    def get_y_r(self, log_p):
        label = torch.argmax(log_p, dim=1)
        return nn.functional.one_hot(  # pylint: disable=E1102
            label, self.n_cat)

    def nll_loss(self, log_p, y):
        return -1.0 * torch.sum(log_p * y)

    def ac_loss(self, log_p, y):
        return torch.sum(self.get_y_r(log_p) * y)


# Model


class Model(AbstractModel):
    # Default parameters
    conv_stem_kernel = 2
    conv_sizes = (16, 32, 64, 128, 256)
    stochastic_depth_prob = 0.5
    lin_sizes = (128, 16)
    dropout = 0.25
    dropout_min_size = 16
    act_type = 'mish'
    learning_rate = 5e-5
    scheduler_decay = 0.9
    scheduler_patience = 4
    scheduler_min_lr = 1e-6
    save_every_n_epoch = 1
    maximum_saves = 1
    seed = 100
    dtype = torch.float32
    project_path = os.path.dirname(__file__)
    def __init__(self, dataset, model_name='msk_stat_cat_1v0', trainable=False,
                 device=None, parameters=None):
        self.dtst = dataset
        super().__init__(model_name, device, parameters)
        # Network
        self.network = Encoder(
            self.dtst.img_size, self.dtst.imst_set_val.n_cat,
            self.conv_stem_kernel, self.conv_sizes, self.lin_sizes,
            self.stochastic_depth_prob, self.dropout, self.dropout_min_size,
            self.act_type)
        self.network.to(self.device)
        # Optimizer
        self.optimizer, self.scheduler = None, None
        if trainable:
            self.optimizer = optim.AdamW(
                self.network.parameters(), lr=self.learning_rate)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=self.scheduler_decay,
                patience=self.scheduler_patience, min_lr=self.scheduler_min_lr)

    def train(self, n_epoch=1, batch_size=64, num_workers=0):
        self.restore('last', training=True)
        writer = SummaryWriter(self.logs_path)
        train_dtld = self.dtst.train_dtld(batch_size, self.seed, num_workers)
        val_dtld = self.dtst.val_dtld(batch_size, num_workers=num_workers)
        for _ in range(int(n_epoch)):
            self.current_epoch += 1
            current_seed = self.seed + self.current_epoch
            torch.manual_seed(current_seed)
            set_dtld_seed(train_dtld, current_seed)
            # Training ########################################################
            self.network.train()
            train_nll_avg = 0.0
            for i, (x, y) in enumerate(tqdm(
                    train_dtld, total=len(train_dtld),
                    desc=f'trn {self.current_epoch:04d}', leave=False)):
                # Increment step
                self.current_step += 1
                # Send data to the device
                x = x.to(self.device)
                y = y.to(self.device)
                # Optimization
                self.network.zero_grad(set_to_none=True)
                log_p = self.network(x)
                nll_loss = self.network.nll_loss(log_p, y)
                nll_loss.backward()
                self.optimizer.step()
                # Summary
                train_nll_avg += nll_loss.item()
            train_nll_avg /= float(len(train_dtld) * batch_size)
            writer.add_scalar(
                'train/nll_loss', train_nll_avg, self.current_epoch)
            # Validation ######################################################
            self.network.eval()
            with torch.no_grad():
                val_nll_avg = 0.0
                for i, (x, y) in enumerate(tqdm(
                        val_dtld, total=len(val_dtld),
                        desc=f'val {self.current_epoch:04d}', leave=False)):
                    # Send data to the device
                    x = x.to(self.device)
                    y = y.to(self.device)
                    # Evaluation
                    log_p = self.network(x)
                    nll_loss = self.network.nll_loss(log_p, y)
                    # Summary
                    val_nll_avg += nll_loss.item()
            val_nll_avg /= float(self.dtst.n_val)
            writer.add_scalar(
                'validation/nll_loss', val_nll_avg, self.current_epoch)
            # Save checkpoint #################################################
            if not self.current_epoch % self.save_every_n_epoch:
                self.save(val_nll_avg)
            # LR scheduler ####################################################
            for i, param_group in enumerate(self.optimizer.param_groups):
                writer.add_scalar(
                    f'learning_rate/lr_{i:d}', float(param_group['lr']),
                    self.current_epoch)
            self.scheduler.step(val_nll_avg)
            self.clean_saves()
        writer.close()
        return self

    @torch.no_grad()
    def score(self, batch_size=64, checkpoint='best'):
        # Restore
        self.restore(checkpoint)
        # Eval
        self.network.eval()
        val_nll_avg, val_ac_avg = 0.0, 0.0
        dtld = self.dtst.val_dtld(batch_size)
        for _, (x, y) in enumerate(tqdm(dtld, total=len(dtld), desc='score')):
            # Send data to the device
            x = x.to(self.device)
            y = y.to(self.device)
            # Evaluation
            log_p = self.network(x)
            nll_loss = self.network.nll_loss(log_p, y)
            ac_loss = self.network.ac_loss(log_p, y)
            # Summary
            val_nll_avg += nll_loss.item()
            val_ac_avg += ac_loss.item()
        val_nll_avg /= float(self.dtst.n_val)
        val_ac_avg /= float(self.dtst.n_val)
        return val_nll_avg, val_ac_avg

    def _fwd(self, x, checkpoint='best'):
        # Restore
        self.restore(checkpoint)
        # Prepare inputs
        x.requires_grad_(True)
        # Prepare model
        self.network.eval()
        # Eval
        y_r = torch.exp(self.network(x))  # log_prob to prob
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

    def _igc_params(self, checkpoint, num_workers=0):
        x_size = (1, self.dtst.img_size, self.dtst.img_size)
        y_size = self.dtst.imst_set_val.n_cat
        dtld_func = self.dtst.val_dtld
        dtld_kwargs = {'num_workers': num_workers}
        fwd_kwargs = {'checkpoint': checkpoint}
        return x_size, y_size, dtld_func, dtld_kwargs, fwd_kwargs

    def _igc_format(self, igc_, batched=False):
        if batched:
            return igc_[:, :, 0]
        return igc_[:, 0]

    @freeze_network
    def grad(self, n_x=None, y_idx=None, x_batch_size=1, y_batch_size=None,
             x_seed=None, checkpoint='best', num_workers=0):
        x_size, y_size, dtld_func, dtld_kw, fwd_kw = self._igc_params(
            checkpoint, num_workers)
        x, y, y_r, grad_ = grad(
            self._fwd, self._bwd, n_x, y_idx, x_size, y_size,
            dtld_func=dtld_func, x_batch_size=x_batch_size,
            y_batch_size=y_batch_size, x_seed=x_seed, dtype=self.dtype,
            device=self.device, dtld_kwargs=dtld_kw, fwd_kwargs=fwd_kw)
        return x, y, y_r, self._igc_format(grad_, batched=True)

    @freeze_network
    def int_grad(self, n_x=None, x_0=8, y_idx=None, n_steps=64, x_batch_size=1,
                 x_0_batch_size=1, y_batch_size=None, x_seed=None,
                 x_0_seed=100, check_error=False, checkpoint='best',
                 num_workers=0):
        x_size, y_size, dtld_func, dtld_kw, fwd_kw = self._igc_params(
            checkpoint, num_workers)
        x, y, y_0, y_r, int_grad_ = int_grad(
            self._fwd, self._bwd, n_x, x_0, y_idx, n_steps, x_size, y_size,
            dtld_func=dtld_func, x_batch_size=x_batch_size,
            x_0_batch_size=x_0_batch_size, y_batch_size=y_batch_size,
            x_seed=x_seed, x_0_seed=x_0_seed, dtype=self.dtype,
            device=self.device, dtld_kwargs=dtld_kw, fwd_kwargs=fwd_kw,
            check_error=check_error)
        return x, y, y_0, y_r, self._igc_format(int_grad_, batched=True)

    @freeze_network
    def int_grad_corr(self, x_0=8, y_idx=None, n_steps=64, x_batch_size=1,
                      x_0_batch_size=1, y_batch_size=None, x_0_seed=100,
                      save_results=True, check_error=False, checkpoint='best',
                      suffix='', num_workers=0, n_x=None):
        if suffix:
            suffix = '_' + suffix
        x_size, y_size, dtld_func, dtld_kw, fwd_kw = self._igc_params(
            checkpoint, num_workers)
        int_grad_corr_ = int_grad_corr(
            self._fwd, self._bwd, dtld_func, x_0, y_idx, n_steps, x_size,
            y_size, x_batch_size=x_batch_size, x_0_batch_size=x_0_batch_size,
            y_batch_size=y_batch_size, x_0_seed=x_0_seed, dtype=self.dtype,
            device=self.device, dtld_kwargs=dtld_kw, fwd_kwargs=fwd_kw,
            check_error=check_error, n_x=n_x)
        int_grad_corr_ = self._igc_format(int_grad_corr_)
        # Save results
        if save_results:
            np.savez(self.get_result_path(
                f'int_grad_corr{suffix}.npz'), data=int_grad_corr_)
        # Return results
        return int_grad_corr_

    @freeze_network
    def int_grad_auto(self, x_0=8, y_idx=None, n_steps=64, x_batch_size=1,
                      x_0_batch_size=1, y_batch_size=None, x_0_seed=100,
                      save_results=True, check_error=False, checkpoint='best',
                      suffix='', num_workers=0, n_x=None):
        if suffix:
            suffix = '_' + suffix
        x_size, y_size, dtld_func, dtld_kw, fwd_kw = self._igc_params(
            checkpoint, num_workers)
        int_grad_auto_corr_ = int_grad_auto_corr(
            self._fwd, self._bwd, dtld_func, x_0, y_idx, n_steps, x_size,
            y_size, x_batch_size=x_batch_size, x_0_batch_size=x_0_batch_size,
            y_batch_size=y_batch_size, x_0_seed=x_0_seed, dtype=self.dtype,
            device=self.device, dtld_kwargs=dtld_kw, fwd_kwargs=fwd_kw,
            check_error=check_error, n_x=n_x)
        int_grad_auto_corr_ = self._igc_format(int_grad_auto_corr_)
        # Save results
        if save_results:
            np.savez(self.get_result_path(
                f'int_grad_auto{suffix}.npz'), data=int_grad_auto_corr_)
        # Return results
        return int_grad_auto_corr_

    @freeze_network
    def int_grad_naive(self, x_0=8, y_idx=None, n_steps=64, x_batch_size=1,
                       x_0_batch_size=1, y_batch_size=None, x_0_seed=100,
                       save_results=True, check_error=False,
                       checkpoint='best', suffix='', num_workers=0,
                       n_x=None):
        if suffix:
            suffix = '_' + suffix
        x_size, y_size, dtld_func, dtld_kw, fwd_kw = self._igc_params(
            checkpoint, num_workers)
        int_grad_mean_, int_grad_std_ = int_grad_naive(
            self._fwd, self._bwd, dtld_func, x_0, y_idx, n_steps, x_size,
            y_size, x_batch_size=x_batch_size, x_0_batch_size=x_0_batch_size,
            y_batch_size=y_batch_size, x_0_seed=x_0_seed, dtype=self.dtype,
            device=self.device, dtld_kwargs=dtld_kw, fwd_kwargs=fwd_kw,
            check_error=check_error, n_x=n_x)
        int_grad_mean_ = self._igc_format(int_grad_mean_)
        int_grad_std_ = self._igc_format(int_grad_std_)
        # Save results
        if save_results:
            np.savez(self.get_result_path(
                f'int_grad_mean{suffix}.npz'), data=int_grad_mean_)
            np.savez(self.get_result_path(
                f'int_grad_std{suffix}.npz'), data=int_grad_std_)
        # Return results
        return int_grad_mean_, int_grad_std_

    @torch.no_grad()
    def corr_naive(self, y_idx=None, x_batch_size=1, y_batch_size=None,
                   x_seed=None, save_results=True, checkpoint='best',
                   suffix='', num_workers=0, n_x=None):
        if suffix:
            suffix = '_' + suffix
        x_size, y_size, dtld_func, dtld_kw, _ = self._igc_params(
            checkpoint, num_workers)
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
                    x_seed=None, save_results=True, checkpoint='best',
                    suffix='', num_workers=0, n_x=None):
        if suffix:
            suffix = '_' + suffix
        cat_ranges = (0.5, 0.5)
        x_size, y_size, dtld_func, dtld_kw, _ = self._igc_params(
            checkpoint, num_workers)
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
    def igc_error(self, igc_name, y_idx=None, x_batch_size=1,
                  checkpoint='best', num_workers=0):
        # Load IGC data
        igc = np.load(self.get_result_path(igc_name))['data']
        # Restore
        self.restore(checkpoint)
        # Compute IGC error
        x_size, y_size, dtld_func, dtld_kw, fwd_kw = self._igc_params(
            checkpoint, num_workers)
        error = igc_error(
            igc, self._fwd, dtld_func, y_idx, x_size, y_size,
            x_batch_size=x_batch_size, dtype=self.dtype, device=self.device,
            dtld_kwargs=dtld_kw, fwd_kwargs=fwd_kw)
        return error

    @torch.no_grad()
    def igac_error(self, igac_name):
        # Load IGC data
        igac = np.load(self.get_result_path(igac_name))['data']
        return igac_error(igac)

    @torch.no_grad()
    def _forward(self, x, y_idx, checkpoint='best'):
        # Restore
        self.restore(checkpoint)
        # Prepare model
        self.network.eval()
        # Eval
        y_r = torch.exp(self.network(x))
        y_r = torch.gather(y_r, dim=1, index=y_idx.unsqueeze(dim=1))
        return y_r.squeeze(dim=1).cpu().numpy()

    @torch.no_grad()
    def bsl_shap(self, n_x=None, x_0=8, y_idx=None, n_iter=8,
                 x_0_batch_size=None, x_seed=None, x_0_seed=100,
                 check_error=False, checkpoint='best', num_workers=0):
        x_size, y_size, dtld_func, dtld_kw, fwd_kw = self._igc_params(
            checkpoint, num_workers)
        x, y, y_0, y_r, bsl_shap_ = bsl_shap(
            self._forward, n_x, y_idx, x_0, n_iter, x_size, y_size, dtld_func,
            x_0_batch_size, x_seed, x_0_seed, self.dtype, self.device, dtld_kw,
            fwd_kw, check_error)
        return x, y, y_0, y_r, self._igc_format(bsl_shap_, batched=True)

    @torch.no_grad()
    def bsl_shap_corr(self, x_0=8, y_idx=None, n_iter=8, x_0_batch_size=None,
                      x_0_seed=100, save_results=True, check_error=False,
                      checkpoint='best', suffix='', num_workers=0, n_x=None):
        if suffix:
            suffix = '_' + suffix
        x_size, y_size, dtld_func, dtld_kw, fwd_kw = self._igc_params(
            checkpoint, num_workers)
        bsl_shap_corr_ = bsl_shap_corr(
            self._forward, dtld_func, x_size, y_size, y_idx, x_0, n_iter,
            x_0_batch_size, x_0_seed, self.dtype, self.device, dtld_kw, fwd_kw,
            check_error, n_x)
        bsl_shap_corr_ = self._igc_format(bsl_shap_corr_)
        # Save results
        if save_results:
            np.savez(self.get_result_path(
                f'bsl_shap_corr{suffix}.npz'), data=bsl_shap_corr_)
        # Return results
        return bsl_shap_corr_
