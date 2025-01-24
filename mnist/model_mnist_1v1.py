"""
Simple convolutional/linear model trained on MNIST to test integrated gradient
correlation dataset-wise attribution method on a classification task.

Author: Pierre Lelievre
"""

import os
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.utils.tensorboard import SummaryWriter

from mnist.mnist import MNIST
from torchutils import (
    AbstractModel, LinearBlock, Conv2dBlock, freeze_network, init_linear,
    set_dtld_seed, set_worker_seed, fix_cpu_affinity)
from igc import grad, int_grad, int_grad_corr, igc_error
from igc.bsc_1v0 import bsl_shap, bsl_shap_corr


# Dataset


class _Dataset(TorchDataset):
    def __init__(self, images, labels, n_digits, seed=None):
        self.images = images
        self.labels = labels
        self.n_digits = n_digits
        self.rng = None
        if seed is not None:
            self.rng = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        return image, nn.functional.one_hot(  # pylint: disable=E1102
            label, self.n_digits)


class Dataset:
    def __init__(self):
        # Load data
        self.mnist = MNIST()
        self.train_images = self.mnist.train_images
        self.val_images = self.mnist.test_images
        self.train_labels = self.mnist.train_labels
        self.val_labels = self.mnist.test_labels
        # Standardize data
        images_mean = np.mean(self.train_images)
        images_std = np.std(self.train_images)
        self.train_images -= images_mean
        self.train_images /= images_std
        self.val_images -= images_mean
        self.val_images /= images_std
        # Informations
        self.n_digits = self.mnist.n_digits
        self.n_train, self.img_size = self.train_images.shape[:2]
        self.n_val = self.val_labels.shape[0]
        # Print info
        print(f'Train : {self.n_train}')
        print(f'Val   : {self.n_val}')

    @torch.no_grad()
    def train_dtld(self, batch_size, seed=100, num_workers=0):
        return DataLoader(
            _Dataset(self.train_images, self.train_labels, self.n_digits),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            drop_last=True, worker_init_fn=set_worker_seed,
            generator=torch.Generator().manual_seed(seed))

    @torch.no_grad()
    def val_dtld(self, batch_size, seed=None, num_workers=0):
        if seed is None:
            return DataLoader(
                _Dataset(self.val_images, self.val_labels, self.n_digits),
                batch_size=batch_size, shuffle=False, num_workers=num_workers,
                worker_init_fn=fix_cpu_affinity)
        return DataLoader(
            _Dataset(self.val_images, self.val_labels, self.n_digits),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            worker_init_fn=fix_cpu_affinity,
            generator=torch.Generator().manual_seed(seed))

    def compute_mean_digits(self):
        average_digits = np.zeros(
            (self.n_digits, self.img_size, self.img_size), dtype=np.float32)
        for i in range(self.n_digits):
            average_digits[i] += np.mean(
                self.val_images[self.val_labels == i], axis=0)
        return average_digits

    def compute_mean_digit(self):
        return np.mean(self.val_images, axis=0)

    def digit_diff_probabilities(self, n_samples=10000, seed=100):
        rng = np.random.default_rng(seed)
        diff_probs = np.zeros(
            (self.n_digits, self.img_size, self.img_size), dtype=np.float32)
        for i in range(self.n_digits):
            mask = self.val_labels == i
            digit_images = rng.choice(
                self.val_images[mask], size=n_samples, axis=0)
            no_digit_images = rng.choice(
                self.val_images[np.invert(mask)], size=n_samples, axis=0)
            diff_probs[i] += np.mean(
                digit_images * no_digit_images < 0.0, axis=0)
        return diff_probs

    def get_10_digits(self, digit=None, seed=100):
        rng = np.random.default_rng(seed)
        x = np.zeros(
            (self.n_digits, self.img_size, self.img_size), dtype=np.float32)
        if digit is None:
            digits = np.arange(self.n_digits)
        else:
            digits = (digit,) * self.n_digits
        for i, d in enumerate(digits):
            x[i] += rng.choice(self.val_images[self.val_labels == d])
        return x


# Modules


class MultiConvLin(nn.Module):
    def __init__(self, img_size, n_digits, conv_sizes, lin_sizes, dropout=0.2,
                 dropout_min_size=16, act_type='mish'):
        super().__init__()
        self.n_digits = n_digits
        # Convolution block
        if conv_sizes is None:
            conv_sizes = tuple()
        conv_out_hw = img_size
        for _ in range(len(conv_sizes)):
            conv_out_hw = int(0.5*(conv_out_hw - 3) + 1)
        conv_sizes = (1,) + conv_sizes
        self.conv = None
        if len(conv_sizes) > 1:
            self.conv = Conv2dBlock(
                conv_sizes, dropout=dropout, dropout_min_size=dropout_min_size,
                act_type=act_type)
        # Linear block
        if lin_sizes is None:
            lin_sizes = tuple()
        lin_sizes = (conv_out_hw**2 * conv_sizes[-1],) + lin_sizes
        self.lin = None
        if len(lin_sizes) > 1:
            self.lin = LinearBlock(
                lin_sizes, dropout, dropout_min_size, act_type)
        # Output layer
        self.lin_out = nn.Linear(lin_sizes[-1], self.n_digits, bias=False)
        init_linear(self.lin_out, init_gain_act='linear')
        # Misc
        self.dropout = nn.Dropout(dropout)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad_(True)

    def forward(self, x):
        x = self.dropout(x)
        x = x.unsqueeze(dim=1)
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
            label, self.n_digits)

    def nll_loss(self, log_p, y):
        return -1.0 * torch.sum(log_p * y)

    def ac_loss(self, log_p, y):
        return torch.sum(self.get_y_r(log_p) * y)


# Model


class Model(AbstractModel):
    # Default parameters
    conv_sizes = (64, 128)
    lin_sizes = (128, 64, 32, 16)
    dropout = 0.2
    dropout_min_size = 16
    act_type = 'mish'
    learning_rate = 1e-5
    scheduler_decay = 0.9
    scheduler_patience = 4
    scheduler_min_lr = 1e-6
    save_every_n_epoch = 1
    maximum_saves = 1
    seed = 100
    dtype = torch.float32
    project_path = os.path.dirname(__file__)
    def __init__(self, dataset, model_name='mnist_1v1', trainable=False,
                 device=None, parameters=None):
        self.dtst = dataset
        super().__init__(model_name, device, parameters)
        # Network
        self.network = MultiConvLin(
            self.dtst.img_size, self.dtst.n_digits, self.conv_sizes,
            self.lin_sizes, self.dropout, self.dropout_min_size, self.act_type)
        self.network.to(self.device)
        # Optimizer
        self.optimizer, self.scheduler = None, None
        if trainable:
            self.optimizer = optim.Adam(
                self.network.parameters(), lr=self.learning_rate)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=self.scheduler_decay,
                patience=self.scheduler_patience, min_lr=self.scheduler_min_lr)

    def train(self, n_epoch=1, batch_size=64):
        self.restore('last', training=True)
        writer = SummaryWriter(self.logs_path)
        train_dtld = self.dtst.train_dtld(batch_size, self.seed)
        val_dtld = self.dtst.val_dtld(batch_size)
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
        # Prepare x
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
        x_size = (self.dtst.img_size, self.dtst.img_size)
        y_size = self.dtst.n_digits
        dtld_func = self.dtst.val_dtld
        dtld_kwargs = {'num_workers': num_workers}
        fwd_kwargs = {'checkpoint': checkpoint}
        return x_size, y_size, dtld_func, dtld_kwargs, fwd_kwargs

    def _igc_format(self, igc_, batched=False):
        if batched:
            return igc_
        return igc_

    @freeze_network
    def grad_x(self, x, y_idx=None, x_batch_size=1, y_batch_size=None,
               checkpoint='best', num_workers=0):
        x_size, y_size, dtld_func, dtld_kw, fwd_kw = self._igc_params(
            checkpoint, num_workers)
        _, _, y_r, grad_ = grad(
            self._fwd, self._bwd, x, y_idx, x_size, y_size,
            dtld_func=dtld_func, x_batch_size=x_batch_size,
            y_batch_size=y_batch_size, dtype=self.dtype, device=self.device,
            dtld_kwargs=dtld_kw, fwd_kwargs=fwd_kw)
        return y_r, self._igc_format(grad_, batched=True)

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
    def int_grad_x(self, x, x_0=32, y_idx=None, n_steps=32, x_batch_size=1,
                   x_0_batch_size=1, y_batch_size=None, x_0_seed=100,
                   check_error=False, checkpoint='best', num_workers=0):
        x_size, y_size, dtld_func, dtld_kw, fwd_kw = self._igc_params(
            checkpoint, num_workers)
        _, _, y_0, y_r, int_grad_ = int_grad(
            self._fwd, self._bwd, x, x_0, y_idx, n_steps, x_size, y_size,
            dtld_func=dtld_func, x_batch_size=x_batch_size,
            x_0_batch_size=x_0_batch_size, y_batch_size=y_batch_size,
            x_0_seed=x_0_seed, dtype=self.dtype, device=self.device,
            dtld_kwargs=dtld_kw, fwd_kwargs=fwd_kw, check_error=check_error)
        return y_0, y_r, self._igc_format(int_grad_, batched=True)

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
    def _forward(self, x, y_idx, checkpoint='best'):
        # Restore
        self.restore(checkpoint)
        # Prepare model
        self.network.eval()
        # Eval
        y_r = torch.exp(self.network(x))  # log_prob to prob
        y_r = torch.gather(y_r, dim=1, index=y_idx.unsqueeze(dim=1))
        return y_r.squeeze(dim=1).cpu().numpy()

    @torch.no_grad()
    def bsl_shap_x(self, x, y_idx=None, x_0=32, n_iter=32, x_0_batch_size=None,
                   x_0_seed=100, check_error=False, checkpoint='best',
                   num_workers=0):
        x_size, y_size, dtld_func, dtld_kw, fwd_kw = self._igc_params(
            checkpoint, num_workers)
        _, _, y_0, y_r, bsl_shap_ = bsl_shap(
            self._forward, x, y_idx, x_0, n_iter, x_size, y_size, dtld_func,
            x_0_batch_size, x_0_seed=x_0_seed, dtype=self.dtype,
            device=self.device, dtld_kwargs=dtld_kw, forward_kwargs=fwd_kw,
            check_error=check_error)
        return y_0, y_r, self._igc_format(bsl_shap_, batched=True)

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
