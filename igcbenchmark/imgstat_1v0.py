"""
Image statistics utilities.

Author: Pierre Lelievre
"""

import os
import copy
import numpy as np
from tqdm import tqdm
from scipy.fft import next_fast_len, rfft2

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from torchutils import set_worker_seed

from .image_1v0 import (
    N_IMAGES, IMG_SIZE_PX, LUMINANCE_TYPES, get_std_dir, get_imgstat_dir,
    ImgSet)
from .mask_1v0 import MASK_INFO, Mask

# Pre-define functions
def cpt_mean(*a, **kwa):
    return compute_mean(*a, **kwa)
def cpt_std(*a, **kwa):
    return compute_std(*a, **kwa)
def cpt_w_sum(*a, **kwa):
    return compute_w_sum(*a, **kwa)
def cpt_max_mean(*a, **kwa):
    return compute_max_mean(*a, **kwa)
def cpt_argmax_mean(*a, **kwa):
    return compute_argmax_mean(*a, **kwa)
def cpt_max_std(*a, **kwa):
    return compute_max_std(*a, **kwa)
def cpt_argmax_std(*a, **kwa):
    return compute_argmax_std(*a, **kwa)
def cpt_max_sim(*a, **kwa):
    return compute_max_sim(*a, **kwa)
def cpt_argmax_sim(*a, **kwa):
    return compute_argmax_sim(*a, **kwa)
def cpt_fft_slope(*a, **kwa):
    return compute_fft_slope(*a, **kwa)
def cpt_max_std_rand(*a, **kwa):
    return compute_max_std_rand(*a, **kwa)
def cpt_max_sim_rand(*a, **kwa):
    return compute_max_sim_rand(*a, **kwa)
def cpt_argmax_sim_rand(*a, **kwa):
    return compute_argmax_sim_rand(*a, **kwa)

EPS = 1e-8
IMGSTAT_INFO = {
    # Image statistic name: (
    #     extraction function, mono channel, binary mask, multi mask,
    #     use_bbox, categorical)
    'mean': (cpt_mean, True, True, False, False, False),
    'std': (cpt_std, True, True, False, False, False),
    'w_sum': (cpt_w_sum, True, False, False, False, False),
    'max_mean': (cpt_max_mean, True, True, True, False, False),
    'argmax_mean': (cpt_argmax_mean, True, True, True, False, True),
    'max_std': (cpt_max_std, True, True, True, False, False),
    'argmax_std': (cpt_argmax_std, True, True, True, False, True),
    'max_sim': (cpt_max_sim, True, True, True, False, False),
    'argmax_sim': (cpt_argmax_sim, True, True, True, False, True),
    'fft_slope': (cpt_fft_slope, True, True, False, True, False),
    'max_std_rand': (cpt_max_std_rand, True, True, True, False, False),
    'max_sim_rand': (cpt_max_sim_rand, True, True, True, False, False),
    'argmax_sim_rand': (cpt_argmax_sim_rand, True, True, True, False, True),
}
IMGSTAT_SET_INFO = {
    # Image statistic set name: (Image statistic names)
    'main_01': (
        'comb_01-log2_Y-w_sum', 'ccat_01-log2_Y-max_mean',
        'ccat_04-log2_Y-max_sim_rand'),
}


# Utils FFT


def compute_fft_slope_np(img, freq_min=2):
    n = max(img.shape)
    n_opt = next_fast_len(n)
    fft_img_0 = np.abs(np.real(rfft2(img, (n_opt, n_opt))))
    n_y, n_x = fft_img_0.shape
    fft_img = fft_img_0[:n_x]
    fft_img[1:1+n_y-n_x] += fft_img_0[n_x:][::-1]
    fft_img[:, 1:] *= 2.0
    del fft_img_0
    y, x = np.indices(fft_img.shape)
    freq_map = np.round(np.sqrt(x**2 + y**2)).astype(np.int32)
    freq_max = 1+n_y-n_x
    denom = 4.0 * np.bincount(freq_map.ravel()) - 4 + EPS
    denom[0] += 1
    mean_amp = np.bincount(freq_map.ravel(), fft_img.ravel()) / denom
    mean_amp = mean_amp[freq_min:freq_max] + EPS
    freq = freq_map[0, freq_min:freq_max]
    return np.polyfit(np.log(freq), np.log(mean_amp), deg=1)[0]


def _lin_fit_torch(x, y):
    x = torch.vstack(
        (torch.ones(x.size(0), dtype=x.dtype, device=x.device), x)).T
    return torch.linalg.lstsq(  # pylint: disable=E1102
        x, y, driver='gelss').solution


def _compute_fft_slope(img, freq_min=2):
    n = max(img.size())
    n_opt = next_fast_len(n)
    fft_img_0 = torch.abs(torch.real(
        torch.fft.rfft2(img, (n_opt, n_opt))))  # pylint: disable=E1102
    n_y, n_x = fft_img_0.size()
    fft_img = fft_img_0[:n_x]
    fft_img[1:1+n_y-n_x] += torch.flip(fft_img_0[n_x:], dims=(0,))
    fft_img[:, 1:] *= 2.0
    del fft_img_0
    x = torch.arange(n_x, dtype=torch.int32, device=img.device)
    y, x = torch.meshgrid(x, x, indexing='ij')
    freq_map = torch.round(torch.sqrt(x**2 + y**2)).to(torch.int32)
    freq_max = 1+n_y-n_x
    denom = 4.0 * torch.bincount(freq_map.flatten()) - 4 + EPS
    denom[0] += 1
    mean_amp = torch.bincount(freq_map.flatten(), fft_img.flatten()) / denom
    mean_amp = torch.log(mean_amp[freq_min:freq_max] + EPS)
    freq = torch.log(freq_map[0, freq_min:freq_max])
    return _lin_fit_torch(freq, mean_amp)[1]


# Utils Statistics


def compute_mean(imgs, mask, rng=None):  # pylint: disable=W0613
    imgs_slct = imgs.flatten(start_dim=1).index_select(
        dim=1, index=mask.flatten().nonzero().squeeze())
    return torch.mean(imgs_slct, dim=1)


def compute_std(imgs, mask, rng=None):  # pylint: disable=W0613
    imgs_slct = imgs.flatten(start_dim=1).index_select(
        dim=1, index=mask.flatten().nonzero().squeeze())
    return torch.std(imgs_slct, dim=1)


def compute_w_sum(imgs, mask, rng=None):  # pylint: disable=W0613
    return torch.sum(imgs * mask, dim=(1, 2))


def compute_max_mean(imgs, masks, rng=None):  # pylint: disable=W0613
    means = torch.zeros(
        imgs.size(0), len(masks), dtype=imgs.dtype, device=imgs.device)
    for i, mask in enumerate(masks):
        imgs_slct = imgs.flatten(start_dim=1).index_select(
            dim=1, index=mask.flatten().nonzero().squeeze())
        means[:, i] += torch.mean(imgs_slct, dim=1)
    return torch.max(means, dim=1)[0]


# pylint: disable=W0613
def compute_argmax_mean(imgs, masks, rng=None, beta=None):
    means = torch.zeros(
        imgs.size(0), len(masks), dtype=imgs.dtype, device=imgs.device)
    for i, mask in enumerate(masks):
        imgs_slct = imgs.flatten(start_dim=1).index_select(
            dim=1, index=mask.flatten().nonzero().squeeze())
        means[:, i] += torch.mean(imgs_slct, dim=1)
    if beta is None:
        return torch.argmax(means, dim=1)
    return nn.functional.softmax(beta * means, dim=1)


def compute_max_std(imgs, masks, rng=None):  # pylint: disable=W0613
    stds = torch.zeros(
        imgs.size(0), len(masks), dtype=imgs.dtype, device=imgs.device)
    for i, mask in enumerate(masks):
        imgs_slct = imgs.flatten(start_dim=1).index_select(
            dim=1, index=mask.flatten().nonzero().squeeze())
        stds[:, i] += torch.std(imgs_slct, dim=1)
    return torch.max(stds, dim=1)[0]


# pylint: disable=W0613
def compute_argmax_std(imgs, masks, rng=None, beta=None):
    stds = torch.zeros(
        imgs.size(0), len(masks), dtype=imgs.dtype, device=imgs.device)
    for i, mask in enumerate(masks):
        imgs_slct = imgs.flatten(start_dim=1).index_select(
            dim=1, index=mask.flatten().nonzero().squeeze())
        stds[:, i] += torch.std(imgs_slct, dim=1)
    if beta is None:
        return torch.argmax(stds, dim=1)
    return nn.functional.softmax(beta * stds, dim=1)


# pylint: disable=E1102
def _compute_similarity(a, b):
    a = a / torch.linalg.vector_norm(a, dim=1, keepdim=True)
    b = b / torch.linalg.vector_norm(b, dim=1, keepdim=True)
    return torch.linalg.vecdot(a, b, dim=1)


def compute_max_sim(imgs, masks, rng=None):  # pylint: disable=W0613
    imgs_slct = []
    for mask in masks:
        imgs_slct.append(imgs.flatten(start_dim=1).index_select(
            dim=1, index=mask.flatten().nonzero().squeeze()))
    sims = torch.zeros(
        imgs.size(0), len(masks)-1, dtype=imgs.dtype, device=imgs.device)
    for i, img_slct in enumerate(imgs_slct[1:]):
        sims[:, i] += _compute_similarity(imgs_slct[0], img_slct)
    return torch.max(sims, dim=1)[0]


# pylint: disable=W0613
def compute_argmax_sim(imgs, masks, rng=None, beta=None):
    imgs_slct = []
    for mask in masks:
        imgs_slct.append(imgs.flatten(start_dim=1).index_select(
            dim=1, index=mask.flatten().nonzero().squeeze()))
    sims = torch.zeros(
        imgs.size(0), len(masks)-1, dtype=imgs.dtype, device=imgs.device)
    for i, imgs_slct_i in enumerate(imgs_slct[1:]):
        sims[:, i] += _compute_similarity(imgs_slct[0], imgs_slct_i)
    if beta is None:
        return torch.argmax(sims, dim=1)
    return nn.functional.softmax(beta * sims, dim=1)


def compute_fft_slope(imgs, mask, rng=None):  # pylint: disable=W0613
    slopes = torch.zeros(imgs.size(0), dtype=imgs.dtype, device=imgs.device)
    for i, img in enumerate(imgs):
        slopes[i] += _compute_fft_slope(img * mask)
    return slopes


def compute_max_std_rand(imgs, masks, rng, probs=None):
    if probs is not None:
        assert len(probs) == len(masks)
    imgs_slct = []
    for mask in masks:
        imgs_slct.append(imgs.flatten(start_dim=1).index_select(
            dim=1, index=mask.flatten().nonzero().squeeze()))
    stds = torch.zeros(
        imgs.size(0), len(masks), dtype=imgs.dtype, device=imgs.device)
    for i, img_slct in enumerate(imgs_slct):
        if (probs is None) or torch.bernoulli(
                torch.full((1,), 1.0-probs[i]), generator=rng):
            stds[:, i] += torch.std(img_slct, dim=1)
    return torch.max(stds, dim=1)[0]


def compute_max_sim_rand(imgs, masks, rng, probs=None):
    n_sims = len(masks)//2
    if probs is not None:
        assert len(probs) == n_sims
    imgs_slct = []
    for mask in masks:
        imgs_slct.append(imgs.flatten(start_dim=1).index_select(
            dim=1, index=mask.flatten().nonzero().squeeze()))
    sims = torch.zeros(
        imgs.size(0), n_sims, dtype=imgs.dtype, device=imgs.device)
    for i in range(n_sims):
        if (probs is None) or torch.bernoulli(
                torch.full((1,), 1.0-probs[i]), generator=rng):
            sims[:, i] += _compute_similarity(
                imgs_slct[i], imgs_slct[n_sims + i])
    return torch.max(sims, dim=1)[0]


def compute_argmax_sim_rand(imgs, masks, rng, permute=False, beta=None):
    n_patch = len(masks) // 2
    # Remove error masks from main masks
    for i in range(n_patch):
        masks[i] -= masks[i+n_patch]
    # Collect pixels
    imgs_slct = []
    for mask in masks:
        imgs_slct.append(imgs.flatten(start_dim=1).index_select(
            dim=1, index=mask.flatten().nonzero().squeeze()))
    # Apply permutation on error masks
    if permute:
        perm = torch.randperm(n_patch-1, generator=rng) + n_patch + 1
    else:
        perm = torch.arange(n_patch-1) + n_patch + 1
    # Compute similarities
    sims = torch.zeros(
        imgs.size(0), n_patch-1, dtype=imgs.dtype, device=imgs.device)
    ref = torch.concatenate((imgs_slct[0], imgs_slct[n_patch]), dim=1)
    for i in range(n_patch-1):
        sims[:, i] += _compute_similarity(ref, torch.concatenate(
            (imgs_slct[i+1], imgs_slct[perm[i]]), dim=1))
    if beta is None:
        return torch.argmax(sims, dim=1)
    return nn.functional.softmax(beta * sims, dim=1)


# ImgStat


class ImgStat:
    def __init__(self, imgstat_name):
        # Attributes
        self.name = imgstat_name
        self.mask_type, self.lum_type, self.stat_type = self._read_name()
        assert self.stat_type in IMGSTAT_INFO, 'Unknown image statistic type.'
        assert self.lum_type in LUMINANCE_TYPES, 'Unknown luminance type.'
        assert self.mask_type in MASK_INFO, 'Unknown mask type.'
        self.extract_func, self.mono_channel, self.binary_mask,\
            self.multi_mask, self.use_bbox, self.categorical = IMGSTAT_INFO[
                self.stat_type]
        # Data
        self.data = None
        self.n, self.n_c = None, 1  # Luminance images only
        self.n_cat = None  # Only used for categorical statistics
        # Standardization
        self.mean, self.std = None, None
        # Augmentation
        self.aug_scale = None
        self.aug_mean = None
        self.aug_std = None

    def _read_name(self):
        name_split = self.name.split('-')
        assert len(name_split) == 3, 'Unknown image statistic type.'
        return name_split

    def extract(self, imgs, mask_set, rng=None, kwargs=None):
        masks = mask_set.torch().to(imgs.device)
        if self.use_bbox and mask_set.bbox is not None:
            slc_h = slice(mask_set.bbox[0], mask_set.bbox[1])
            slc_w = slice(mask_set.bbox[2], mask_set.bbox[3])
            imgs = imgs[:, :, slc_h, slc_w]
            masks = masks[:, slc_h, slc_w]
        if self.mono_channel:
            assert imgs.size(1) == 1, 'Image has too many channels.'
            imgs = imgs.squeeze(dim=1)
        if not self.multi_mask:
            assert mask_set.n_c == 1, 'Too many masks.'
            masks = masks.squeeze(dim=0)
        if kwargs is None:
            kwargs = {}
        return self.extract_func(imgs, masks, rng, **kwargs)

    def zeros(self, n=N_IMAGES, dtype=np.float32):
        self.n = n
        self.data = np.zeros((self.n, self.n_c), dtype=dtype)
        return self

    def store(self, data):
        n, n_c = data.shape
        assert n_c == self.n_c, 'Wrong number of channels.'
        self.n = n
        self.data = data
        if self.categorical:
            self.n_cat = int(np.max(self.data)) + 1
        return self

    def save(self):
        data = self.data.astype(np.float32, copy=False)
        file_path = os.path.join(get_imgstat_dir(), f'{self.name}.npz')
        np.savez_compressed(file_path, data=data)
        return self

    def load(self):
        file_path = os.path.join(get_imgstat_dir(), f'{self.name}.npz')
        self.store(np.load(file_path)['data'])
        return self

    def compute_std_values(self):
        assert self.data is not None, 'Load data first.'
        if self.categorical:
            prob = np.bincount(
                self.data.flatten().astype(np.int64)).astype(np.float32)
            prob /= self.n
            file_path = os.path.join(get_std_dir(), f'{self.name}.npz')
            np.savez_compressed(file_path, data=prob)
            with np.printoptions(precision=8, suppress=True):
                print(f'{self.name} prob : {prob}')
            return self
        self.mean = np.nanmean(
            self.data, axis=0, keepdims=True,
            dtype=np.float64).astype(dtype=np.float32)
        self.std = np.nanstd(
            self.data, axis=0, keepdims=True,
            dtype=np.float64).astype(dtype=np.float32)
        file_path = os.path.join(get_std_dir(), f'{self.name}.npz')
        np.savez_compressed(file_path, mean=self.mean, std=self.std)
        with np.printoptions(precision=8, suppress=True):
            print(f'{self.name} mean : {self.mean}')
            print(f'{self.name} std  : {self.std}')
        return self

    def standardize(self):
        if self.categorical:
            return self
        assert self.data is not None, 'Load data first.'
        if self.mean is None:
            file_path = os.path.join(get_std_dir(), f'{self.name}.npz')
            file_data = np.load(file_path)
            self.mean = file_data['mean']
            self.std = file_data['std']
        self.data -= self.mean
        self.data /= self.std
        return self

    def get_img_set(self, img_size=None):
        if img_size is None:
            img_size = IMG_SIZE_PX
        return ImgSet(self.lum_type, img_size)

    def get_mask(self, img_size=None):
        if img_size is None:
            img_size = IMG_SIZE_PX
        return Mask(self.mask_type, img_size)

    def __deepcopy__(self, memo):
        cls = self.__class__
        imgstat = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k == 'data':
                imgstat.__dict__.update({k: None})
            else:
                imgstat.__dict__.update({k: copy.deepcopy(v, memo)})
        return imgstat

    def select(self, indices):
        imgstat = copy.deepcopy(self)
        imgstat.data = self.data[indices]
        imgstat.n = len(indices)
        return imgstat

    def compute_augmentation_values(self, aug_std=None):
        if self.categorical:
            return self
        # Assume standardized input and target.
        if aug_std is not None:
            assert 0.0 <= aug_std < 1.0, 'aug_std must be in range [0, 1[.'
            self.aug_scale = np.sqrt(1.0 - aug_std**2)
            self.aug_mean = torch.zeros(self.n_c, dtype=torch.float32)
            self.aug_std = aug_std * torch.ones(self.n_c, dtype=torch.float32)
        return self

    def torch(self, idx, rng=None):
        if self.categorical:
            x = torch.tensor(self.data[idx], dtype=torch.int32)
        else:
            x = torch.tensor(self.data[idx], dtype=torch.float32)
        if rng is not None and self.aug_scale is not None:
            x *= self.aug_scale
            x += torch.normal(self.aug_mean, self.aug_std, generator=rng)
        return x

    def torch_onehot(self, idx, rng=None):
        assert self.categorical, 'Statistic is not categorical.'
        id_ = self.torch(idx, rng).squeeze(dim=0)
        id_onehot = nn.functional.one_hot(  # pylint: disable=E1102
            id_.to(torch.int64), num_classes=self.n_cat)
        return id_onehot


class ImgStatSet:
    def __init__(self, imgstat_set_name):
        self.name = imgstat_set_name
        if self.name in IMGSTAT_SET_INFO:
            self.imst_names = IMGSTAT_SET_INFO[self.name]
        else:
            self.imst_names = (self.name,)
        # Data
        self.imsts = tuple(ImgStat(imst_name) for imst_name in self.imst_names)
        # Check that image luminance type is the same for all image statistics
        if len(set((imst.lum_type for imst in self.imsts))) > 1:
            raise ValueError(
                'Set with image statistics of different luminance types.')
        # Check if set is categorical
        self.categorical = False
        for imst in self.imsts:
            if imst.categorical and len(self.imsts) > 1:
                raise ValueError(
                    'Set with a categorical image statistic must be unique.')
            if imst.categorical:
                self.categorical = True
        # Attributes
        self.n_imst = len(self.imst_names)
        self.n, self.n_c = None, int(np.sum((imst.n_c for imst in self.imsts)))
        self.n_cat = None  # Only used for categorical statistics
        self.lum_type = self.imsts[0].lum_type

    def get(self, imst_name_or_idx=0):
        if isinstance(imst_name_or_idx, int):
            return self.imsts[imst_name_or_idx]
        for i, imst_name in enumerate(self.imst_names):
            if imst_name == imst_name_or_idx:
                return self.imsts[i]
        raise ValueError('Unknown image statistic.')

    def load(self):
        for imst in self.imsts:
            imst.load()
        if len(set((imst.n for imst in self.imsts))) > 1:
            raise ValueError('Set with image statistics of different sizes.')
        self.n = self.imsts[0].n
        if self.categorical:
            self.n_cat = self.imsts[0].n_cat
        return self

    def compute_std_values(self):
        for imst in self.imsts:
            imst.compute_std_values()
        return self

    def standardize(self):
        for imst in self.imsts:
            imst.standardize()
        return self

    def get_img_set(self, img_size=None):
        if img_size is None:
            img_size = IMG_SIZE_PX
        return ImgSet(self.lum_type, img_size)

    def get_masks(self, img_size=None):
        return tuple(imst.get_mask(img_size) for imst in self.imsts)

    def select(self, indices):
        cls = self.__class__
        imgstat_set = cls.__new__(cls)
        imgstat_set.name = self.name
        imgstat_set.imst_names = self.imst_names
        imgstat_set.imsts = tuple(imst.select(indices) for imst in self.imsts)
        imgstat_set.categorical = self.categorical
        imgstat_set.n_imst = self.n_imst
        imgstat_set.n = imgstat_set.imsts[0].n
        imgstat_set.n_c = self.n_c
        imgstat_set.n_cat = self.n_cat
        return imgstat_set

    def compute_augmentation_values(self, aug_std=None):
        for imst in self.imsts:
            imst.compute_augmentation_values(aug_std)
        return self

    def torch(self, idx, rng=None):
        return torch.concatenate(
            tuple(imst.torch(idx, rng) for imst in self.imsts))

    def torch_onehot(self, idx, rng=None):
        assert self.categorical, 'Statistic is not categorical.'
        return self.imsts[0].torch_onehot(idx, rng)


# Image statistics extractor


class _ImgStatsExtractorDataset(Dataset):
    def __init__(self, imgstat, img_set, mask, device, seed=None,
                 extract_kwargs=None):
        self.imgstat = imgstat
        self.img_set = img_set
        self.mask = mask
        self.device = device
        self.rng = None
        if seed is not None:
            self.rng = torch.Generator().manual_seed(seed)
        self.extract_kwargs = extract_kwargs

    def __len__(self):
        return self.img_set.n

    def __getitem__(self, idx):
        # Load image
        img = self.img_set.torch(idx).unsqueeze(dim=0).to(self.device)
        # Compute statistics
        return self.imgstat.extract(
            img, self.mask, self.rng, self.extract_kwargs).squeeze(dim=0)


class ImgStatsExtractor:
    def __init__(self, imgstat_set_name, img_size=None, seed=100):
        self.imst_set = ImgStatSet(imgstat_set_name)
        self.img_set = self.imst_set.get_img_set(img_size)
        self.masks = self.imst_set.get_masks(img_size)
        self.seed = seed

    @torch.no_grad()
    def _flow(self, imgstat, mask, batch_size, num_workers=0, device='cpu',
              extract_kwargs=None):
        imgstat_ = ImgStat(imgstat.name)  # Avoid data copy to the dataloader
        return DataLoader(
            _ImgStatsExtractorDataset(
                imgstat_, self.img_set, mask, device, self.seed,
                extract_kwargs),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            worker_init_fn=set_worker_seed)

    @torch.no_grad()
    def extract(self, batch_size=64, num_workers=0, device='cpu',
                extract_kwargs=None):
        if not isinstance(extract_kwargs, tuple):
            extract_kwargs = (extract_kwargs,)
        for imgstat, mask, kwargs in zip(
                self.imst_set.imsts, self.masks, extract_kwargs):
            # Init image statistics with zeros
            imgstat.zeros()
            dtld = self._flow(
                imgstat, mask, batch_size, num_workers, device, kwargs)
            # Compute image statistics
            for i, x in enumerate(
                    tqdm(dtld, total=len(dtld), desc=imgstat.name)):
                slc = slice(i*batch_size, (i+1)*batch_size)
                if x.dim() == 1:
                    x = x.unsqueeze(dim=1)
                imgstat.data[slc] += x.cpu().numpy()
            # Save data
            imgstat.save()
