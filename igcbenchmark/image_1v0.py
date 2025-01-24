"""
Image utilities.

Author: Pierre Lelievre
"""

import os
import copy
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
N_IMAGES = 73000
IMG_SIZE_PX = 425
SRGB_TO_CIE1931_Y = (0.2126729, 0.7151522, 0.0721750)
MATLAB_RGB2GRAY = (0.2989, 0.5870, 0.1140)  # nearly REC601 to CIE1931_Y
NEUTRAL_GRAY_EV = (-2.51401829, -2.44076063, -2.31796921)  # for 50 Lab
#                 (-2.30958135, -2.23629420, -2.11351562) for 127 sRGB
IMGNET_MEAN = (0.485, 0.456, 0.406)
IMGNET_STD = (0.229, 0.224, 0.225)
COLOR_TYPES = ('none', 'lin_rgb', 'log2_rgb', 'imgnet')
LUMINANCE_TYPES = ('lin_Y', 'log2_Y', 'nsd_lum')


# Directory utils


def _create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


def get_image_dir(data_dir=DATA_DIR):
    return _create_dir(os.path.join(data_dir, 'images'))


def get_std_dir(data_dir=DATA_DIR):
    return _create_dir(os.path.join(data_dir, 'std'))


def get_imgstat_dir(data_dir=DATA_DIR):
    return _create_dir(os.path.join(data_dir, 'imgstats'))


# Color space utils


def srgb_to_linear(img):
    mask = img <= 0.04045
    img[mask] /= 12.92
    mask = np.invert(mask)
    img[mask] += 0.055
    img[mask] /= 1.055
    img[mask] **= 2.4
    return img


def color_to_grayscale(img, coefs=SRGB_TO_CIE1931_Y, dtype=np.float32):
    return np.sum(img * np.array(coefs, dtype=dtype)[None, None, :], axis=2)


def luminance_to_ev(img, min_luminance=1e-3):
    return np.log2(np.clip(img, min_luminance, 1.0)) - NEUTRAL_GRAY_EV[1]


def color_to_ev(img, min_luminance=1e-3):
    return np.log2(np.clip(img, min_luminance, 1.0)) - NEUTRAL_GRAY_EV


# Image Set


class ImgSet:
    def __init__(self, img_type='lin_rgb', img_size=None):
        # Attributes
        assert img_type in (COLOR_TYPES + LUMINANCE_TYPES),\
            'Unknown image type.'
        self.img_type = img_type
        self.img_size = img_size
        if self.img_size is None:
            self.img_size = IMG_SIZE_PX
        # Data
        image_dir = get_image_dir()
        self.paths = tuple(os.path.join(
            image_dir, f'nsd_{i:05d}.png') for i in range(N_IMAGES))
        self.paths = tuple(path for path in self.paths if os.path.isfile(path))
        self.n, self.n_c = len(self.paths), 3
        if self.img_type in LUMINANCE_TYPES:
            self.n_c = 1
        # Standardization
        self.mean, self.std = None, None
        self.apply_standardization = False
        # Augmentation
        self.apply_augmentation = False

    def compute_std_values(self):
        self.mean = np.zeros(self.n_c, dtype=np.float64)
        self.std = np.zeros(self.n_c, dtype=np.float64)
        # Load all images
        img_set = ImgSet(img_type=self.img_type)
        for idx in tqdm(range(self.n), total=self.n, desc=self.img_type):
            img = img_set.get_image(idx)
            self.mean += np.mean(img, axis=(0, 1))
            self.std += np.mean(img**2, axis=(0, 1))
        # Compute std values
        self.mean /= self.n
        self.std /= self.n
        self.std -= self.mean**2
        self.std = np.sqrt(self.std)
        self.mean = self.mean.astype(dtype=np.float32)
        self.std = self.std.astype(dtype=np.float32)
        # Save std values
        file_path = os.path.join(get_std_dir(), f'{self.img_type}.npz')
        np.savez_compressed(file_path, mean=self.mean, std=self.std)
        with np.printoptions(precision=8, suppress=True):
            print(f'{self.img_type} mean : {self.mean}')
            print(f'{self.img_type} std  : {self.std}')
        return self

    def standardize(self):
        std_file_path = os.path.join(get_std_dir(), f'{self.img_type}.npz')
        if os.path.isfile(std_file_path):
            file_data = np.load(std_file_path)
            self.mean = file_data['mean']
            self.std = file_data['std']
        elif self.img_type == 'imgnet':
            self.mean = np.array(IMGNET_MEAN, dtype=np.float32)
            self.std = np.array(IMGNET_STD, dtype=np.float32)
        else:
            print('No standardization values.')
        self.apply_standardization = True
        return self

    def get_random_split(self, val_ratio=0.1, seed=100):
        # Define random generator
        rng = np.random.default_rng(seed)
        # Compute validation indices
        val_size = int(np.ceil(val_ratio * self.n))
        indices = np.arange(self.n)
        val_indices = rng.choice(
            indices, size=val_size, replace=False, shuffle=False).tolist()
        val_indices = np.array(val_indices, dtype=np.int64)
        # Compute training indices
        train_mask = np.ones(self.n, dtype=np.bool_)
        train_mask[val_indices] &= False
        train_indices = np.nonzero(train_mask)[0]
        return val_indices, train_indices

    def __deepcopy__(self, memo):
        cls = self.__class__
        imgstat = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k == 'paths':
                imgstat.__dict__.update({k: None})
            else:
                imgstat.__dict__.update({k: copy.deepcopy(v, memo)})
        return imgstat

    def select(self, indices):
        img_set = copy.deepcopy(self)
        img_set.paths = tuple(self.paths[i] for i in indices)
        img_set.n = len(indices)
        return img_set

    def augment(self):
        self.apply_augmentation = True
        return self

    def get_image(self, idx, augment=False):
        path = self.paths[idx]
        # Compute image size
        img_size = self.img_size
        if augment:
            img_size *= 2
        # Load image
        while True:
            try:
                with Image.open(path) as pil_img:
                    pil_img = pil_img.convert('RGB')
                    # Resize image
                    if img_size != IMG_SIZE_PX:
                        pil_img = pil_img.resize(
                            (img_size, img_size),
                            resample=Image.Resampling.LANCZOS)
                    # Remap image in range [0, 1]
                    img = np.array(pil_img, dtype=np.float32)
                    img /= 255.0
                break
            except OSError:
                print(f'Failed loading : {path}')
        # Color transformations
        if self.img_type in ('lin_rgb', 'log2_rgb'):
            img = srgb_to_linear(img)
            if self.img_type == 'log2_rgb':
                img = color_to_ev(img)
        elif self.img_type in ('lin_Y', 'log2_Y'):
            img = color_to_grayscale(srgb_to_linear(img))
            if self.img_type == 'log2_Y':
                img = luminance_to_ev(img)
        elif self.img_type == 'nsd_lum':
            img = np.power(color_to_grayscale(img, MATLAB_RGB2GRAY), 2.0)
        # Reshape for augmentation
        if augment:
            img = np.reshape(
                img, (self.img_size, 2, self.img_size, 2, self.n_c))
            img = np.transpose(img, (0, 2, 1, 3, 4))
            img = np.reshape(
                img, (self.img_size, self.img_size, 4, self.n_c))
        return img

    @torch.no_grad()
    def torch(self, idx, rng=None):
        augment = self.apply_standardization and (rng is not None)
        # Get image
        x = torch.tensor(self.get_image(idx, augment), dtype=torch.float32)
        # Standardize
        if self.apply_standardization and (self.mean is not None):
            x -= self.mean
            x /= self.std
        # Augment
        if augment:
            aug_idx = torch.randint(
                4, size=(self.img_size, self.img_size, 1, 1), generator=rng)
            aug_idx = aug_idx.expand(-1, -1, -1, self.n_c)
            x = torch.gather(x, dim=2, index=aug_idx).squeeze(dim=2)
        # Permute channels
        if augment or self.img_type in COLOR_TYPES:
            x = x.permute(2, 0, 1)
        else:
            x = x.unsqueeze(dim=0)
        return x
