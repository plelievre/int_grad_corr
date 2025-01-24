"""
Mask utilities.

Author: Pierre Lelievre
"""

import torch
import numpy as np

from .image_1v0 import IMG_SIZE_PX


MASK_INFO = {  # name: (method, kwargs)
    # comb_01
    'gauss_00': ('draw_gaussian', {
        'mu_x': 0, 'mu_y': 0, 'sigma_x': 0.10, 'alpha': 2.0}),
    'gauss_01': ('draw_gaussian', {
        'mu_x': 0, 'mu_y': 0, 'sigma_x': 0.20, 'alpha': -1.0}),
    'comb_01': ('combine', {'mask_names': (
        'gauss_00', 'gauss_01')}),
    # ccat_01
    'ellip_10': ('draw_ellipsis', {  # 0, 8.32, 8
        'mu_x': 0.0, 'mu_y': 0.1300079387, 'sigma_x': 0.125}),
    'ellip_11': ('draw_ellipsis', {  # -10, -9, 8
        'mu_x': -0.15625, 'mu_y': -0.140625, 'sigma_x': 0.125}),
    'ellip_12': ('draw_ellipsis', {  # 10, -9, 8
        'mu_x': 0.15625, 'mu_y': -0.140625, 'sigma_x': 0.125}),
    'ccat_01': ('concat', {'mask_names': (
        'ellip_10', 'ellip_11', 'ellip_12')}),
    # ccat_02 and ccat_03
    'ellip_20': ('draw_ellipsis', {
        'mu_x': 0.0, 'mu_y': 0.0, 'sigma_x': 0.07}),
    'ellip_21': ('draw_ellipsis', {  # -18, 18, env 4.5
        'mu_x': -0.28125, 'mu_y': 0.28125, 'sigma_x': 0.07}),
    'ellip_22': ('draw_ellipsis', {
        'mu_x': 0.28125, 'mu_y': 0.28125, 'sigma_x': 0.07}),
    'ellip_23': ('draw_ellipsis', {
        'mu_x': -0.28125, 'mu_y': -0.28125, 'sigma_x': 0.07}),
    'ellip_24': ('draw_ellipsis', {
        'mu_x': 0.28125, 'mu_y': -0.28125, 'sigma_x': 0.07}),
    'rect_20': ('draw_rectangle', {
        'c_x': 0.0, 'c_y': 0.0, 'size_x': 0.125}),
    'rect_21': ('draw_rectangle', {  # -18, 18, 8
        'c_x': -0.28125, 'c_y': 0.28125, 'size_x': 0.125}),
    'rect_22': ('draw_rectangle', {
        'c_x': 0.28125, 'c_y': 0.28125, 'size_x': 0.125}),
    'rect_23': ('draw_rectangle', {
        'c_x': -0.28125, 'c_y': -0.28125, 'size_x': 0.125}),
    'rect_24': ('draw_rectangle', {
        'c_x': 0.28125, 'c_y': -0.28125, 'size_x': 0.125}),
    'ccat_02': ('concat', {'mask_names': (
        'ellip_20', 'ellip_21', 'ellip_22', 'ellip_23', 'ellip_24')}),
    'ccat_03': ('concat', {'mask_names': (
        'ellip_20', 'ellip_21', 'ellip_22', 'ellip_23', 'ellip_24',
        'rect_20', 'rect_21', 'rect_22', 'rect_23', 'rect_24')}),
    # ccat_04
    'ellip_30': ('draw_ellipsis', {  # -16, 12, 5
        'mu_x': -0.25, 'mu_y': 0.21875, 'sigma_x': 0.078125}),
    'ellip_31': ('draw_ellipsis', {
        'mu_x': 0.25, 'mu_y': 0.21875, 'sigma_x': 0.078125}),
    'ellip_32': ('draw_ellipsis', {
        'mu_x': -0.25, 'mu_y': -0.21875, 'sigma_x': 0.078125}),
    'ellip_33': ('draw_ellipsis', {
        'mu_x': 0.25, 'mu_y': -0.21875, 'sigma_x': 0.078125}),
    'ccat_04': ('concat', {'mask_names': (
        'ellip_30', 'ellip_31', 'ellip_32', 'ellip_33')}),
}


# Utils


def _get_xy_map(img_size):
    y, x = np.indices((img_size, img_size), dtype=np.float32)
    x -= 0.5 * (img_size - 1)
    y -= 0.5 * (img_size - 1)
    x /= img_size
    y /= -1.0 * img_size
    return x, y, 1.0 / img_size**2


def _draw_gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y=None, rho=0.0,
                   density=True):
    if sigma_y is None:
        sigma_y = sigma_x
    out = ((x - mu_x) / sigma_x)**2
    out += ((y - mu_y) / sigma_y)**2
    out -= 2*rho * ((x - mu_x) / sigma_x) * ((y - mu_y) / sigma_y)
    out /= -2.0 * (1.0 - rho**2)
    out = np.exp(out)
    if density:
        out /= 2.0 * np.pi * sigma_x * sigma_y * np.sqrt(1.0 - rho**2)
    return out


def _draw_rectangle(x, y, c_x, c_y, size_x, size_y=None):
    if size_y is None:
        size_y = size_x
    out = y >= (c_y - 0.5 * size_y)
    out &= y <= (c_y + 0.5 * size_y)
    out &= x >= (c_x - 0.5 * size_x)
    out &= x <= (c_x + 0.5 * size_x)
    return out


# Mask


class Mask:
    def __init__(self, mask_name, img_size=None):
        # Attributes
        assert mask_name in MASK_INFO, 'Unknown mask type.'
        self.mask_name = mask_name
        self.img_size = img_size
        if self.img_size is None:
            self.img_size = IMG_SIZE_PX
        self.mask, self.bbox, self.binary, self.n_c = None, None, None, None
        # Compute mask and bbox (y_min, y_max, x_min, x_max)
        draw_func, draw_kwargs = MASK_INFO[self.mask_name]
        getattr(self, draw_func)(**draw_kwargs)

    def draw_gaussian(self, mu_x, mu_y, sigma_x, sigma_y=None, rho=0.0,
                      alpha=1.0):
        x, y, _ = _get_xy_map(self.img_size)
        mask = _draw_gaussian(
            x, y, mu_x, mu_y, sigma_x, sigma_y, rho, density=False)
        self.mask = np.expand_dims(mask, axis=2).astype(np.float32)
        self.mask *= alpha
        self.bbox = None
        self.binary = False
        self.n_c = 1
        return self

    def draw_ellipsis(self, mu_x, mu_y, sigma_x, sigma_y=None, rho=0.0,
                      alpha=1.0):
        x, y, _ = _get_xy_map(self.img_size)
        mask = _draw_gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y, rho) > 1.0
        self.mask = np.expand_dims(mask, axis=2).astype(np.float32)
        self.mask *= alpha
        indices = np.nonzero(mask)
        self.bbox = (
            np.min(indices[0]), np.max(indices[0]) + 1,
            np.min(indices[1]), np.max(indices[1]) + 1)
        self.binary = True
        if alpha != 1.0:
            self.binary = False
        self.n_c = 1
        return self

    def draw_rectangle(self, c_x, c_y, size_x, size_y=None, alpha=1.0):
        x, y, _ = _get_xy_map(self.img_size)
        mask = _draw_rectangle(x, y, c_x, c_y, size_x, size_y)
        self.mask = np.expand_dims(mask, axis=2).astype(np.float32)
        self.mask *= alpha
        indices = np.nonzero(mask)
        self.bbox = (
            np.min(indices[0]), np.max(indices[0]) + 1,
            np.min(indices[1]), np.max(indices[1]) + 1)
        self.binary = True
        if alpha != 1.0:
            self.binary = False
        self.n_c = 1
        return self

    def combine(self, mask_names):
        mask = np.zeros((self.img_size, self.img_size, 1), dtype=np.float32)
        bbox = False
        binary = True
        for mask_name in mask_names:
            draw_func, draw_kwargs = MASK_INFO[mask_name]
            getattr(self, draw_func)(**draw_kwargs)
            self.mask = np.sum(self.mask, axis=2, keepdims=True)
            mask += self.mask
            bbox |= self.bbox is not None
            binary &= self.binary
        self.mask = mask.astype(np.float32)
        self.bbox = None
        if bbox:
            indices = np.nonzero(mask)
            self.bbox = (
                np.min(indices[0]), np.max(indices[0]) + 1,
                np.min(indices[1]), np.max(indices[1]) + 1)
        self.binary = binary
        self.n_c = 1
        return self

    def concat(self, mask_names):
        masks = []
        bbox = False
        binary = True
        n_c = 0
        for mask_name in mask_names:
            draw_func, draw_kwargs = MASK_INFO[mask_name]
            getattr(self, draw_func)(**draw_kwargs)
            masks.append(self.mask)
            bbox |= self.bbox is not None
            binary &= self.binary
            n_c += self.n_c
        self.mask = np.concatenate(masks, axis=2).astype(np.float32)
        self.bbox = None
        if bbox:
            indices = np.nonzero(np.sum(self.mask, axis=2))
            self.bbox = (
                np.min(indices[0]), np.max(indices[0]) + 1,
                np.min(indices[1]), np.max(indices[1]) + 1)
        self.binary = binary
        self.n_c = n_c
        return self

    @torch.no_grad()
    def torch(self):
        x = torch.tensor(self.mask, dtype=torch.float32)
        # Permute channels
        x = x.permute(2, 0, 1)
        return x
