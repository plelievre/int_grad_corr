"""
Custom modules.

Author: Pierre Lelievre
"""

import numpy as np
from functools import partial

import torch
from torch import nn
from torchvision.models.convnext import LayerNorm2d, CNBlock

from .weight_init_1v0 import init_linear, init_conv


# Utils


def _act_sel_init(act_type, layer):
    assert act_type in ('relu', 'mish', 'pmish')
    if act_type in ('relu', 'mish'):
        init_linear(layer, init_gain_act='relu')
    elif act_type == 'pmish':
        init_linear(
            layer, init_gain_act='leaky_relu', kaiming_args=(0.5, 'fan_in'))
    if act_type == 'mish':
        return nn.Mish()
    if act_type == 'pmish':
        return PMish(init=0.0)
    return nn.ReLU()


# Activation functions


class PMish(nn.Module):
    """
    Parameterized version of Mish activation function [arXiv:1908.08681].
    As weight -> -inf, PMish -> Mish
       weight -> +inf, PMish -> identity function
    """
    def __init__(self, init=0.0):
        super().__init__()
        weight = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(weight, init)
        self.register_parameter('weight', weight)

    def forward(self, x):
        alpha = nn.functional.sigmoid(self.weight)
        return x * nn.functional.tanh(
            alpha + nn.functional.softplus(x))  # pylint: disable=E1102


# Linear block


class LinearBlock(nn.Module):
    def __init__(self, lin_sizes, dropout=0.0, dropout_min_size=16,
                 act_type='relu', output_act=True):
        super().__init__()
        self.lin_sizes = lin_sizes
        self.last_lin = len(self.lin_sizes) - 2
        if output_act:
            self.last_lin += 1
        self.dropout_min_size = dropout_min_size
        # Layers
        for i, (size_in, size_out) in enumerate(
                zip(self.lin_sizes[:-1], self.lin_sizes[1:])):
            lin = nn.Linear(size_in, size_out, bias=i==self.last_lin)
            if i != self.last_lin:
                act = _act_sel_init(act_type, lin)
                setattr(self, f'bn_{i}', nn.BatchNorm1d(size_out))
                setattr(self, f'act_{i}', act)
            else:
                init_linear(lin, init_gain_act='linear')
            setattr(self, f'lin_{i}', lin)
        # Misc
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i, layer_size in enumerate(self.lin_sizes[1:]):
            x = getattr(self, f'lin_{i}')(x)
            if i != self.last_lin:
                x = getattr(self, f'bn_{i}')(x)
                x = getattr(self, f'act_{i}')(x)
                if layer_size >= self.dropout_min_size:
                    x = self.dropout(x)
        return x


# Convolution 2d block


class Conv2dBlock(nn.Module):
    def __init__(self, conv_sizes, kernel_size=3, stride=2, dropout=0.0,
                 dropout_min_size=16, act_type='relu', output_act=True):
        super().__init__()
        self.conv_sizes = conv_sizes
        self.last_conv = len(self.conv_sizes) - 2
        if output_act:
            self.last_conv += 1
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout_min_size = dropout_min_size
        # Layers
        for i, (size_in, size_out) in enumerate(
                zip(self.conv_sizes[:-1], self.conv_sizes[1:])):
            conv = nn.Conv2d(
                size_in, size_out, kernel_size=self.kernel_size,
                stride=self.stride, bias=i==self.last_conv)
            if i != self.last_conv:
                act = _act_sel_init(act_type, conv)
                setattr(self, f'bn_{i}', nn.BatchNorm2d(size_out))
                setattr(self, f'act_{i}', act)
            else:
                init_conv(conv, init_gain_act='linear')
            setattr(self, f'conv_{i}', conv)
        # Misc
        self.dropout_2d = nn.Dropout2d(dropout)

    def forward(self, x):
        for i, layer_size in enumerate(self.conv_sizes[1:]):
            x = getattr(self, f'conv_{i}')(x)
            x = getattr(self, f'bn_{i}')(x)
            x = getattr(self, f'act_{i}')(x)
            if layer_size >= self.dropout_min_size:
                x = self.dropout_2d(x)
        return x


# ConvNeXt modules


class ConvNeXtStem(nn.Module):
    '''
    Modified from: https://github.com/pytorch/vision/blob/main/torchvision/
                   models/convnext.py
    '''
    def __init__(self, in_channels, out_channels, kernel_size=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels, out_channels, bias=True, kernel_size=self.kernel_size,
            stride=self.kernel_size, padding=0)
        self.l_n = partial(LayerNorm2d, eps=1e-6)(out_channels)
        # Init
        init_conv(self.conv, init_gain_act=0.02, init_bias='uniform')

    def get_output_spatial_size(self, spatial_size_in):
        k = float(self.kernel_size)
        return int(np.floor((spatial_size_in - k) / k + 1.0))

    def forward(self, x):
        x = self.conv(x)
        x = self.l_n(x)
        return x


class ConvNeXtBlock(nn.Module):
    '''
    Modified from: https://github.com/pytorch/vision/blob/main/torchvision/
                   models/convnext.py
    '''
    def __init__(self, conv_sizes, num_layers_per_block=3,
                 stochastic_depth_prob=0.0, layer_scale=1e-6):
        super().__init__()
        self.n_block = len(conv_sizes) - 1
        assert self.n_block, 'conv_sizes requires at least 2 values.'
        layers = []
        norm_layer = partial(LayerNorm2d, eps=1e-6)
        total_stage_blocks = num_layers_per_block * self.n_block
        # ConvNeXt blocks
        stage_block_id = 0
        for size_in, size_out in zip(conv_sizes[:-1], conv_sizes[1:]):
            # Bottlenecks
            stage = []
            for _ in range(num_layers_per_block):
                # Adjust stochastic depth probability based on the depth of the
                # stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (
                    total_stage_blocks - 1.0)
                stage.append(CNBlock(size_in, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # Downsampling
            layers.append(nn.Sequential(
                norm_layer(size_in),
                nn.Conv2d(size_in, size_out, kernel_size=2, stride=2)))
        self.features = nn.Sequential(*layers)
        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_linear(m, init_gain_act=0.02)
            elif isinstance(m, nn.Conv2d):
                init_conv(m, init_gain_act=0.02)

    def get_output_spatial_size(self, spatial_size_in):
        spatial_size_out = spatial_size_in
        for _ in range(self.n_block):
            spatial_size_out = np.floor((spatial_size_out - 2.0) / 2.0 + 1.0)
        return int(spatial_size_out)

    def forward(self, x):
        return self.features(x)
