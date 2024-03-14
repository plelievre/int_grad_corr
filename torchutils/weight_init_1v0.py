"""
Utils to initialize module weights.

Author: Pierre Lelievre
"""

from torch import nn


def init_linear(layer, init_gain_act='linear', kaiming_args=None,
                init_bias=0.0):
    """
    Initialization for linear layers. Xavier or Kaiming method (with normal
    distributions) is selected dependeing on the type of activation function
    following this layer.
    """
    if init_gain_act in ('relu', 'leaky_relu'):
        if kaiming_args is None:
            kaiming_args = (0.0, 'fan_in')
        nn.init.kaiming_normal_(
            layer.weight, a=kaiming_args[0], mode=kaiming_args[1],
            nonlinearity=init_gain_act)
    else:
        nn.init.xavier_normal_(
            layer.weight, gain=nn.init.calculate_gain(init_gain_act))
    if layer.bias is not None:
        nn.init.constant_(layer.bias, init_bias)


def init_conv(layer, init_gain_act='linear', kaiming_args=None, init_bias=0.0):
    """
    Initialization for convolution layers. Xavier or Kaiming method (with
    normal distributions) is selected dependeing on the type of activation
    function following this layer.
    """
    init_linear(layer, init_gain_act, kaiming_args, init_bias)


def init_lstm(lstm, forget_bias=1.0):
    """
    Initialization for LSTM (Long Short-Term Memory) layer, as implemented for
    sketch_rnn from the magenta project [arxiv:1704.03477].
    """
    forget_bias /= 2.0
    size_h = lstm.weight_hh_l0.size(1)
    for i in range(lstm.num_layers):
        nn.init.xavier_uniform_(getattr(lstm, 'weight_ih_l%d' % i), gain=1.0)
        weight_hh = getattr(lstm, 'weight_hh_l%d' % i)
        nn.init.orthogonal_(weight_hh[:size_h], gain=1.0)
        nn.init.orthogonal_(weight_hh[size_h:size_h * 2], gain=1.0)
        nn.init.orthogonal_(weight_hh[size_h * 2:size_h * 3], gain=1.0)
        nn.init.orthogonal_(weight_hh[size_h * 3:], gain=1.0)
        bias_ih = getattr(lstm, 'bias_ih_l%d' % i)
        nn.init.constant_(bias_ih, 0)
        nn.init.constant_(bias_ih[size_h:size_h * 2], forget_bias)
        bias_hh = getattr(lstm, 'bias_hh_l%d' % i)
        nn.init.constant_(bias_hh, 0)
        nn.init.constant_(bias_hh[size_h:size_h * 2], forget_bias)
