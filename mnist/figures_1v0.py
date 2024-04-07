"""
Figure utils.

Author: Pierre Lelievre
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# Adjust matplotlib defaults


mpl.rcParams['font.size'] = 10
mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['legend.fontsize'] = 'small'


# Utils


def filter_contour(contour, threshold=1.0):
    for cc in contour.collections:
        for pp in cc.get_paths():
            mask = np.ones(len(pp.vertices), dtype=np.bool_)
            i = 0
            for poly in pp.to_polygons():
                n = len(poly)
                diameter = np.sum(np.linalg.norm(poly, axis=1))
                if diameter < threshold:
                    mask[i:i+n] &= False
                i += n
            pp.vertices = pp.vertices[mask]
            pp.codes = pp.codes[mask]


# Figures


def mnist_img(x, contour_m=None, contour_p=None):
    fig, axs = plt.subplots(3, 5, figsize=(6.0, 2.73), dpi=300, gridspec_kw = {
        'top': 1.0, 'bottom': 0.0, 'left': 0.0, 'right': 1.0,
        'wspace': 0.05, 'hspace': 0.05,
        'height_ratios': [0.44, 0.44, 0.12]})
    v_max = np.max(np.abs(x))
    v_max_tick_exp = np.power(10.0, np.floor(np.log10(v_max)))
    v_max_tick_val = np.floor(v_max / v_max_tick_exp)
    v_max_tick = v_max_tick_val * v_max_tick_exp
    n_ticks = 3
    ticks = np.linspace(-v_max_tick, v_max_tick, n_ticks)
    tick_labels = None
    if v_max_tick <= 0.0001:
        tick_labels = [f'{i:.0e}' for i in ticks]
    blue_color = mpl.colormaps['RdBu_r'](0.0)
    if (contour_m is not None) or (contour_p is not None):
        xx, yy = np.meshgrid(np.arange(28), np.arange(28))
    for i in range(10):
        ax = axs[i//5][i%5]
        ax.axis('off')
        ax.imshow(x[i], cmap='RdBu_r', vmin=-v_max, vmax=v_max)
        if contour_m is None:
            ax.annotate(
                str(i), (0.8, 0.1), xycoords='axes fraction', fontsize='small',
                bbox={
                    'edgecolor': 'none', 'facecolor': 'w', 'alpha': 0.7,
                    'pad': 0.25, 'boxstyle': 'round'})
        else:
            levels_m = np.quantile(contour_m[i], (0.8,))
            ax.contour(
                xx, yy, contour_m[i], levels=levels_m, colors='k',
                linestyles=(':',), antialiased=True,
                negative_linestyles='solid')
        if contour_p is not None:
            contour_obj = ax.contour(
                xx, yy, contour_p[i], levels=(0.5,), colors=(blue_color,),
                linewidths=(1.0,), alpha=0.5, antialiased=True,
                negative_linestyles='solid')
            filter_contour(contour_obj, 200.0)
    for i in range(5):
        axs[2, i].axis('off')
    c_bar_scale = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=-v_max, vmax=v_max), cmap='RdBu_r')
    c_bar = fig.colorbar(
        c_bar_scale, ax=axs[2, :2], anchor=(0.5, 0.8),
        location='bottom', orientation='horizontal', fraction=1.0,
        aspect=40.0, shrink=0.8)
    c_bar.set_ticks(ticks=ticks, labels=tick_labels)
    handles = []
    if contour_m is not None:
        handles.append(mpl.lines.Line2D(
            [], [], color='k', linestyle=':', label='mean digit'))
    if contour_p is not None:
        handles.append(mpl.lines.Line2D(
            [], [], color=blue_color, alpha=0.5, linewidth=1.0,
            label='high difference probability'))
    fig.legend(handles=handles, loc='lower right', ncol=2)
    return fig


def mnist_img_10(x, contour_m=None, digits=None):
    fig, axs = plt.subplots(
        10, 10, figsize=(2.7, 3.0), dpi=300, gridspec_kw = {
            'top': 0.9, 'bottom': 0.1, 'left': 0.1/0.9, 'right': 1.0,
            'wspace': 0.0, 'hspace': 0.0})
    if contour_m is not None:
        xx, yy = np.meshgrid(np.arange(28), np.arange(28))
    if digits is None:
        digits = np.arange(10)
    elif isinstance(digits, int):
        digits = (digits,) * 10
    for j in range(10):  # sample idx
        v_max = np.max(np.abs(x[j]))
        for i in range(10):  # y idx
            ax = axs[j][i]
            ax.axis('off')
            ax.imshow(x[j, i], cmap='RdBu_r', vmin=-v_max, vmax=v_max)
            if i==digits[j] and (contour_m is not None):
                levels_m = np.quantile(contour_m[j], (0.8,))
                ax.contour(
                    xx, yy, contour_m[j], levels=levels_m, colors='k',
                    linestyles=(':',), antialiased=True,
                    negative_linestyles='solid')
    fig.text(
        0.05, 0.5, 'digit samples', rotation='vertical', va='center',
        ha='center')
    fig.text(0.5, 0.95, 'w.r.t. 0-9 predicted class', va='center', ha='center')
    handles = []
    if contour_m is not None:
        handles.append(mpl.lines.Line2D(
            [], [], color='k', linestyle=':', label='sample outline'))
    fig.legend(handles=handles, loc='lower right')
    return fig


def mnist_plot(inside, outside, low, high):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6, 2), dpi=300, gridspec_kw = {
            'top': 0.96, 'bottom': 0.09, 'left':0.10, 'right': 0.99,
            'wspace': 0.2})
    colors = mpl.colormaps['RdBu_r']
    n_rois = 10
    n_imst = 2
    n_imst_f = float(n_imst)
    ax1_ylim = (0.0, 1.0)
    ax1_nticks = 5
    ax1_label = 'integrated gradient\ncorrelation'
    ax2_ylim = (0.0, 1.0)
    ax2_nticks = 5
    legend_loc = 'upper right'
    bar_width = 0.85/(n_imst_f+1.0)
    # Inside/outside
    ax1.grid()
    x = np.arange(n_rois)
    y = np.linspace(*ax1_ylim, ax1_nticks)
    ax1.set_xlim(-0.5, n_rois-0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x, va='top')
    ax1.set_ylim(*ax1_ylim)
    ax1.set_yticks(y)
    ax1.set_yticklabels(np.round(y, 2)+0.0)
    ax1.set_ylabel(ax1_label, fontsize='small')
    for i in range(n_imst):
        color = colors(i/(n_imst_f-1))
        for j in range(n_rois):
            label = None
            if not j:
                label = ('inside', 'outside')[i]
            loc = j + (i+1.0)/(n_imst_f+1.0) - 0.5
            ax1.bar(
                loc, (inside, outside)[i][j], width=bar_width,
                color=color, label=label)
    ax1.legend(loc=legend_loc, ncol=n_imst)
    # Low/high difference probability
    ax2.grid()
    x = np.arange(n_rois)
    y = np.linspace(*ax2_ylim, ax2_nticks)
    ax2.set_xlim(-0.5, n_rois-0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x, va='top')
    ax2.set_ylim(*ax2_ylim)
    ax2.set_yticks(y)
    ax2.set_yticklabels(np.round(y, 2)+0.0)
    # ax2.set_ylabel(ax2_label)
    for i in range(n_imst):
        color = colors(i/(n_imst_f-1))
        for j in range(n_rois):
            label = None
            if not j:
                label = (
                    'low',
                    'high difference probability')[i]
            loc = j + (i+1.0)/(n_imst_f+1.0) - 0.5
            ax2.bar(
                loc, (low, high)[i][j], width=bar_width,
                color=color, label=label)
    ax2.legend(loc=legend_loc, ncol=n_imst)
    return fig
