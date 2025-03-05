from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

__version__ = "1.0.0"


def hex2rbg(h):
    """ Convert a hex color code to RGB. """
    return tuple(int(h.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4))


def cm_from_hex_list(name, hex_list):
    """ Create colormap for matplotlib from list of hex colors. """
    rgb_list = []
    for i in hex_list:
        rgb_list.append(hex2rbg(i))
    cm = LinearSegmentedColormap.from_list(name, rgb_list)
    return cm


def cm_test_plot(cmap):
    """ Make a plot to test a colormap. """
    delta = 0.025

    x = np.arange(-3.0, 3.01, delta)
    y = np.arange(-1.5, 2.5, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-X ** 2 - Y ** 2)
    Z2 = np.exp(-(X - 1) ** 2 - (Y - 1) ** 2)
    Z = (Z1 - Z2) * 2

    fig, ax = plt.subplots(figsize=(4, 4))
    cs = ax.imshow(Z, cmap=cmap)
    fig.colorbar(cs, orientation='horizontal')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(cmap.name)


# colors
dark = [
    '#6a7486',
    '#b81a4d',
    '#0a2463',
    '#3d93cc',
    '#81d3e3',
    '#ed9a5f'
]
light = [
    '#bbc7d8',
    '#eea5bc',
    '#a9b8db',
    '#add0e9',
    '#b8e7ee',
    '#f6c8a8'
]
extra = [
    '#42c3c1',
    '#289694',
    '#e56e5e'
]

# colormaps
_white = '#ffffff'

_multi_cm_list = [
    '#0a2463',
    '#432970',
    '#6c2c79',
    '#932f7b',
    '#b73479',
    '#d74072',
    '#e65d65',
    '#ec7c5e',
    '#ed9a5f',
    '#f1b67f',
    '#f5d0a3'
]

uni = cm_from_hex_list('uni', _multi_cm_list)
div = cm_from_hex_list('div', [dark[1], _white, dark[2]])
div2 = cm_from_hex_list('div2', [dark[2], dark[4], _white, light[5], dark[1]])

cmaps = [uni, div, div2]

# set default color cycler to dark tones (excluding gray)
default_cycler = (cycler(color=dark[1:]))
plt.rc('axes', prop_cycle=default_cycler)


# demo functions
def demo_colors():
    fig, axs = plt.subplots(3, figsize=(12, 2.2), sharex=True, constrained_layout=True)
    for ax, name, c_list in zip(axs, ['dark', 'light', 'extra'], [dark, light, extra]):
        for i, c in enumerate(c_list):
            ax.fill_betweenx([0, 1], i, i + 1, color=c)
        ax.axis('off')
        ax.set_title(name)


def demo_colorbars():
    for cmap in cmaps:
        cm_test_plot(cmap)
# demo_colorbars()
# plt.show()