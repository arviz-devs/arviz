import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gaussian, convolve  # pylint: disable=no-name-in-module
from scipy.stats import entropy

from .plot_utils import _scale_text


def kdeplot(values, cumulative=False, rug=False, label=None, bw=4.5, rotated=False,
            figsize=None, textsize=None, plot_kwargs=None, fill_kwargs=None,
            rug_kwargs=None, ax=None):
    """
    1D KDE plot taking into account boundary conditions

    Parameters
    ----------
    values : array-like
        Values to plot
    cumulative : bool
        If true plot the estimated cumulative distribution function. Defaults to False
    rug : bool
        If True adds a rugplot. Defaults to False
    label : string
        Text to include as part of the legend
    bw : float
        Bandwidth scaling factor. Should be larger than 0. The higher this number the smoother the
        KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule of thumb
        (the default rule used by SciPy).
    rotated : bool
        Whether to rotate the plot 90 degrees
    figsize : tuple
        Size of figure in inches. Defaults to (12, 8)
    textsize : float
        Size of text on figure.
    plot_kwargs : dict
        Keywords passed to the pdf line
    fill_kwargs : dict
        Keywords passed to the fill under the line (use fill_kwargs={'alpha': 0} to disable fill)
    rug_kwargs : dict
        Keywords passed to the rug plot. Ignored if rug=False
    ax : matplotlib axes
    Returns
    ----------
    ax : matplotlib axes

    """
    if plot_kwargs is None:
        plot_kwargs = {}
    plot_kwargs.setdefault('color', 'C0')

    if fill_kwargs is None:
        fill_kwargs = {}

    fill_kwargs.setdefault('alpha', 0.2)
    fill_kwargs.setdefault('color', 'C0')

    if rug_kwargs is None:
        rug_kwargs = {}
    rug_kwargs.setdefault('marker', '_' if rotated else '|')
    rug_kwargs.setdefault('linestyle', 'None')
    rug_kwargs.setdefault('color', 'C0')

    if figsize is None:
        if ax:
            figsize = ax.get_figure().get_size_inches()
        else:
            figsize = (12, 8)
    textsize, linewidth, markersize = _scale_text(figsize, textsize, 1)
    plot_kwargs.setdefault('linewidth', linewidth)
    rug_kwargs.setdefault('markersize', 2 * markersize)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    density, lower, upper = fast_kde(values, cumulative, bw)
    x = np.linspace(lower, upper, len(density))
    fill_func = ax.fill_between
    fill_x, fill_y = x, density
    if rotated:
        x, density = density, x
        fill_func = ax.fill_betweenx

    ax.plot(x, density, label=label, **plot_kwargs)
    if rotated:
        ax.set_xlim(0, auto=True)
        rug_x, rug_y = np.zeros_like(values), values
    else:
        ax.set_ylim(0, auto=True)
        rug_x, rug_y = values, np.zeros_like(values)

    if rug:
        ax.plot(rug_x, rug_y, **rug_kwargs)
    fill_func(fill_x, fill_y, **fill_kwargs)
    if label:
        ax.legend()

    return ax


def fast_kde(x, cumulative=False, bw=4.5):
    """
    A fft-based Gaussian kernel density estimate (KDE)
    The code was adapted from https://github.com/mfouesneau/faststats

    Parameters
    ----------
    x : Numpy array or list
    cumulative : bool
        If true, estimate the cdf instead of the pdf
    bw : float
        Bandwidth scaling factor for the KDE. Should be larger than 0. The higher this number the
        smoother the KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule
        of thumb (the default rule used by SciPy).

    Returns
    -------
    density: A gridded 1D KDE of the input points (x)
    xmin: minimum value of x
    xmax: maximum value of x
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    len_x = len(x)
    xmin, xmax = np.min(x), np.max(x)

    std_x = entropy(x - xmin) * bw

    n_bins = min(int(len_x**(1/3) * std_x * 2), 200)
    grid, _ = np.histogram(x, bins=n_bins)
    dx = (xmax - xmin) / (n_bins - 1)

    scotts_factor = len_x ** (-0.2)
    kern_nx = int(scotts_factor * 2 * np.pi * std_x)
    kernel = gaussian(kern_nx, scotts_factor * std_x)

    npad = min(n_bins, 2 * kern_nx)
    grid = np.concatenate([grid[npad: 0: -1], grid, grid[n_bins: n_bins - npad: -1]])
    density = convolve(grid, kernel, mode='same')[npad: npad + n_bins]

    norm_factor = len_x * dx * (2 * np.pi * std_x ** 2 * scotts_factor ** 2) ** 0.5

    density = density / norm_factor

    if cumulative:
        cs_density = np.cumsum(density)
        density = cs_density / cs_density[-1]

    return density, xmin, xmax
