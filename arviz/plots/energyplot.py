"""Plot energy transition distribution in HMC inference."""
import numpy as np
import matplotlib.pyplot as plt

from ..data import convert_to_dataset
from ..stats import bfmi as e_bfmi
from .kdeplot import plot_kde
from .plot_utils import _scale_text


def plot_energy(data, kind='kde', bfmi=True, figsize=None, legend=True, fill_alpha=(1, .75),
                fill_color=('C0', 'C5'), bw=4.5, textsize=None, fill_kwargs=None, plot_kwargs=None,
                ax=None):
    """Plot energy transition distribution and marginal energy distribution in HMC algorithms.

    This may help to diagnose poor exploration by gradient-based algorithms like HMC or NUTS.

    Parameters
    ----------
    data : xarray dataset, or object that can be converted (must represent
           `sample_stats` and have an `energy` variable)
    kind : str
        Type of plot to display (kde or histogram)
    bfmi : bool
        If True add to the plot the value of the estimated Bayesian fraction of missing information
    figsize : figure size tuple
        If None, size is (8 x 6)
    legend : bool
        Flag for plotting legend (defaults to True)
    fill_alpha : tuple of floats
        Alpha blending value for the shaded area under the curve, between 0
        (no shade) and 1 (opaque). Defaults to (1, .75)
    fill_color : tuple of valid matplotlib color
        Color for Marginal energy distribution and Energy transition distribution.
        Defaults to ('C0', 'C5')
    bw : float
        Bandwidth scaling factor for the KDE. Should be larger than 0. The higher this number the
        smoother the KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule
        of thumb (the default rule used by SciPy). Only works if `kind='kde'`
    textsize: int
        Text size for labels
    fill_kwargs : dicts, optional
        Additional keywords passed to `arviz.plot_kde` (to control the shade)
    plot_kwargs : dicts, optional
        Additional keywords passed to `arviz.plot_kde` or `plt.hist` (if type='hist')
    ax : axes
        Matplotlib axes.

    Returns
    -------
    ax : matplotlib axes
    """
    energy = convert_to_dataset(data, group='sample_stats').energy.values

    if figsize is None:
        figsize = (8, 6)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if fill_kwargs is None:
        fill_kwargs = {}

    if plot_kwargs is None:
        plot_kwargs = {}
    textsize, linewidth, _ = _scale_text(figsize, textsize=textsize, scale_ratio=1)

    series = zip(
        fill_alpha,
        fill_color,
        ('Marginal Energy', 'Energy transition'),
        (energy - energy.mean(), np.diff(energy))
    )

    if kind == 'kde':
        for alpha, color, label, value in series:
            fill_kwargs['alpha'] = alpha
            fill_kwargs['color'] = color
            plot_kwargs.setdefault('color', color)
            plot_kwargs.setdefault('alpha', 0)
            plot_kwargs.setdefault('linewidth', linewidth)
            plot_kde(value, bw=bw, label=label, textsize=textsize,
                     plot_kwargs=plot_kwargs, fill_kwargs=fill_kwargs, ax=ax)

    elif kind == 'hist':
        for alpha, color, label, value in series:
            ax.hist(value.flatten(), bins='auto', density=True, alpha=alpha,
                    label=label, color=color, **plot_kwargs)

    else:
        raise ValueError('Plot type {} not recognized.'.format(kind))

    if bfmi:
        for idx, val in enumerate(e_bfmi(energy)):
            ax.plot([], label='chain {:>2} BFMI = {:.2f}'.format(idx, val), alpha=0)

    ax.set_xticks([])
    ax.set_yticks([])

    if legend:
        ax.legend()

    return ax
