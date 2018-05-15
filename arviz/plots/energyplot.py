import numpy as np
import matplotlib.pyplot as plt
from .kdeplot import kdeplot
from ..stats import bfmi as e_bfmi
from ..utils import get_stats


def energyplot(trace, kind='kde', bfmi=True, figsize=None, legend=True, fill_alpha=(1, .75),
               fill_color=('C0', 'C5'), bw=4.5, skip_first=0, kwargs_shade=None, ax=None,
               **kwargs):
    """Plot energy transition distribution and marginal energy distribution in
    order to diagnose poor exploration by HMC algorithms.

    Parameters
    ----------
    trace : Pandas DataFrame or PyMC3 trace
        Posterior samples
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
    skip_first : int
        Number of first samples not shown in plots (burn-in).
    kwargs_shade : dicts, optional
        Additional keywords passed to `fill_between` (to control the shade)
    ax : axes
        Matplotlib axes.

    Returns
    -------
    ax : matplotlib axes
    """

    energy = get_stats(trace[skip_first:], 'energy')

    if figsize is None:
        figsize = (6, 6)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if kwargs_shade is None:
        kwargs_shade = {}

    series = zip(
        fill_alpha,
        fill_color,
        ('Marginal Energy', 'Energy transition'),
        (energy - energy.mean(), np.diff(energy))
    )

    if kind == 'kde':
        for alpha, color, label, value in series:
            kdeplot(value, fill_alpha=alpha, bw=bw, alpha=0, fill_color=color, ax=ax,
                    kwargs_shade=kwargs_shade, **kwargs)
            plt.plot([], label=label, color=color)

    elif kind == 'hist':
        for alpha, color, label, value in series:
            ax.hist(value, alpha=alpha, label=label, color=color, **kwargs)

    else:
        raise ValueError('Plot type {} not recognized.'.format(kind))

    if bfmi:
        for idx, v in enumerate(e_bfmi(trace)):
            plt.plot([], label='chain {:>2} BFMI = {:.2f}'.format(idx, v), alpha=0)

    ax.set_xticks([])
    ax.set_yticks([])

    if legend:
        ax.legend()

    return ax
