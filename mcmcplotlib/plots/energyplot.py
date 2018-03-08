import numpy as np
import matplotlib.pyplot as plt
from .kdeplot import kdeplot
from ..utils.utils import get_stats


def energyplot(trace, kind='kde', figsize=None, legend=True, shade=0.35, bw=4.5,
               frame=True, kwargs_shade=None, ax=None, **kwargs):
    """Plot energy transition distribution and marginal energy distribution in
    order to diagnose poor exploration by HMC algorithms.

    Parameters
    ----------

    trace : result of MCMC run
    kind : str
        Type of plot to display (kde or histogram)
    figsize : figure size tuple
        If None, size is (8 x 6)
    legend : bool
        Flag for plotting legend (defaults to True)
    shade : float
        Alpha blending value for the shaded area under the curve, between 0
        (no shade) and 1 (opaque). Defaults to 0.35
    bw : float
        Bandwidth scaling factor for the KDE. Should be larger than 0. The higher this number the
        smoother the KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule
        of thumb (the default rule used by SciPy). Only works if `kind='kde'`.
    frame : bool
        Flag for plotting frame around figure.
    ax : axes
        Matplotlib axes.
    kwargs_shade : dicts, optional
        Additional keywords passed to `fill_between` (to control the shade)
    Returns
    -------

    ax : matplotlib axes
    """

    energy = get_stats(trace, 'energy')
    
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    series = [('Marginal energy distribution', energy - energy.mean()),
              ('Energy transition distribution', np.diff(energy))]

    if figsize is None:
        figsize = (8, 6)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if kwargs_shade is None:
        kwargs_shade = {}

    if kind == 'kde':
        for label, value in series:
            kdeplot(value, label=label, shade=shade, bw=bw, ax=ax, kwargs_shade=kwargs_shade,
                    **kwargs)

    elif kind == 'hist':
        for label, value in series:
            ax.hist(value, alpha=shade, label=label, **kwargs)

    else:
        raise ValueError('Plot type {} not recognized.'.format(kind))

    ax.set_xticks([])
    ax.set_yticks([])

    if not frame:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if legend:
        ax.legend()
