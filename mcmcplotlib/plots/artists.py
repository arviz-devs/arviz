import numpy as np
from ..stats.stats import hpd
from .kdeplot import fast_kde, kdeplot


def _histplot_bins(column, bins=100):
    """Helper to get bins for histplot."""
    col_min = np.min(column)
    col_max = np.max(column)
    return range(col_min, col_max + 2, max((col_max - col_min) // bins, 1))


def histplot_op(ax, data, alpha=.35):
    """Add a histogram for each column of the data to the provided axes."""
    hs = []
    for column in data.T:
        hs.append(ax.hist(column, bins=_histplot_bins(
                  column), alpha=alpha, align='left'))
    ax.set_xlim(np.min(data) - 0.5, np.max(data) + 0.5)
    return hs


def kdeplot_op(ax, data, bw, prior=None, prior_alpha=1, prior_style='--'):
    """Get a list of density and likelihood plots, if a prior is provided."""
    ls = []
    pls = []
    errored = []
    for i, d in enumerate(data.T):
        try:
            density, l, u = fast_kde(d, bw)
            x = np.linspace(l, u, len(density))
            if prior is not None:
                p = prior.logpdf(x)
                pls.append(ax.plot(x, np.exp(p), alpha=prior_alpha, ls=prior_style))

            ls.append(ax.plot(x, density))
        except ValueError:
            errored.append(str(i))

    if errored:
        ax.text(.27, .47, 'WARNING: KDE plot failed for: ' + ','.join(errored),
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10},
                style='italic')

    return ls, pls
