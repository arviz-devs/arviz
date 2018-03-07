import numpy as np
from ..stats.stats import hpd
from .kdeplot import fast_kde, kdeplot


def histplot_op(ax, data, alpha=.35, prior=None, prior_alpha=1, prior_style='--'):
    """Add a histogram for each column of the data to the provided axes."""
    hs = []
    for column in data.T:
        bins = range(column.min(), column.max())
        hs.append(ax.hist(column, bins=bins, alpha=alpha, align='left',
                          density=True))
        if prior is not None:
            x_sample = prior.rvs(1000)
            x = np.arange(x_sample.min(), x_sample.max())
            p = prior.pmf(x)
            ax.step(x, p, where='mid', alpha=prior_alpha, ls=prior_style)

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
            ls.append(ax.plot(x, density))
            if prior is not None:
                x_sample = prior.rvs(10000)
                x = np.linspace(x_sample.min(), x_sample.max(), 1000)
                p = prior.pdf(x)
                pls.append(ax.plot(x, p, alpha=prior_alpha, ls=prior_style))

        except ValueError:
            errored.append(str(i))

    if errored:
        ax.text(.27, .47, 'WARNING: KDE plot failed for: ' + ','.join(errored),
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10},
                style='italic')

    return ls, pls
