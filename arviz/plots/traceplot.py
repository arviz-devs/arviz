import numpy as np
import matplotlib.pyplot as plt
from ..stats import hpd
from .kdeplot import fast_kde, kdeplot
from .plot_utils import identity_transform, get_axis, make_2d
from ..utils import get_varnames, trace_to_dataframe


def traceplot(trace, varnames=None, transform=identity_transform, figsize=None, lines=None,
              combined=False, grid=True, shade=0.35, priors=None, prior_shade=1,
              prior_style='--', bw=4.5, skip_first=0, ax=None):
    """Plot samples histograms and values.

    Parameters
    ----------
    trace : Pandas DataFrame or PyMC3 trace
        Posterior samples
    varnames : list of variable names
        Variables to be plotted, if None all variable are plotted
    transform : callable
        Function to transform data (defaults to identity)
    figsize : figure size tuple
        If None, size is (12, num of variables * 2) inch
    lines : dict
        Dictionary of variable name / value  to be overplotted as vertical lines to the posteriors
        and horizontal lines on sample values e.g. mean of posteriors, true values of a simulation.
        If an array of values, line colors are matched to posterior colors. Otherwise, a default
        `C3` line.
    combined : bool
        Flag for combining multiple chains into a single chain. If False (default), chains will be
        plotted separately.
    grid : bool
        Flag for adding gridlines to histogram. Defaults to True.
    shade : float
        Alpha blending value for plot line. Defaults to 0.35.
    priors : iterable of scipy distributions
        Prior distribution(s) to be plotted alongside posterior. Defaults to None (no prior plots).
    prior_Shade : float
        Alpha blending value for prior plot. Defaults to 1.
    prior_style : str
        Line style for prior plot. Defaults to '--' (dashed line).
    bw : float
        Bandwidth scaling factor for the KDE. Should be larger than 0. The higher this number the
        smoother the KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule
        of thumb (the default rule used by SciPy).
    skip_first : int
        Number of first samples not shown in plots (burn-in).
    ax : axes
        Matplotlib axes. Accepts an array of axes, e.g.:

        >>> fig, axs = plt.subplots(3, 2) # 3 RVs
        >>> pymc3.traceplot(trace, ax=axs)

        Creates own axes by default.

    Returns
    -------

    ax : matplotlib axes

    """
    trace = trace_to_dataframe(trace, combined)[skip_first:]
    varnames = get_varnames(trace, varnames)

    if figsize is None:
        figsize = (12, len(varnames) * 2)

    ax = get_axis(ax, len(varnames), 2, squeeze=False, figsize=figsize)

    for i, v in enumerate(varnames):
        if priors is not None:
            prior = priors[i]
        else:
            prior = None

        d = trace[v].values
        d = np.squeeze(transform(d))
        d = make_2d(d)
        width = len(d)
        if d.dtype.kind == 'i':
            hist_objs = _histplot_op(
                ax[i, 0], d, shade, prior, prior_shade, prior_style)
            colors = [h[-1][0].get_facecolor() for h in hist_objs]
        else:
            artists = _kdeplot_op(ax[i, 0], d, bw, prior, prior_shade, prior_style)[0]
            colors = [a[0].get_color() for a in artists]
        ax[i, 0].set_title(v)
        ax[i, 0].grid(grid)
        ax[i, 1].set_title(v)
        ax[i, 1].plot(range(width), d, alpha=shade)

        ax[i, 0].set_yticks([])
        ax[i, 1].set_ylabel("Sample value")

        if lines:
            try:
                if isinstance(lines[v], (float, int)):
                    line_values, colors = [lines[v]], ['C3']
                else:
                    line_values = np.atleast_1d(lines[v]).ravel()
                    if len(colors) != len(line_values):
                        raise AssertionError("An incorrect number of lines was specified for "
                                             "'{}'. Expected an iterable of length {} or to "
                                             " a scalar".format(v, len(colors)))
                for c, l in zip(colors, line_values):
                    ax[i, 0].axvline(x=l, color=c, lw=1.5, alpha=0.75)
                    ax[i, 1].axhline(y=l, color=c, lw=1.5, alpha=shade)
            except KeyError:
                pass

        ax[i, 0].set_ylim(ymin=0)
    plt.tight_layout()
    return ax


def _histplot_op(ax, data, shade=.35, prior=None, prior_shade=1, prior_style='--'):
    """Add a histogram for each column of the data to the provided axes."""
    hs = []
    for column in data.T:
        bins = range(column.min(), column.max() + 2)
        hs.append(ax.hist(column, bins=bins, alpha=shade, align='left',
                          density=True))
        if prior is not None:
            x_sample = prior.rvs(1000)
            x = np.arange(x_sample.min(), x_sample.max())
            p = prior.pmf(x)
            ax.step(x, p, where='mid', alpha=prior_shade, ls=prior_style)
    ax.set_xticks(range(np.min(data), np.max(data) + 1))

    return hs


def _kdeplot_op(ax, data, bw, prior=None, prior_shade=1, prior_style='--'):
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
                pls.append(ax.plot(x, p, alpha=prior_shade, ls=prior_style))

        except ValueError:
            errored.append(str(i))

    if errored:
        ax.text(.27, .47, 'WARNING: KDE plot failed for: ' + ','.join(errored),
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10},
                style='italic')

    return ls, pls
