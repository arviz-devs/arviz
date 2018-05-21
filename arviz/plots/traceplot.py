import numpy as np
import matplotlib.pyplot as plt

from .kdeplot import fast_kde
from .plot_utils import make_2d, get_bins, _scale_text
from ..utils import get_varnames, trace_to_dataframe


def traceplot(trace, varnames=None, figsize=None, textsize=None, lines=None, combined=False,
              grid=True, shade=0.35, priors=None, prior_shade=1, prior_style='--', bw=4.5,
              skip_first=0, ax=None):
    """Plot samples histograms and values.

    Parameters
    ----------
    trace : Pandas DataFrame or PyMC3 trace
        Posterior samples
    varnames : list of variable names
        Variables to be plotted, if None all variable are plotted
    figsize : figure size tuple
        If None, size is (12, num of variables * 2) inch
    textsize: int
        Text size for labels, titles and lines. If None it will be autoscaled based on figsize.
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
    trace = trace_to_dataframe(trace[skip_first:], combined)
    varnames = get_varnames(trace, varnames)

    if figsize is None:
        figsize = (12, len(varnames) * 2)

    textsize, linewidth, _ = _scale_text(figsize, textsize=textsize, scale_ratio=1)

    _, ax = plt.subplots(len(varnames), 2, squeeze=False, figsize=figsize)

    for i, varname in enumerate(varnames):
        if priors is not None:
            prior = priors[i]
        else:
            prior = None

        data = trace[varname].values
        data = np.squeeze(data)
        data = make_2d(data)
        width = len(data)
        if data.dtype.kind == 'i':
            hist_objs = _histplot_op(ax[i, 0], data, shade, prior, prior_shade, prior_style)
            colors = [h[-1][0].get_facecolor() for h in hist_objs]
        else:
            artists = _kdeplot_op(ax[i, 0], data, bw, linewidth, prior, prior_shade, prior_style)[0]
            colors = [a[0].get_color() for a in artists]
        ax[i, 0].set_title(varname, fontsize=textsize)
        ax[i, 0].grid(grid)
        ax[i, 1].set_title(varname, fontsize=textsize)
        ax[i, 1].plot(range(width), data, lw=linewidth, alpha=shade)

        ax[i, 0].set_yticks([])
        ax[i, 0].tick_params(labelsize=textsize)
        ax[i, 1].tick_params(labelsize=textsize)

        if lines:
            try:
                if isinstance(lines[varname], (float, int)):
                    line_values, colors = [lines[varname]], ['C3']
                else:
                    line_values = np.atleast_1d(lines[varname]).ravel()
                    if len(colors) != len(line_values):
                        raise AssertionError("An incorrect number of lines was specified for "
                                             "'{}'. Expected an iterable of length {} or to "
                                             " a scalar".format(varname, len(colors)))
                for color, line_value in zip(colors, line_values):
                    ax[i, 0].axvline(x=line_value, color=color, lw=1.5, alpha=0.75)
                    ax[i, 1].axhline(y=line_value, color=color, lw=1.5, alpha=shade)
            except KeyError:
                pass

        ax[i, 0].set_ylim(ymin=0)
    plt.tight_layout()
    return ax


def _histplot_op(ax, data, shade=.35, prior=None, prior_shade=1, prior_style='--'):
    """Add a histogram for each column of the data to the provided axes."""
    hists = []
    for column in data.T:
        bins = get_bins(column)
        hists.append(ax.hist(column, bins=bins, alpha=shade, align='left',
                             density=True))
        if prior is not None:
            x_sample = prior.rvs(1000)
            x = np.arange(x_sample.min(), x_sample.max())
            pmf = prior.pmf(x)
            ax.step(x, pmf, where='mid', alpha=prior_shade, ls=prior_style)
    xticks = get_bins(data, max_bins=10, fenceposts=1)
    ax.set_xticks(xticks)

    return hists


def _kdeplot_op(ax, data, bw, linewidth, prior=None, prior_shade=1, prior_style='--'):
    """Get a list of density and likelihood plots, if a prior is provided."""
    densities = []
    priors = []
    errored = []
    for i, col in enumerate(data.T):
        try:
            density, lower, upper = fast_kde(col, bw=bw)
            x = np.linspace(lower, upper, len(density))
            densities.append(ax.plot(x, density, lw=linewidth))
            if prior is not None:
                x_sample = prior.rvs(10000)
                x = np.linspace(x_sample.min(), x_sample.max(), 1000)
                pdf = prior.pdf(x)
                priors.append(ax.plot(x, pdf, alpha=prior_shade, ls=prior_style))

        except ValueError:
            errored.append(str(i))

    if errored:
        ax.text(.27, .47, 'WARNING: KDE plot failed for: ' + ','.join(errored),
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10},
                style='italic')

    return densities, priors
