import numpy as np
import matplotlib.pyplot as plt
from .artists import histplot_op, kdeplot_op
from .plot_utils import identity_transform, get_axis, make_2d
from ..utils.utils import expand_variable_names, trace_to_dataframe


def traceplot(trace, varnames=None, transform=identity_transform, figsize=None, lines=None,
              combined=False, grid=True, alpha=0.35, priors=None, prior_alpha=1,
              prior_style='--', bw=4.5, skip_first=0, ax=None):
    """Plot samples histograms and values.

    Parameters
    ----------

    trace : result of MCMC run
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
    alpha : float
        Alpha value for plot line. Defaults to 0.35.
    priors : iterable of scipy distributions
        Prior distribution(s) to be plotted alongside posterior. Defaults to None (no prior plots).
    prior_alpha : float
        Alpha value for prior plot. Defaults to 1.
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

    if varnames is None:
        varnames = np.unique(trace.columns)
    else:
        varnames = expand_variable_names(trace, varnames)

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
            hist_objs = histplot_op(
                ax[i, 0], d, alpha, prior, prior_alpha, prior_style)
            colors = [h[-1][0].get_facecolor() for h in hist_objs]
        else:
            artists = kdeplot_op(ax[i, 0], d, bw, prior, prior_alpha, prior_style)[0]
            colors = [a[0].get_color() for a in artists]
        ax[i, 0].set_title(v)
        ax[i, 0].grid(grid)
        ax[i, 1].set_title(v)
        ax[i, 1].plot(range(width), d, alpha=alpha)

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
                    ax[i, 1].axhline(y=l, color=c, lw=1.5, alpha=alpha)
            except KeyError:
                pass

        ax[i, 0].set_ylim(ymin=0)
    plt.tight_layout()
    return ax
