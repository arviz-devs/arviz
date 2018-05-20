import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
from . import kdeplot
from .kdeplot import fast_kde
from ..stats import hpd
from ..utils import trace_to_dataframe, expand_variable_names
from .plot_utils import _scale_text, _create_axes_grid



def posteriorplot(trace, varnames=None, figsize=None, textsize=None, alpha=0.05, round_to=1,
                  point_estimate='mean', rope=None, ref_val=None, kind='kde', bw=4.5, bins=None,
                  skip_first=0, ax=None, **kwargs):
    """
    Plot Posterior densities in the style of John K. Kruschke's book.

    Parameters
    ----------
    trace : Pandas DataFrame or PyMC3 trace
        Posterior samples
    varnames : list of variable names
        Variables to be plotted, if None all variable are plotted
    figsize : tuple
        Figure size. If None, size is (12, num of variables * 2)
    textsize: int
        Text size of the point_estimates, axis ticks, and HPD. If None it will be autoscaled
        based on figsize.
    alpha : float, optional
        Alpha value for (1-alpha)*100% credible intervals. Defaults to 0.05.
    round_to : int
        Controls formatting for floating point numbers
    point_estimate: str
        Must be in ('mode', 'mean', 'median')
    rope: tuple of list of tuples
        Lower and upper values of the Region Of Practical Equivalence. If a list is provided, its
        length should match the number of variables.
    ref_val: float or list-like
        display the percentage below and above the values in ref_val. If a list is provided, its
        length should match the number of variables.
    kind: str
        Type of plot to display (kde or hist) For discrete variables this argument is ignored and
        a histogram is always used.
    bw : float
        Bandwidth scaling factor for the KDE. Should be larger than 0. The higher this number the
        smoother the KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule
        of thumb (the default rule used by SciPy). Only works if `kind == kde`.
    bins : integer or sequence or 'auto', optional
        Controls the number of bins, accepts the same keywords `matplotlib.hist()` does. Only works
        if `kind == hist`. If None (default) it will use `auto` for continuous variables and
        `range(xmin, xmax + 1)` for discrete variables.
    skip_first : int
        Number of first samples not shown in plots (burn-in).
    ax : axes
        Matplotlib axes. Defaults to None.
    **kwargs
        Passed as-is to plt.hist() or plt.plot() function depending on the value of `kind`.

    Returns
    -------
    ax : matplotlib axes

    """

    trace = trace_to_dataframe(trace[skip_first:], combined=True)

    if varnames is not None:
        varnames = expand_variable_names(trace, varnames)
        trace = trace[varnames]

    ax, figsize = _create_axes_grid(trace, figsize, ax)

    textsize, linewidth, _ = _scale_text(figsize, textsize, 1.5)

    var_num = trace.shape[1]
    if ref_val is None:
        ref_val = [None] * var_num
    elif np.isscalar(ref_val):
        ref_val = [ref_val for _ in range(var_num)]

    if rope is None:
        rope = [None] * var_num
    elif np.ndim(rope) == 1:
        rope = [rope] * var_num

    for idx, (ax_, col) in enumerate(zip(np.atleast_1d(ax), trace.columns)):
        _plot_posterior_op(trace[col], ax=ax_, bw=bw, linewidth=linewidth, bins=bins, kind=kind,
                           point_estimate=point_estimate, round_to=round_to, alpha=alpha,
                           ref_val=ref_val[idx], rope=rope[idx], textsize=textsize, **kwargs)
        ax_.set_title(col, fontsize=textsize)

    plt.tight_layout()
    return ax


def _plot_posterior_op(trace_values, ax, bw, linewidth, bins, kind, point_estimate, round_to,
                       alpha, ref_val, rope, textsize, **kwargs):
    """
    Artist to draw posterior.
    """
    def format_as_percent(x, round_to=0):
        return '{0:.{1:d}f}%'.format(100 * x, round_to)

    def display_ref_val(ref_val):
        less_than_ref_probability = (trace_values < ref_val).mean()
        greater_than_ref_probability = (trace_values >= ref_val).mean()
        ref_in_posterior = "{} <{:g}< {}".format(format_as_percent(less_than_ref_probability, 1),
                                                 ref_val,
                                                 format_as_percent(greater_than_ref_probability, 1))
        ax.axvline(ref_val, ymin=0.05, ymax=.75, color='C1', lw=linewidth, alpha=0.65)
        ax.text(trace_values.mean(), plot_height * 0.6, ref_in_posterior, size=textsize,
                color='C1', weight='semibold', horizontalalignment='center')

    def display_rope(rope):
        ax.plot(rope, (plot_height * 0.02, plot_height * 0.02), lw=linewidth*5, color='C2',
                solid_capstyle='round')
        text_props = dict(size=textsize, horizontalalignment='center', color='C2')
        ax.text(rope[0], plot_height * 0.2, rope[0], weight='semibold', **text_props)
        ax.text(rope[1], plot_height * 0.2, rope[1], weight='semibold', **text_props)

    def display_point_estimate():
        if not point_estimate:
            return
        if point_estimate not in ('mode', 'mean', 'median'):
            raise ValueError(
                "Point Estimate should be in ('mode','mean','median')")
        if point_estimate == 'mean':
            point_value = trace_values.mean()
        elif point_estimate == 'mode':
            if isinstance(trace_values.iloc[0], float):
                density, lower, upper = fast_kde(trace_values, bw=bw)
                x = np.linspace(lower, upper, len(density))
                point_value = x[np.argmax(density)]
            else:
                point_value = mode(trace_values.round(round_to))[0][0]
        elif point_estimate == 'median':
            point_value = np.median(trace_values)
        point_text = '{}={:.{}f}'.format(point_estimate, point_value, round_to)

        ax.text(point_value, plot_height * 0.8, point_text, size=textsize,
                horizontalalignment='center')

    def display_hpd():
        hpd_intervals = hpd(trace_values, alpha=alpha)
        ax.plot(hpd_intervals, (plot_height * 0.02, plot_height * 0.02), lw=linewidth*2, color='k',
                solid_capstyle='round')
        ax.text(hpd_intervals[0], plot_height * 0.07,
                hpd_intervals[0].round(round_to),
                size=textsize, horizontalalignment='center')
        ax.text(hpd_intervals[1], plot_height * 0.07,
                hpd_intervals[1].round(round_to),
                size=textsize, horizontalalignment='center')
        ax.text((hpd_intervals[0] + hpd_intervals[1]) / 2, plot_height * 0.3,
                format_as_percent(1 - alpha) + ' HPD',
                size=textsize, horizontalalignment='center')

    def format_axes():
        ax.yaxis.set_ticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.tick_params(axis='x', direction='out', width=1, length=3,
                       color='0.5', labelsize=textsize)
        ax.spines['bottom'].set_color('0.5')

    if kind == 'kde' and isinstance(trace_values.iloc[0], float):
        kdeplot(trace_values,
                fill_alpha=kwargs.pop('fill_alpha', 0.35),
                bw=bw,
                ax=ax,
                lw=linewidth,
                **kwargs)

    else:
        if bins is None:
            if trace_values.dtype.kind == 'i':
                xmin = trace_values.min()
                xmax = trace_values.max()
                bins = range(xmin, xmax + 2)
                ax.set_xlim(xmin - 0.5, xmax + 0.5)
            else:
                bins = 'auto'
        kwargs.setdefault('align', 'left')
        kwargs.setdefault('color', 'C0')
        ax.hist(trace_values, bins=bins, alpha=0.35, **kwargs)

    plot_height = ax.get_ylim()[1]

    format_axes()
    display_hpd()
    display_point_estimate()
    if ref_val is not None:
        display_ref_val(ref_val)
    if rope is not None:
        display_rope(rope)
