import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
from . import kdeplot
from .kdeplot import fast_kde
from ..stats import hpd
from ..utils import trace_to_dataframe, expand_variable_names
from .plot_utils import identity_transform


def posteriorplot(trace, varnames=None, transform=identity_transform, figsize=None, text_size=None,
                  alpha=0.05, round_to=1, point_estimate='mean', rope=None, ref_val=None,
                  kind='kde', bw=4.5, skip_first=0, ax=None, **kwargs):
    """Plot Posterior densities in the style of John K. Kruschke's book.

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
    text_size : int
        Text size of the point_estimates, axis ticks, and HPD (Default:16)
    alpha : float
        Defines range for High Posterior Density
    round_to : int
        Controls formatting for floating point numbers
    point_estimate: str
        Must be in ('mode', 'mean', 'median')
    rope: list or numpy array
        Lower and upper values of the Region Of Practical Equivalence
    ref_val: float or list-like
        display the percentage below and above the values in ref_val. If a list is provided, its
        length should match the number of variables.
    kind: str
        `kde` or `hist`. The former will plot a KDE, the last one a histogram. For discrete
        variables a this argument is ignored and a histogram is always used.
    bw : float
        Bandwidth scaling factor for the KDE. Should be larger than 0. The higher this number the
        smoother the KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule
        of thumb (the default rule used by SciPy). Only works if `kind == kde` is True.
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

    trace = trace_to_dataframe(trace, combined=True)[skip_first:]

    if varnames is not None:
        varnames = expand_variable_names(trace, varnames)
        trace = trace[varnames]

    if ax is None:
        fig, ax = _create_axes_grid(figsize, trace)

    var_num = trace.shape[1]
    if ref_val is None:
        ref_val = [None] * var_num
    elif np.isscalar(ref_val):
        ref_val = [ref_val for _ in range(var_num)]

    if rope is None:
        rope = [None] * var_num
    elif np.ndim(rope) == 1:
        rope = [rope] * var_num

    for idx, (a, v) in enumerate(zip(np.atleast_1d(ax), trace.columns)):
        tr_values = transform(trace[v])
        _plot_posterior_op(tr_values, ax=a, bw=bw, kind=kind, point_estimate=point_estimate,
                           round_to=round_to, alpha=alpha, ref_val=ref_val[idx], rope=rope[idx],
                           text_size=_scale_text(figsize, text_size), **kwargs)
        a.set_title(v, fontsize=_scale_text(figsize, text_size))

    plt.tight_layout()
    return ax


def _plot_posterior_op(trace_values, ax, bw, kind, point_estimate, round_to,
                       alpha, ref_val, rope, text_size=16, **kwargs):
    """Artist to draw posterior."""
    def format_as_percent(x, round_to=0):
        return '{0:.{1:d}f}%'.format(100 * x, round_to)

    def display_ref_val(ref_val):
        less_than_ref_probability = (trace_values < ref_val).mean()
        greater_than_ref_probability = (trace_values >= ref_val).mean()
        ref_in_posterior = "{} <{:g}< {}".format(format_as_percent(less_than_ref_probability, 1),
                                                 ref_val,
                                                 format_as_percent(greater_than_ref_probability, 1))
        ax.axvline(ref_val, ymin=0.02, ymax=.75,
                   color='C1', linewidth=4, alpha=0.65)
        ax.text(trace_values.mean(), plot_height * 0.6, ref_in_posterior,
                size=text_size, horizontalalignment='center')

    def display_rope(rope):
        ax.plot(rope, (plot_height * 0.02, plot_height * 0.02),
                linewidth=20, color='C2', alpha=0.75)
        text_props = dict(
            size=text_size, horizontalalignment='center', color='C2')
        ax.text(rope[0], plot_height * 0.14, rope[0], **text_props)
        ax.text(rope[1], plot_height * 0.14, rope[1], **text_props)

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
                density, l, u = fast_kde(trace_values, bw)
                x = np.linspace(l, u, len(density))
                point_value = x[np.argmax(density)]
            else:
                point_value = mode(trace_values.round(round_to))[0][0]
        elif point_estimate == 'median':
            point_value = np.median(trace_values)
        point_text = '{point_estimate}={point_value:.{round_to}f}'.format(point_estimate=point_estimate,
                                                                          point_value=point_value, round_to=round_to)

        ax.text(point_value, plot_height * 0.8, point_text,
                size=text_size, horizontalalignment='center')

    def display_hpd():
        hpd_intervals = hpd(trace_values, alpha=alpha)
        ax.plot(hpd_intervals, (plot_height * 0.02,
                                plot_height * 0.02), linewidth=4, color='k')
        ax.text(hpd_intervals[0], plot_height * 0.07,
                hpd_intervals[0].round(round_to),
                size=text_size, horizontalalignment='right')
        ax.text(hpd_intervals[1], plot_height * 0.07,
                hpd_intervals[1].round(round_to),
                size=text_size, horizontalalignment='left')
        ax.text((hpd_intervals[0] + hpd_intervals[1]) / 2, plot_height * 0.2,
                format_as_percent(1 - alpha) + ' HPD',
                size=text_size, horizontalalignment='center')

    def format_axes():
        ax.yaxis.set_ticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.tick_params(axis='x', direction='out', width=1, length=3,
                       color='0.5', labelsize=text_size)
        ax.spines['bottom'].set_color('0.5')

    def set_key_if_doesnt_exist(d, key, value):
        if key not in d:
            d[key] = value

    if kind == 'kde' and isinstance(trace_values.iloc[0], float):
        kdeplot(trace_values, alpha=kwargs.pop(
            'alpha', 1), bw=bw, ax=ax, **kwargs)

    else:
        set_key_if_doesnt_exist(kwargs, 'bins', 30)
        set_key_if_doesnt_exist(kwargs, 'edgecolor', 'w')
        set_key_if_doesnt_exist(kwargs, 'align', 'right')
        ax.hist(trace_values, **kwargs)

    plot_height = ax.get_ylim()[1]

    format_axes()
    display_hpd()
    display_point_estimate()
    if ref_val is not None:
        display_ref_val(ref_val)
    if rope is not None:
        display_rope(rope)


def _scale_text(figsize, text_size):
    """Scale text to figsize."""

    if text_size is None and figsize is not None:
        if figsize[0] <= 11:
            return 12
        else:
            return figsize[0]
    else:
        return text_size


def _create_axes_grid(figsize, trace):
    l_trace = trace.shape[1]
    if l_trace == 1:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        n = np.ceil(l_trace / 2.0).astype(int)
        if figsize is None:
            figsize = (12, n * 2.5)
        fig, ax = plt.subplots(n, 2, figsize=figsize)
        ax = ax.reshape(2 * n)
        if l_trace % 2 == 1:
            ax[-1].set_axis_off()
            ax = ax[:-1]
    return fig, ax
