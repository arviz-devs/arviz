"""Plot posterior densities."""
from typing import Optional
from numbers import Number
import numpy as np
from scipy.stats import mode

from ..data import convert_to_dataset
from ..stats import hpd
from .kdeplot import plot_kde, _fast_kde
from .plot_utils import (
    xarray_var_iter,
    _scale_fig_size,
    make_label,
    default_grid,
    _create_axes_grid,
    get_coords,
    filter_plotters_list,
)
from ..utils import _var_names, format_sig_figs


def plot_posterior(
    data,
    var_names=None,
    coords=None,
    figsize=None,
    textsize=None,
    credible_interval=0.94,
    round_to: Optional[int] = None,
    point_estimate="mean",
    group="posterior",
    rope=None,
    ref_val=None,
    kind="kde",
    bw=4.5,
    bins=None,
    ax=None,
    **kwargs
):
    """Plot Posterior densities in the style of John K. Kruschke's book.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names : list of variable names
        Variables to be plotted, two variables are required.
    coords : mapping, optional
        Coordinates of var_names to be plotted. Passed to `Dataset.sel`
    figsize : tuple
        Figure size. If None it will be defined automatically.
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.
    credible_interval : float, optional
        Credible intervals. Defaults to 0.94. Use None to hide the credible interval
    round_to : int, optional
        Controls formatting of floats. Defaults to 2 or the integer part, whichever is bigger.
    point_estimate: str
        Must be in ('mode', 'mean', 'median', None)
    group : str, optional
        Specifies which InferenceData group should be plotted. Defaults to ‘posterior’.
    rope: tuple or dictionary of tuples
        Lower and upper values of the Region Of Practical Equivalence. If a list is provided, its
        length should match the number of variables.
    ref_val: float or dictionary of floats
        display the percentage below and above the values in ref_val. Must be None (default),
        a constant, a list or a dictionary like see an example below. If a list is provided, its
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
    ax : axes
        Matplotlib axes. Defaults to None.
    **kwargs
        Passed as-is to plt.hist() or plt.plot() function depending on the value of `kind`.

    Returns
    -------
    ax : matplotlib axes

    Examples
    --------
    Show a default kernel density plot following style of John Kruschke

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('centered_eight')
        >>> az.plot_posterior(data)

    Plot subset variables by specifying variable name exactly

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(data, var_names=['mu'])

    Plot Region of Practical Equivalence (rope) for all distributions

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(data, var_names=['mu', 'theta'], rope=(-1, 1))

    Plot Region of Practical Equivalence for selected distributions

    .. plot::
        :context: close-figs

        >>> rope = {'mu': [{'rope': (-2, 2)}], 'theta': [{'school': 'Choate', 'rope': (2, 4)}]}
        >>> az.plot_posterior(data, var_names=['mu', 'theta'], rope=rope)


    Add reference lines

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(data, var_names=['mu', 'theta'], ref_val=0)

    Show point estimate of distribution

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(data, var_names=['mu', 'theta'], point_estimate='mode')

    Show reference values using variable names and coordinates

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(data, ref_val= {"theta": [{"school": "Deerfield", "ref_val": 4},
                                                        {"school": "Choate", "ref_val": 3}]})

    Show reference values using a list

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(data, ref_val=[1] + [5] * 8 + [1])


    Plot posterior as a histogram

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(data, var_names=['mu'], kind='hist')

    Change size of credible interval

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(data, var_names=['mu'], credible_interval=.75)
    """
    data = convert_to_dataset(data, group=group)
    var_names = _var_names(var_names, data)

    if coords is None:
        coords = {}

    plotters = filter_plotters_list(
        list(xarray_var_iter(get_coords(data, coords), var_names=var_names, combined=True)),
        "plot_posterior",
    )
    length_plotters = len(plotters)
    rows, cols = default_grid(length_plotters)

    (figsize, ax_labelsize, titlesize, xt_labelsize, _linewidth, _) = _scale_fig_size(
        figsize, textsize, rows, cols
    )
    kwargs.setdefault("linewidth", _linewidth)

    if ax is None:
        _, ax = _create_axes_grid(
            length_plotters, rows, cols, figsize=figsize, squeeze=False, constrained_layout=True
        )
    idx = 0
    for (var_name, selection, x), ax_ in zip(plotters, np.ravel(ax)):
        _plot_posterior_op(
            idx,
            x.flatten(),
            var_name,
            selection,
            ax=ax_,
            bw=bw,
            bins=bins,
            kind=kind,
            point_estimate=point_estimate,
            round_to=round_to,
            credible_interval=credible_interval,
            ref_val=ref_val,
            rope=rope,
            ax_labelsize=ax_labelsize,
            xt_labelsize=xt_labelsize,
            **kwargs
        )
        idx += 1
        ax_.set_title(make_label(var_name, selection), fontsize=titlesize, wrap=True)

    return ax


def _plot_posterior_op(
    idx,
    values,
    var_name,
    selection,
    ax,
    bw,
    linewidth,
    bins,
    kind,
    point_estimate,
    credible_interval,
    ref_val,
    rope,
    ax_labelsize,
    xt_labelsize,
    round_to: Optional[int] = None,
    **kwargs
):  # noqa: D202
    """Artist to draw posterior."""

    significant_fig_func = lambda v: format_sig_figs(v, default=round_to)

    def format_as_percent(x, round_to=0):
        return "{0:.{1:d}f}%".format(100 * x, round_to)

    def display_ref_val():
        if ref_val is None:
            return
        elif isinstance(ref_val, dict):
            val = None
            for sel in ref_val.get(var_name, []):
                if all(
                    k in selection and selection[k] == v for k, v in sel.items() if k != "ref_val"
                ):
                    val = sel["ref_val"]
                    break
            if val is None:
                return
        elif isinstance(ref_val, list):
            val = ref_val[idx]
        elif isinstance(ref_val, Number):
            val = ref_val
        else:
            raise ValueError(
                "Argument `ref_val` must be None, a constant, a list or a "
                'dictionary like {"var_name": [{"ref_val": ref_val}]}'
            )
        less_than_ref_probability = (values < val).mean()
        greater_than_ref_probability = (values >= val).mean()
        ref_in_posterior = "{} <{:g}< {}".format(
            format_as_percent(less_than_ref_probability, 1),
            val,
            format_as_percent(greater_than_ref_probability, 1),
        )
        ax.axvline(val, ymin=0.05, ymax=0.75, color="C1", lw=linewidth, alpha=0.65)
        ax.text(
            values.mean(),
            plot_height * 0.6,
            ref_in_posterior,
            size=ax_labelsize,
            color="C1",
            weight="semibold",
            horizontalalignment="center",
        )

    def display_rope():
        if rope is None:
            return
        elif isinstance(rope, dict):
            vals = None
            for sel in rope.get(var_name, []):
                # pylint: disable=line-too-long
                if all(k in selection and selection[k] == v for k, v in sel.items() if k != "rope"):
                    vals = sel["rope"]
                    break
            if vals is None:
                return
        elif len(rope) == 2:
            vals = rope
        else:
            raise ValueError(
                "Argument `rope` must be None, a dictionary like"
                '{"var_name": {"rope": (lo, hi)}}, or an'
                "iterable of length 2"
            )

        ax.plot(
            vals,
            (plot_height * 0.02, plot_height * 0.02),
            lw=linewidth * 5,
            color="C2",
            solid_capstyle="round",
            zorder=0,
            alpha=0.7,
        )
        text_props = {"size": ax_labelsize, "horizontalalignment": "center", "color": "C2"}
        ax.text(vals[0], plot_height * 0.2, vals[0], weight="semibold", **text_props)
        ax.text(vals[1], plot_height * 0.2, vals[1], weight="semibold", **text_props)

    def display_point_estimate():
        if not point_estimate:
            return
        if point_estimate not in ("mode", "mean", "median"):
            raise ValueError("Point Estimate should be in ('mode','mean','median')")
        if point_estimate == "mean":
            point_value = values.mean()
        elif point_estimate == "mode":
            if isinstance(values[0], float):
                density, lower, upper = _fast_kde(values, bw=bw)
                x = np.linspace(lower, upper, len(density))
                point_value = x[np.argmax(density)]
            else:
                point_value = mode(values)[0][0]
        elif point_estimate == "median":
            point_value = np.median(values)
        sig_figs = significant_fig_func(point_value)
        point_text = "{point_estimate}={point_value:.{sig_figs}g}".format(
            point_estimate=point_estimate, point_value=point_value, sig_figs=sig_figs
        )
        ax.text(
            point_value,
            plot_height * 0.8,
            point_text,
            size=ax_labelsize,
            horizontalalignment="center",
        )

    def display_hpd():
        # np.ndarray with 2 entries, min and max
        # pylint: disable=line-too-long
        hpd_intervals = hpd(values, credible_interval=credible_interval)  # type: np.ndarray

        def round_num(n: float) -> str:
            sig_figs = significant_fig_func(n)
            return "{n:.{sig_figs}g}".format(n=n, sig_figs=sig_figs)

        ax.plot(
            hpd_intervals,
            (plot_height * 0.02, plot_height * 0.02),
            lw=linewidth * 2,
            color="k",
            solid_capstyle="round",
        )
        ax.text(
            hpd_intervals[0],
            plot_height * 0.07,
            round_num(hpd_intervals[0]),
            size=ax_labelsize,
            horizontalalignment="center",
        )
        ax.text(
            hpd_intervals[1],
            plot_height * 0.07,
            round_num(hpd_intervals[1]),
            size=ax_labelsize,
            horizontalalignment="center",
        )
        ax.text(
            (hpd_intervals[0] + hpd_intervals[1]) / 2,
            plot_height * 0.3,
            format_as_percent(credible_interval) + " HPD",
            size=ax_labelsize,
            horizontalalignment="center",
        )

    def format_axes():
        ax.yaxis.set_ticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.xaxis.set_ticks_position("bottom")
        ax.tick_params(
            axis="x", direction="out", width=1, length=3, color="0.5", labelsize=xt_labelsize
        )
        ax.spines["bottom"].set_color("0.5")

    if kind == "kde" and values.dtype.kind == "f":
        kwargs.setdefault("linewidth", linewidth)
        plot_kde(
            values,
            bw=bw,
            fill_kwargs={"alpha": kwargs.pop("fill_alpha", 0)},
            plot_kwargs=kwargs,
            ax=ax,
            rug=False,
        )
    else:
        if bins is None:
            if values.dtype.kind == "i":
                xmin = values.min()
                xmax = values.max()
                bins = range(xmin, xmax + 2)
                ax.set_xlim(xmin - 0.5, xmax + 0.5)
            else:
                bins = "auto"
        kwargs.setdefault("align", "left")
        kwargs.setdefault("color", "C0")
        ax.hist(values, bins=bins, alpha=0.35, **kwargs)

    plot_height = ax.get_ylim()[1]

    format_axes()
    if credible_interval is not None:
        display_hpd()
    display_point_estimate()
    display_ref_val()
    display_rope()
