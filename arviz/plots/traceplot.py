"""Plot histogram and values from MCMC samples."""
import matplotlib.pyplot as plt
import numpy as np

from ..data import convert_to_dataset
from .kdeplot import plot_kde
from .plot_utils import _scale_fig_size, get_bins, xarray_var_iter, make_label, get_coords
from ..utils import _var_names


def plot_trace(
    data,
    var_names=None,
    coords=None,
    divergences="bottom",
    figsize=None,
    textsize=None,
    lines=None,
    combined=False,
    kde_kwargs=None,
    hist_kwargs=None,
    trace_kwargs=None,
):
    """Plot samples histograms and values.

    If `divergences` data is available in `sample_stats`, will plot the location of divergences as
    dashed vertical lines.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names : list of variable names
        Variables to be plotted, two variables are required.
    coords : mapping, optional
        Coordinates of var_names to be plotted. Passed to `Dataset.sel`
    divergences : {"bottom", "top", None, False}
        Plot location of divergences on the traceplots. Options are "bottom", "top", or False-y.
    figsize : figure size tuple
        If None, size is (12, variables * 2)
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.
    lines : tuple
        Tuple of (var_name, {'coord': selection}, [line, positions]) to be overplotted as
        vertical lines on the density and horizontal lines on the trace.
    combined : bool
        Flag for combining multiple chains into a single line. If False (default), chains will be
        plotted separately.
    kde_kwargs : dict
        Extra keyword arguments passed to `arviz.plot_kde`. Only affects continuous variables.
    hist_kwargs : dict
        Extra keyword arguments passed to `plt.hist`. Only affects discrete variables.
    trace_kwargs : dict
        Extra keyword arguments passed to `plt.plot`
    Returns
    -------
    axes : matplotlib axes


    Examples
    --------
    Plot a subset variables

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('non_centered_eight')
        >>> coords = {'theta_t_dim_0': [0, 1], 'school':['Lawrenceville']}
        >>> az.plot_trace(data, var_names=('theta_t', 'theta'), coords=coords)

    Combine all chains into one distribution and trace

    .. plot::
        :context: close-figs

        >>> coords = {'theta_t_dim_0': [0, 1], 'school':['Lawrenceville']}
        >>> az.plot_trace(data, var_names=('theta_t', 'theta'), coords=coords, combined=True)


    Plot reference lines against distribution and trace

    .. plot::
        :context: close-figs

        >>> lines = (('theta_t',{'theta_t_dim_0':0}, [-1]),)
        >>> coords = {'theta_t_dim_0': [0, 1], 'school':['Lawrenceville']}
        >>> az.plot_trace(data, var_names=('theta_t', 'theta'), coords=coords, lines=lines)
    """
    if divergences:
        try:
            divergence_data = convert_to_dataset(data, group="sample_stats").diverging
        except (ValueError, AttributeError):  # No sample_stats, or no `.diverging`
            divergences = False

    data = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names)

    if coords is None:
        coords = {}

    if lines is None:
        lines = ()

    plotters = list(xarray_var_iter(get_coords(data, coords), var_names=var_names, combined=True))

    if figsize is None:
        figsize = (12, len(plotters) * 2)

    if trace_kwargs is None:
        trace_kwargs = {}

    trace_kwargs.setdefault("alpha", 0.35)

    if kde_kwargs is None:
        kde_kwargs = {}

    if hist_kwargs is None:
        hist_kwargs = {}

    hist_kwargs.setdefault("alpha", 0.35)

    figsize, _, titlesize, xt_labelsize, linewidth, _ = _scale_fig_size(
        figsize, textsize, rows=len(plotters), cols=2
    )
    trace_kwargs.setdefault("linewidth", linewidth)
    kde_kwargs.setdefault("plot_kwargs", {"linewidth": linewidth})

    _, axes = plt.subplots(
        len(plotters), 2, squeeze=False, figsize=figsize, constrained_layout=True
    )

    colors = {}
    for idx, (var_name, selection, value) in enumerate(plotters):
        colors[idx] = []
        if combined:
            value = value.flatten()
        value = np.atleast_2d(value)

        for row in value:
            axes[idx, 1].plot(np.arange(len(row)), row, **trace_kwargs)

            colors[idx].append(axes[idx, 1].get_lines()[-1].get_color())
            kde_kwargs["plot_kwargs"]["color"] = colors[idx][-1]
            if row.dtype.kind == "i":
                _histplot_op(axes[idx, 0], row, **hist_kwargs)
            else:
                plot_kde(row, textsize=xt_labelsize, ax=axes[idx, 0], **kde_kwargs)

        axes[idx, 0].set_yticks([])
        for col in (0, 1):
            axes[idx, col].set_title(make_label(var_name, selection), fontsize=titlesize, wrap=True)
            axes[idx, col].tick_params(labelsize=xt_labelsize)

        xlims = [ax.get_xlim() for ax in axes[idx, :]]
        ylims = [ax.get_ylim() for ax in axes[idx, :]]

        if divergences:
            div_selection = {k: v for k, v in selection.items() if k in divergence_data.dims}
            divs = divergence_data.sel(**div_selection).values
            if combined:
                divs = divs.flatten()
            divs = np.atleast_2d(divs)

            for chain, chain_divs in enumerate(divs):
                div_idxs = np.arange(len(chain_divs))[chain_divs]
                if div_idxs.size > 0:
                    if divergences == "top":
                        ylocs = [ylim[1] for ylim in ylims]
                    else:
                        ylocs = [ylim[0] for ylim in ylims]
                    values = value[chain, div_idxs]
                    axes[idx, 1].plot(
                        div_idxs,
                        np.zeros_like(div_idxs) + ylocs[1],
                        marker="|",
                        color="black",
                        markeredgewidth=1.5,
                        markersize=30,
                        linestyle="None",
                        alpha=hist_kwargs["alpha"],
                        zorder=-5,
                    )
                    axes[idx, 1].set_ylim(*ylims[1])
                    axes[idx, 0].plot(
                        values,
                        np.zeros_like(values) + ylocs[0],
                        marker="|",
                        color="black",
                        markeredgewidth=1.5,
                        markersize=30,
                        linestyle="None",
                        alpha=trace_kwargs["alpha"],
                        zorder=-5,
                    )
                    axes[idx, 0].set_ylim(*ylims[0])

        for _, _, vlines in (j for j in lines if j[0] == var_name and j[1] == selection):
            if isinstance(vlines, (float, int)):
                line_values = [vlines]
            else:
                line_values = np.atleast_1d(vlines).ravel()
            axes[idx, 0].vlines(
                line_values, *ylims[0], colors=colors[idx][0], linewidth=1.5, alpha=0.75
            )
            axes[idx, 1].hlines(
                line_values,
                *xlims[1],
                colors=colors[idx][0],
                linewidth=1.5,
                alpha=trace_kwargs["alpha"]
            )
        axes[idx, 0].set_ylim(bottom=0, top=ylims[0][1])
        axes[idx, 1].set_xlim(left=0, right=data.draw.max())
        axes[idx, 1].set_ylim(*ylims[1])
    return axes


def _histplot_op(ax, data, **kwargs):
    """Add a histogram for the data to the axes."""
    bins = get_bins(data)
    ax.hist(data, bins=bins, align="left", density=True, **kwargs)
    xticks = get_bins(data, max_bins=10, fenceposts=1)
    ax.set_xticks(xticks)
    return ax
