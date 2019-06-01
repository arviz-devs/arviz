"""Plot kde or histograms and values from MCMC samples."""
import warnings
from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from ..data import convert_to_dataset
from .distplot import plot_dist
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
    compact=False,
    combined=False,
    legend=False,
    plot_kwargs=None,
    fill_kwargs=None,
    rug_kwargs=None,
    hist_kwargs=None,
    trace_kwargs=None,
    max_plots=40,
):
    """Plot distribution (histogram or kernel density estimates) and sampled values.

    If `divergences` data is available in `sample_stats`, will plot the location of divergences as
    dashed vertical lines.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names : string, or list of strings
        One or more variables to be plotted.
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
    compact : bool
        Plot multidimensional variables in a single plot.
    combined : bool
        Flag for combining multiple chains into a single line. If False (default), chains will be
        plotted separately.
    legend : bool
        Add a legend to the figure with the chain color code.
    plot_kwargs : dict
        Extra keyword arguments passed to `arviz.plot_dist`. Only affects continuous variables.
    fill_kwargs : dict
        Extra keyword arguments passed to `arviz.plot_dist`. Only affects continuous variables.
    rug_kwargs : dict
        Extra keyword arguments passed to `arviz.plot_dist`. Only affects continuous variables.
    hist_kwargs : dict
        Extra keyword arguments passed to `arviz.plot_dist`. Only affects discrete variables.
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

    Show all dimensions of multidimensional variables in the same plot

    .. plot::
        :context: close-figs

        >>> az.plot_trace(data, compact=True)

    Combine all chains into one distribution

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

    if coords is None:
        coords = {}

    data = get_coords(convert_to_dataset(data, group="posterior"), coords)
    var_names = _var_names(var_names, data)

    if divergences:
        divergence_data = get_coords(
            divergence_data, {k: v for k, v in coords.items() if k in ("chain", "draw")}
        )

    if lines is None:
        lines = ()

    num_colors = len(data.chain) + 1 if combined else len(data.chain)
    colors = [
        prop
        for _, prop in zip(
            range(num_colors), cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        )
    ]

    if compact:
        skip_dims = set(data.dims) - {"chain", "draw"}
    else:
        skip_dims = set()

    plotters = list(xarray_var_iter(data, var_names=var_names, combined=True, skip_dims=skip_dims))
    max_plots = len(plotters) if max_plots is None else max_plots
    if len(plotters) > max_plots:
        plotters = plotters[:max_plots]
        warnings.warn(
            "max_plots is smaller than the number of variables to plot "
            "generating only max_plots traceplots",
            SyntaxWarning,
        )

    if figsize is None:
        figsize = (12, len(plotters) * 2)

    if trace_kwargs is None:
        trace_kwargs = {}

    trace_kwargs.setdefault("alpha", 0.35)

    if hist_kwargs is None:
        hist_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    if fill_kwargs is None:
        fill_kwargs = {}
    if rug_kwargs is None:
        rug_kwargs = {}

    hist_kwargs.setdefault("alpha", 0.35)

    figsize, _, titlesize, xt_labelsize, linewidth, _ = _scale_fig_size(
        figsize, textsize, rows=len(plotters), cols=2
    )
    trace_kwargs.setdefault("linewidth", linewidth)
    plot_kwargs.setdefault("linewidth", linewidth)

    _, axes = plt.subplots(
        len(plotters), 2, squeeze=False, figsize=figsize, constrained_layout=True
    )

    for idx, (var_name, selection, value) in enumerate(plotters):
        value = np.atleast_2d(value)

        if len(value.shape) == 2:
            _plot_chains(
                axes,
                idx,
                value,
                data,
                colors,
                combined,
                xt_labelsize,
                trace_kwargs,
                hist_kwargs,
                plot_kwargs,
                fill_kwargs,
                rug_kwargs,
            )
        else:
            value = value.reshape((value.shape[0], value.shape[1], -1))
            for sub_idx in range(value.shape[2]):
                _plot_chains(
                    axes,
                    idx,
                    value[..., sub_idx],
                    data,
                    colors,
                    combined,
                    xt_labelsize,
                    trace_kwargs,
                    hist_kwargs,
                    plot_kwargs,
                    fill_kwargs,
                    rug_kwargs,
                )

        if value[0].dtype.kind == "i":
            xticks = get_bins(value)
            axes[idx, 0].set_xticks(xticks[:-1])
        axes[idx, 0].set_yticks([])
        for col in (0, 1):
            axes[idx, col].set_title(make_label(var_name, selection), fontsize=titlesize, wrap=True)
            axes[idx, col].tick_params(labelsize=xt_labelsize)

        xlims = [ax.get_xlim() for ax in axes[idx, :]]
        ylims = [ax.get_ylim() for ax in axes[idx, :]]

        if divergences:
            div_selection = {k: v for k, v in selection.items() if k in divergence_data.dims}
            divs = divergence_data.sel(**div_selection).values
            # if combined:
            #     divs = divs.flatten()
            divs = np.atleast_2d(divs)

            for chain, chain_divs in enumerate(divs):
                div_draws = data.draw.values[chain_divs]
                div_idxs = np.arange(len(chain_divs))[chain_divs]
                if div_idxs.size > 0:
                    if divergences == "top":
                        ylocs = [ylim[1] for ylim in ylims]
                    else:
                        ylocs = [ylim[0] for ylim in ylims]
                    values = value[chain, div_idxs]
                    axes[idx, 1].plot(
                        div_draws,
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
            axes[idx, 0].vlines(line_values, *ylims[0], colors="black", linewidth=1.5, alpha=0.75)
            axes[idx, 1].hlines(
                line_values, *xlims[1], colors="black", linewidth=1.5, alpha=trace_kwargs["alpha"]
            )
        axes[idx, 0].set_ylim(bottom=0, top=ylims[0][1])
        axes[idx, 1].set_xlim(left=data.draw.min(), right=data.draw.max())
        axes[idx, 1].set_ylim(*ylims[1])
    if legend:
        handles = [
            Line2D([], [], color=color, label=chain_id)
            for chain_id, color in zip(data.chain.values, colors)
        ]
        if combined:
            handles.insert(0, Line2D([], [], color=colors[-1], label="combined"))
        axes[0, 1].legend(handles=handles, title="chain")
    return axes


def _plot_chains(
    axes,
    idx,
    value,
    data,
    colors,
    combined,
    xt_labelsize,
    trace_kwargs,
    hist_kwargs,
    plot_kwargs,
    fill_kwargs,
    rug_kwargs,
):
    for chain_idx, row in enumerate(value):
        axes[idx, 1].plot(data.draw.values, row, color=colors[chain_idx], **trace_kwargs)

        if not combined:
            plot_kwargs["color"] = colors[chain_idx]
            plot_dist(
                row,
                textsize=xt_labelsize,
                ax=axes[idx, 0],
                hist_kwargs=hist_kwargs,
                plot_kwargs=plot_kwargs,
                fill_kwargs=fill_kwargs,
                rug_kwargs=rug_kwargs,
            )

    if combined:
        plot_kwargs["color"] = colors[-1]
        plot_dist(
            value.flatten(),
            textsize=xt_labelsize,
            ax=axes[idx, 0],
            hist_kwargs=hist_kwargs,
            plot_kwargs=plot_kwargs,
            fill_kwargs=fill_kwargs,
            rug_kwargs=rug_kwargs,
        )
