"""Matplotlib traceplot."""
import warnings
from itertools import cycle

from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

from ....stats.density_utils import get_bins
from ...distplot import plot_dist
from ...plot_utils import _scale_fig_size, format_coords_as_labels
from ...rankplot import plot_rank
from . import backend_kwarg_defaults, backend_show, dealiase_sel_kwargs, matplotlib_kwarg_dealiaser


def plot_trace(
    data,
    var_names,  # pylint: disable=unused-argument
    divergences,
    kind,
    figsize,
    rug,
    lines,
    circ_var_names,
    circ_var_units,
    compact,
    compact_prop,
    combined,
    chain_prop,
    legend,
    labeller,
    plot_kwargs,
    fill_kwargs,
    rug_kwargs,
    hist_kwargs,
    trace_kwargs,
    rank_kwargs,
    plotters,
    divergence_data,
    axes,
    backend_kwargs,
    backend_config,  # pylint: disable=unused-argument
    show,
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
    divergences : {"bottom", "top", None, False}
        Plot location of divergences on the traceplots. Options are "bottom", "top", or False-y.
    kind : {"trace", "rank_bar", "rank_vlines"}, optional
        Choose between plotting sampled values per iteration and rank plots.
    figsize : figure size tuple
        If None, size is (12, variables * 2)
    rug : bool
        If True adds a rugplot. Defaults to False. Ignored for 2D KDE. Only affects continuous
        variables.
    lines : tuple or list
        List of tuple of (var_name, {'coord': selection}, [line_positions]) to be overplotted as
        vertical lines on the density and horizontal lines on the trace.
    circ_var_names : string, or list of strings
        List of circular variables to account for when plotting KDE.
    circ_var_units : str
        Whether the variables in `circ_var_names` are in "degrees" or "radians".
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
    rank_kwargs : dict
        Extra keyword arguments passed to `arviz.plot_rank`
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
        >>> coords = {'school': ['Choate', 'Lawrenceville']}
        >>> az.plot_trace(data, var_names=('theta_t', 'theta'), coords=coords)

    Show all dimensions of multidimensional variables in the same plot

    .. plot::
        :context: close-figs

        >>> az.plot_trace(data, compact=True)

    Combine all chains into one distribution

    .. plot::
        :context: close-figs

        >>> az.plot_trace(data, var_names=('theta_t', 'theta'), coords=coords, combined=True)


    Plot reference lines against distribution and trace

    .. plot::
        :context: close-figs

        >>> lines = (('theta_t',{'school': "Choate"}, [-1]),)
        >>> az.plot_trace(data, var_names=('theta_t', 'theta'), coords=coords, lines=lines)

    """
    # Set plot default backend kwargs
    if backend_kwargs is None:
        backend_kwargs = {}

    if circ_var_names is None:
        circ_var_names = []

    backend_kwargs = {**backend_kwarg_defaults(), **backend_kwargs}

    if lines is None:
        lines = ()

    num_chain_props = len(data.chain) + 1 if combined else len(data.chain)
    if not compact:
        chain_prop = "color" if chain_prop is None else chain_prop
    else:
        chain_prop = (
            {
                "linestyle": ("solid", "dotted", "dashed", "dashdot"),
            }
            if chain_prop is None
            else chain_prop
        )
        compact_prop = "color" if compact_prop is None else compact_prop

    if isinstance(chain_prop, str):
        chain_prop = {chain_prop: plt.rcParams["axes.prop_cycle"].by_key()[chain_prop]}
    if isinstance(chain_prop, tuple):
        warnings.warn(
            "chain_prop as a tuple will be deprecated in a future warning, use a dict instead",
            FutureWarning,
        )
        chain_prop = {chain_prop[0]: chain_prop[1]}
    chain_prop = {
        prop_name: [prop for _, prop in zip(range(num_chain_props), cycle(props))]
        for prop_name, props in chain_prop.items()
    }

    if isinstance(compact_prop, str):
        compact_prop = {compact_prop: plt.rcParams["axes.prop_cycle"].by_key()[compact_prop]}
    if isinstance(compact_prop, tuple):
        warnings.warn(
            "compact_prop as a tuple will be deprecated in a future warning, use a dict instead",
            FutureWarning,
        )
        compact_prop = {compact_prop[0]: compact_prop[1]}

    if figsize is None:
        figsize = (12, len(plotters) * 2)

    backend_kwargs.setdefault("figsize", figsize)

    trace_kwargs = matplotlib_kwarg_dealiaser(trace_kwargs, "plot")
    trace_kwargs.setdefault("alpha", 0.35)

    hist_kwargs = matplotlib_kwarg_dealiaser(hist_kwargs, "hist")
    hist_kwargs.setdefault("alpha", 0.35)

    plot_kwargs = matplotlib_kwarg_dealiaser(plot_kwargs, "plot")
    fill_kwargs = matplotlib_kwarg_dealiaser(fill_kwargs, "fill_between")
    rug_kwargs = matplotlib_kwarg_dealiaser(rug_kwargs, "scatter")
    rank_kwargs = matplotlib_kwarg_dealiaser(rank_kwargs, "bar")

    textsize = plot_kwargs.pop("textsize", 10)

    figsize, _, titlesize, xt_labelsize, linewidth, _ = _scale_fig_size(
        figsize, textsize, rows=len(plotters), cols=2
    )

    trace_kwargs.setdefault("linewidth", linewidth)
    plot_kwargs.setdefault("linewidth", linewidth)

    # Check the input for lines
    if lines is not None:
        all_var_names = set(plotter[0] for plotter in plotters)

        invalid_var_names = set()
        for line in lines:
            if line[0] not in all_var_names:
                invalid_var_names.add(line[0])
        if invalid_var_names:
            warnings.warn(
                "A valid var_name should be provided, found {} expected from {}".format(
                    invalid_var_names, all_var_names
                )
            )

    if axes is None:
        fig = plt.figure(**backend_kwargs)
        spec = gridspec.GridSpec(ncols=2, nrows=len(plotters), figure=fig)

    # pylint: disable=too-many-nested-blocks
    for idx, (var_name, selection, isel, value) in enumerate(plotters):
        for idy in range(2):
            value = np.atleast_2d(value)

            circular = var_name in circ_var_names and not idy
            if var_name in circ_var_names and idy:
                circ_units_trace = circ_var_units
            else:
                circ_units_trace = False

            if axes is None:
                ax = fig.add_subplot(spec[idx, idy], polar=circular)
            else:
                ax = axes[idx, idy]

            if len(value.shape) == 2:
                if compact_prop:
                    aux_plot_kwargs = dealiase_sel_kwargs(plot_kwargs, compact_prop, 0)
                    aux_trace_kwargs = dealiase_sel_kwargs(trace_kwargs, compact_prop, 0)
                else:
                    aux_plot_kwargs = plot_kwargs
                    aux_trace_kwargs = trace_kwargs

                ax = _plot_chains_mpl(
                    ax,
                    idy,
                    value,
                    data,
                    chain_prop,
                    combined,
                    xt_labelsize,
                    rug,
                    kind,
                    aux_trace_kwargs,
                    hist_kwargs,
                    aux_plot_kwargs,
                    fill_kwargs,
                    rug_kwargs,
                    rank_kwargs,
                    circular,
                    circ_var_units,
                    circ_units_trace,
                )

            else:
                sub_data = data[var_name].sel(**selection)
                legend_labels = format_coords_as_labels(sub_data, skip_dims=("chain", "draw"))
                legend_title = ", ".join(
                    [
                        f"{coord_name}"
                        for coord_name in sub_data.coords
                        if coord_name not in {"chain", "draw"}
                    ]
                )
                value = value.reshape((value.shape[0], value.shape[1], -1))
                compact_prop_iter = {
                    prop_name: [prop for _, prop in zip(range(value.shape[2]), cycle(props))]
                    for prop_name, props in compact_prop.items()
                }
                handles = []
                for sub_idx, label in zip(range(value.shape[2]), legend_labels):
                    aux_plot_kwargs = dealiase_sel_kwargs(plot_kwargs, compact_prop_iter, sub_idx)
                    aux_trace_kwargs = dealiase_sel_kwargs(trace_kwargs, compact_prop_iter, sub_idx)
                    ax = _plot_chains_mpl(
                        ax,
                        idy,
                        value[..., sub_idx],
                        data,
                        chain_prop,
                        combined,
                        xt_labelsize,
                        rug,
                        kind,
                        aux_trace_kwargs,
                        hist_kwargs,
                        aux_plot_kwargs,
                        fill_kwargs,
                        rug_kwargs,
                        rank_kwargs,
                        circular,
                        circ_var_units,
                        circ_units_trace,
                    )
                    if legend:
                        handles.append(
                            Line2D(
                                [],
                                [],
                                label=label,
                                **dealiase_sel_kwargs(aux_plot_kwargs, chain_prop, 0),
                            )
                        )
                if legend and idy == 0:
                    ax.legend(handles=handles, title=legend_title)

            if value[0].dtype.kind == "i" and idy == 0:
                xticks = get_bins(value)
                ax.set_xticks(xticks[:-1])
            y = 1 / textsize
            if not idy:
                ax.set_yticks([])
                if circular:
                    y = 0.13 if selection else 0.12
            ax.set_title(
                labeller.make_label_vert(var_name, selection, isel),
                fontsize=titlesize,
                wrap=True,
                y=textsize * y,
            )
            ax.tick_params(labelsize=xt_labelsize)

            xlims = ax.get_xlim()
            ylims = ax.get_ylim()

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
                            ylocs = ylims[1]
                        else:
                            ylocs = ylims[0]
                        values = value[chain, div_idxs]

                        if circular:
                            tick = [ax.get_rmin() + ax.get_rmax() * 0.60, ax.get_rmax()]
                            for val in values:
                                ax.plot(
                                    [val, val],
                                    tick,
                                    color="black",
                                    markeredgewidth=1.5,
                                    markersize=30,
                                    alpha=trace_kwargs["alpha"],
                                    zorder=0.6,
                                )
                        else:
                            if kind == "trace" and idy:
                                ax.plot(
                                    div_draws,
                                    np.zeros_like(div_idxs) + ylocs,
                                    marker="|",
                                    color="black",
                                    markeredgewidth=1.5,
                                    markersize=30,
                                    linestyle="None",
                                    alpha=hist_kwargs["alpha"],
                                    zorder=0.6,
                                )
                            elif not idy:
                                ax.plot(
                                    values,
                                    np.zeros_like(values) + ylocs,
                                    marker="|",
                                    color="black",
                                    markeredgewidth=1.5,
                                    markersize=30,
                                    linestyle="None",
                                    alpha=trace_kwargs["alpha"],
                                    zorder=0.6,
                                )

            for _, _, vlines in (j for j in lines if j[0] == var_name and j[1] == selection):
                if isinstance(vlines, (float, int)):
                    line_values = [vlines]
                else:
                    line_values = np.atleast_1d(vlines).ravel()
                    if not np.issubdtype(line_values.dtype, np.number):
                        raise ValueError(f"line-positions should be numeric, found {line_values}")
                if idy:
                    ax.hlines(
                        line_values,
                        xlims[0],
                        xlims[1],
                        colors="black",
                        linewidth=1.5,
                        alpha=trace_kwargs["alpha"],
                    )

                else:
                    ax.vlines(
                        line_values,
                        ylims[0],
                        ylims[1],
                        colors="black",
                        linewidth=1.5,
                        alpha=trace_kwargs["alpha"],
                    )

        if kind == "trace" and idy:
            ax.set_xlim(left=data.draw.min(), right=data.draw.max())

    if legend:
        legend_kwargs = trace_kwargs if combined else plot_kwargs
        handles = [
            Line2D(
                [], [], label=chain_id, **dealiase_sel_kwargs(legend_kwargs, chain_prop, chain_id)
            )
            for chain_id in range(data.dims["chain"])
        ]
        if combined:
            handles.insert(
                0,
                Line2D(
                    [], [], label="combined", **dealiase_sel_kwargs(plot_kwargs, chain_prop, -1)
                ),
            )
        ax.figure.axes[0].legend(handles=handles, title="chain", loc="upper right")

    if axes is None:
        axes = np.array(ax.figure.axes).reshape(-1, 2)

    if backend_show(show):
        plt.show()

    return axes


def _plot_chains_mpl(
    axes,
    idy,
    value,
    data,
    chain_prop,
    combined,
    xt_labelsize,
    rug,
    kind,
    trace_kwargs,
    hist_kwargs,
    plot_kwargs,
    fill_kwargs,
    rug_kwargs,
    rank_kwargs,
    circular,
    circ_var_units,
    circ_units_trace,
):

    if not circular:
        circ_var_units = False

    for chain_idx, row in enumerate(value):
        if kind == "trace":
            aux_kwargs = dealiase_sel_kwargs(trace_kwargs, chain_prop, chain_idx)
            if idy:
                axes.plot(data.draw.values, row, **aux_kwargs)
                if circ_units_trace == "degrees":
                    y_tick_locs = axes.get_yticks()
                    y_tick_labels = [i + 2 * 180 if i < 0 else i for i in np.rad2deg(y_tick_locs)]
                    axes.yaxis.set_major_locator(mticker.FixedLocator(y_tick_locs))
                    axes.set_yticklabels([f"{i:.0f}°" for i in y_tick_labels])

        if not combined:
            aux_kwargs = dealiase_sel_kwargs(plot_kwargs, chain_prop, chain_idx)
            if not idy:
                axes = plot_dist(
                    values=row,
                    textsize=xt_labelsize,
                    rug=rug,
                    ax=axes,
                    hist_kwargs=hist_kwargs,
                    plot_kwargs=aux_kwargs,
                    fill_kwargs=fill_kwargs,
                    rug_kwargs=rug_kwargs,
                    backend="matplotlib",
                    show=False,
                    is_circular=circ_var_units,
                )

    if kind == "rank_bars" and idy:
        axes = plot_rank(data=value, kind="bars", ax=axes, **rank_kwargs)
    elif kind == "rank_vlines" and idy:
        axes = plot_rank(data=value, kind="vlines", ax=axes, **rank_kwargs)

    if combined:
        aux_kwargs = dealiase_sel_kwargs(plot_kwargs, chain_prop, -1)
        if not idy:
            axes = plot_dist(
                values=value.flatten(),
                textsize=xt_labelsize,
                rug=rug,
                ax=axes,
                hist_kwargs=hist_kwargs,
                plot_kwargs=aux_kwargs,
                fill_kwargs=fill_kwargs,
                rug_kwargs=rug_kwargs,
                backend="matplotlib",
                show=False,
                is_circular=circ_var_units,
            )
    return axes
