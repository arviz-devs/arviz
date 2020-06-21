"""Matplotlib traceplot."""
from itertools import cycle

import warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from . import backend_kwarg_defaults, backend_show
from ...distplot import plot_dist
from ...rankplot import plot_rank
from ...plot_utils import _scale_fig_size, make_label, format_coords_as_labels, _dealiase_sel_kwargs
from ....numeric_utils import get_bins


def plot_trace(
    data,
    var_names,  # pylint: disable=unused-argument
    divergences,
    kind,
    figsize,
    rug,
    lines,
    compact_prop,
    combined,
    chain_prop,
    legend,
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
        list of tuple of (var_name, {'coord': selection}, [line_positions]) to be overplotted as
        vertical lines on the density and horizontal lines on the trace.
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

    backend_kwargs = {**backend_kwarg_defaults(), **backend_kwargs}

    textsize = plot_kwargs.pop("textsize", 10)

    figsize, _, titlesize, xt_labelsize, linewidth, _ = _scale_fig_size(
        figsize, textsize, rows=len(plotters), cols=2
    )

    trace_kwargs.setdefault("linewidth", linewidth)
    plot_kwargs.setdefault("linewidth", linewidth)

    if axes is None:
        _, axes = plt.subplots(len(plotters), 2, squeeze=False, figsize=figsize, **backend_kwargs)

    axes = np.atleast_2d(axes)
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

    for idx, (var_name, selection, value) in enumerate(plotters):
        value = np.atleast_2d(value)

        if len(value.shape) == 2:
            if compact_prop:
                aux_plot_kwargs = _dealiase_sel_kwargs(plot_kwargs, compact_prop, 0)
                aux_trace_kwargs = _dealiase_sel_kwargs(trace_kwargs, compact_prop, 0)
            else:
                aux_plot_kwargs = plot_kwargs
                aux_trace_kwargs = trace_kwargs
            _plot_chains_mpl(
                axes,
                idx,
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
            )
        else:
            sub_data = data[var_name].sel(**selection)
            legend_labels = format_coords_as_labels(sub_data, skip_dims=("chain", "draw"))
            legend_title = ", ".join(
                [
                    "{}".format(coord_name)
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
                aux_plot_kwargs = _dealiase_sel_kwargs(plot_kwargs, compact_prop_iter, sub_idx)
                aux_trace_kwargs = _dealiase_sel_kwargs(trace_kwargs, compact_prop_iter, sub_idx)
                _plot_chains_mpl(
                    axes,
                    idx,
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
                )
                if legend:
                    handles.append(
                        Line2D(
                            [],
                            [],
                            label=label,
                            **_dealiase_sel_kwargs(aux_plot_kwargs, chain_prop, 0)
                        )
                    )
            if legend:
                axes[idx, 0].legend(handles=handles, title=legend_title)

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
                    if kind == "trace":
                        axes[idx, 1].plot(
                            div_draws,
                            np.zeros_like(div_idxs) + ylocs[1],
                            marker="|",
                            color="black",
                            markeredgewidth=1.5,
                            markersize=30,
                            linestyle="None",
                            alpha=hist_kwargs["alpha"],
                            zorder=0.6,
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
                        zorder=0.6,
                    )
                    axes[idx, 0].set_ylim(*ylims[0])

        for _, _, vlines in (j for j in lines if j[0] == var_name and j[1] == selection):
            if isinstance(vlines, (float, int)):
                line_values = [vlines]
            else:
                line_values = np.atleast_1d(vlines).ravel()
                if not np.issubdtype(line_values.dtype, np.number):
                    raise ValueError(
                        "line-positions should be numeric, found {}".format(line_values)
                    )
            axes[idx, 0].vlines(line_values, *ylims[0], colors="black", linewidth=1.5, alpha=0.75)
            if kind == "trace":
                axes[idx, 1].hlines(
                    line_values,
                    *xlims[1],
                    colors="black",
                    linewidth=1.5,
                    alpha=trace_kwargs["alpha"]
                )
        axes[idx, 0].set_ylim(ylims[0])
        if kind == "trace":
            axes[idx, 1].set_xlim(left=data.draw.min(), right=data.draw.max())
            axes[idx, 1].set_ylim(*ylims[1])
    if legend:
        legend_kwargs = trace_kwargs if combined else plot_kwargs
        handles = [
            Line2D(
                [], [], label=chain_id, **_dealiase_sel_kwargs(legend_kwargs, chain_prop, chain_id)
            )
            for chain_id in range(data.dims["chain"])
        ]
        if combined:
            handles.insert(
                0,
                Line2D(
                    [], [], label="combined", **_dealiase_sel_kwargs(plot_kwargs, chain_prop, -1)
                ),
            )
        axes[0, 0].legend(handles=handles, title="chain", loc="upper right")

    if backend_show(show):
        plt.show()

    return axes


def _plot_chains_mpl(
    axes,
    idx,
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
):
    for chain_idx, row in enumerate(value):
        if kind == "trace":
            aux_kwargs = _dealiase_sel_kwargs(trace_kwargs, chain_prop, chain_idx)
            axes[idx, 1].plot(data.draw.values, row, **aux_kwargs)

        if not combined:
            aux_kwargs = _dealiase_sel_kwargs(plot_kwargs, chain_prop, chain_idx)
            plot_dist(
                values=row,
                textsize=xt_labelsize,
                rug=rug,
                ax=axes[idx, 0],
                hist_kwargs=hist_kwargs,
                plot_kwargs=aux_kwargs,
                fill_kwargs=fill_kwargs,
                rug_kwargs=rug_kwargs,
                backend="matplotlib",
                show=False,
            )

    if kind == "rank_bars":
        plot_rank(data=value, kind="bars", ax=axes[idx, 1], **rank_kwargs)
    elif kind == "rank_vlines":
        plot_rank(data=value, kind="vlines", ax=axes[idx, 1], **rank_kwargs)

    if combined:
        aux_kwargs = _dealiase_sel_kwargs(plot_kwargs, chain_prop, -1)
        plot_dist(
            values=value.flatten(),
            textsize=xt_labelsize,
            rug=rug,
            ax=axes[idx, 0],
            hist_kwargs=hist_kwargs,
            plot_kwargs=aux_kwargs,
            fill_kwargs=fill_kwargs,
            rug_kwargs=rug_kwargs,
            backend="matplotlib",
            show=False,
        )
