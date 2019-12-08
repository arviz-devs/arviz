# pylint: disable=all
"""Bokeh Traceplot."""
from collections.abc import Iterable
from itertools import cycle
import warnings

import bokeh.plotting as bkp
from bokeh.models import ColumnDataSource, Dash, Span
from bokeh.models.annotations import Title
from bokeh.layouts import gridplot
import matplotlib.pyplot as plt
import numpy as np


from ....data import convert_to_dataset
from ...distplot import plot_dist
from ...plot_utils import _scale_fig_size, xarray_var_iter, make_label, get_coords
from ....rcparams import rcParams
from ....utils import _var_names


def _plot_trace_bokeh(
    data,
    var_names=None,
    coords=None,
    divergences="bottom",
    figsize=None,
    rug=False,
    lines=None,
    compact=False,
    combined=False,
    legend=False,
    plot_kwargs=None,
    fill_kwargs=None,
    rug_kwargs=None,
    hist_kwargs=None,
    trace_kwargs=None,
    backend_kwargs=None,
    show=True,
):
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
    max_plots = rcParams["plot.max_subplots"]
    max_plots = len(plotters) if max_plots is None else max_plots
    if len(plotters) > max_plots:
        warnings.warn(
            "rcParams['plot.max_subplots'] ({max_plots}) is smaller than the number "
            "of variables to plot ({len_plotters}), generating only {max_plots} "
            "plots".format(max_plots=max_plots, len_plotters=len(plotters)),
            SyntaxWarning,
        )
        plotters = plotters[:max_plots]

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

    figsize, _, _, _, linewidth, _ = _scale_fig_size(figsize, 10, rows=len(plotters), cols=2)

    trace_kwargs.setdefault("line_width", linewidth)
    plot_kwargs.setdefault("line_width", linewidth)

    if backend_kwargs is None:
        backend_kwargs = dict()

    backend_kwargs.setdefault("tools", rcParams["plot.bokeh.tools"])
    backend_kwargs.setdefault("output_backend", rcParams["plot.bokeh.output_backend"])
    backend_kwargs.setdefault(
        "height", int(figsize[1] * rcParams["plot.bokeh.figure.dpi"] // len(plotters))
    )
    backend_kwargs.setdefault("width", int(figsize[0] * rcParams["plot.bokeh.figure.dpi"] // 2))

    axes = []
    for i in range(len(plotters)):
        if i != 0:
            _axes = [
                bkp.figure(**backend_kwargs),
                bkp.figure(x_range=axes[0][1].x_range, **backend_kwargs),
            ]
        else:
            _axes = [bkp.figure(**backend_kwargs), bkp.figure(**backend_kwargs)]
        axes.append(_axes)

    axes = np.array(axes)

    cds_data = {}
    cds_var_groups = {}
    draw_name = "draw"

    for var_name, selection, value in list(
        xarray_var_iter(data, var_names=var_names, combined=True)
    ):
        if selection:
            cds_name = "{}_ARVIZ_CDS_SELECTION_{}".format(
                var_name,
                "_".join(
                    str(item)
                    for key, value in selection.items()
                    for item in (
                        [key, value]
                        if (isinstance(value, str) or not isinstance(value, Iterable))
                        else [key, *value]
                    )
                ),
            )
        else:
            cds_name = var_name

        if var_name not in cds_var_groups:
            cds_var_groups[var_name] = []
        cds_var_groups[var_name].append(cds_name)

        for chain_idx, _ in enumerate(data.chain.values):
            if chain_idx not in cds_data:
                cds_data[chain_idx] = {}
            _data = value[chain_idx]
            cds_data[chain_idx][cds_name] = _data

    while any(key == draw_name for key in cds_data[0]):
        draw_name += "w"

    for chain_idx in cds_data:
        cds_data[chain_idx][draw_name] = data.draw.values

    cds_data = {chain_idx: ColumnDataSource(cds) for chain_idx, cds in cds_data.items()}

    for idx, (var_name, selection, value) in enumerate(plotters):
        value = np.atleast_2d(value)

        if len(value.shape) == 2:
            y_name = (
                var_name
                if not selection
                else "{}_ARVIZ_CDS_SELECTION_{}".format(
                    var_name,
                    "_".join(
                        str(item)
                        for key, value in selection.items()
                        for item in (
                            (key, value)
                            if (isinstance(value, str) or not isinstance(value, Iterable))
                            else (key, *value)
                        )
                    ),
                )
            )
            if rug:
                rug_kwargs["y"] = y_name
            _plot_chains_bokeh(
                ax_density=axes[idx, 0],
                ax_trace=axes[idx, 1],
                data=cds_data,
                x_name=draw_name,
                y_name=y_name,
                colors=colors,
                combined=combined,
                rug=rug,
                legend=legend,
                trace_kwargs=trace_kwargs,
                hist_kwargs=hist_kwargs,
                plot_kwargs=plot_kwargs,
                fill_kwargs=fill_kwargs,
                rug_kwargs=rug_kwargs,
            )
        else:
            for y_name in cds_var_groups[var_name]:
                if rug:
                    rug_kwargs["y"] = y_name
                _plot_chains_bokeh(
                    ax_density=axes[idx, 0],
                    ax_trace=axes[idx, 1],
                    data=cds_data,
                    x_name=draw_name,
                    y_name=y_name,
                    colors=colors,
                    combined=combined,
                    rug=rug,
                    legend=legend,
                    trace_kwargs=trace_kwargs,
                    hist_kwargs=hist_kwargs,
                    plot_kwargs=plot_kwargs,
                    fill_kwargs=fill_kwargs,
                    rug_kwargs=rug_kwargs,
                )

        for col in (0, 1):
            _title = Title()
            _title.text = make_label(var_name, selection)
            axes[idx, col].title = _title

        for _, _, vlines in (j for j in lines if j[0] == var_name and j[1] == selection):
            if isinstance(vlines, (float, int)):
                line_values = [vlines]
            else:
                line_values = np.atleast_1d(vlines).ravel()

            for line_value in line_values:
                vline = Span(
                    location=line_value,
                    dimension="height",
                    line_color="black",
                    line_width=1.5,
                    line_alpha=0.75,
                )
                hline = Span(
                    location=line_value,
                    dimension="width",
                    line_color="black",
                    line_width=1.5,
                    line_alpha=trace_kwargs["alpha"],
                )

                axes[idx, 0].renderers.append(vline)
                axes[idx, 1].renderers.append(hline)

        if legend:
            for col in (0, 1):
                axes[idx, col].legend.location = "top_left"
                axes[idx, col].legend.click_policy = "hide"
        else:
            for col in (0, 1):
                if axes[idx, col].legend:
                    axes[idx, col].legend.visible = False

        if divergences:
            div_density_kwargs = {}
            div_density_kwargs.setdefault("size", 14)
            div_density_kwargs.setdefault("line_color", "red")
            div_density_kwargs.setdefault("line_width", 2)
            div_density_kwargs.setdefault("line_alpha", 0.50)
            div_density_kwargs.setdefault("angle", np.pi / 2)

            div_trace_kwargs = {}
            div_trace_kwargs.setdefault("size", 14)
            div_trace_kwargs.setdefault("line_color", "red")
            div_trace_kwargs.setdefault("line_width", 2)
            div_trace_kwargs.setdefault("line_alpha", 0.50)
            div_trace_kwargs.setdefault("angle", np.pi / 2)

            div_selection = {k: v for k, v in selection.items() if k in divergence_data.dims}
            divs = divergence_data.sel(**div_selection).values
            divs = np.atleast_2d(divs)

            for chain, chain_divs in enumerate(divs):
                div_idxs = np.arange(len(chain_divs))[chain_divs]
                if div_idxs.size > 0:
                    values = value[chain, div_idxs]
                    tmp_cds = ColumnDataSource({"y": values, "x": div_idxs})
                    if divergences == "top":
                        y_div_trace = value.max()
                    else:
                        y_div_trace = value.min()
                    glyph_density = Dash(x="y", y=0.0, **div_density_kwargs)
                    glyph_trace = Dash(x="x", y=y_div_trace, **div_trace_kwargs)

                    axes[idx, 0].add_glyph(tmp_cds, glyph_density)
                    axes[idx, 1].add_glyph(tmp_cds, glyph_trace)

    if show:
        grid = gridplot([list(item) for item in axes], toolbar_location="above")
        bkp.show(grid)

    return axes


def _plot_chains_bokeh(
    ax_density,
    ax_trace,
    data,
    x_name,
    y_name,
    colors,
    combined,
    rug,
    legend,
    trace_kwargs,
    hist_kwargs,
    plot_kwargs,
    fill_kwargs,
    rug_kwargs,
):
    marker = trace_kwargs.pop("marker", True)
    for chain_idx, cds in data.items():
        if legend:
            trace_kwargs["legend_label"] = "chain {}".format(chain_idx)
        ax_trace.line(
            x=x_name, y=y_name, source=cds, line_color=colors[chain_idx], **trace_kwargs,
        )
        if marker:
            ax_trace.circle(
                x=x_name,
                y=y_name,
                source=cds,
                radius=0.30,
                line_color=colors[chain_idx],
                fill_color=colors[chain_idx],
                alpha=0.5,
            )
        if not combined:
            rug_kwargs["cds"] = cds
            if legend:
                plot_kwargs["legend_label"] = "chain {}".format(chain_idx)
            plot_kwargs["line_color"] = colors[chain_idx]
            plot_dist(
                cds.data[y_name],
                ax=ax_density,
                color=colors[chain_idx],
                rug=rug,
                hist_kwargs=hist_kwargs,
                plot_kwargs=plot_kwargs,
                fill_kwargs=fill_kwargs,
                rug_kwargs=rug_kwargs,
                backend="bokeh",
                show=False,
            )

    if combined:
        rug_kwargs["cds"] = data
        if legend:
            plot_kwargs["legend_label"] = "combined chains"
        plot_dist(
            np.concatenate([item.data[y_name] for item in data.values()]).flatten(),
            ax=ax_density,
            color=colors[-1],
            rug=rug,
            hist_kwargs=hist_kwargs,
            plot_kwargs=plot_kwargs,
            fill_kwargs=fill_kwargs,
            rug_kwargs=rug_kwargs,
            backend="bokeh",
            show=False,
        )
