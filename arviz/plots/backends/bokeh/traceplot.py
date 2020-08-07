# pylint: disable=all
"""Bokeh Traceplot."""
from collections.abc import Iterable
from typing import Dict

import bokeh.plotting as bkp
from bokeh.models import ColumnDataSource, Dash, Span, DataRange1d
from bokeh.models.annotations import Title
import numpy as np

from . import backend_kwarg_defaults
from .. import show_layout
from ...distplot import plot_dist
from ...rankplot import plot_rank
from ...plot_utils import xarray_var_iter, make_label, _scale_fig_size, _dealiase_sel_kwargs
from ....rcparams import rcParams


def plot_trace(
    data,
    var_names,
    divergences,
    kind,
    figsize,
    rug,
    lines,
    combined,
    chain_prop,
    legend,
    plot_kwargs: [Dict],
    fill_kwargs: [Dict],
    rug_kwargs: [Dict],
    hist_kwargs: [Dict],
    trace_kwargs: [Dict],
    rank_kwargs: [Dict],
    plotters,
    divergence_data,
    axes,
    backend_config,
    backend_kwargs: [Dict],
    show,
):
    """Bokeh traceplot."""
    # If divergences are plotted they must be provided
    if divergences is not False:
        assert divergence_data is not None

    if backend_config is None:
        backend_config = {}

    backend_config = {
        **backend_kwarg_defaults(("bounds_y_range", "plot.bokeh.bounds_y_range"),),
        **backend_config,
    }

    # Set plot default backend kwargs
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(("dpi", "plot.bokeh.figure.dpi"),),
        **backend_kwargs,
    }
    dpi = backend_kwargs.pop("dpi")

    backend_kwargs.setdefault("height", int(figsize[1] * dpi // len(plotters)))
    backend_kwargs.setdefault("width", int(figsize[0] * dpi // 2))

    figsize, _, _, _, linewidth, _ = _scale_fig_size(figsize, 10, rows=len(plotters), cols=2)

    trace_kwargs.setdefault("line_width", linewidth)
    plot_kwargs.setdefault("line_width", linewidth)

    if rank_kwargs is None:
        rank_kwargs = {}

    if axes is None:
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

    axes = np.atleast_2d(axes)

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
                chain_prop=chain_prop,
                combined=combined,
                rug=rug,
                kind=kind,
                legend=legend,
                trace_kwargs=trace_kwargs,
                hist_kwargs=hist_kwargs,
                plot_kwargs=plot_kwargs,
                fill_kwargs=fill_kwargs,
                rug_kwargs=rug_kwargs,
                rank_kwargs=rank_kwargs,
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
                    chain_prop=chain_prop,
                    combined=combined,
                    rug=rug,
                    kind=kind,
                    legend=legend,
                    trace_kwargs=trace_kwargs,
                    hist_kwargs=hist_kwargs,
                    plot_kwargs=plot_kwargs,
                    fill_kwargs=fill_kwargs,
                    rug_kwargs=rug_kwargs,
                    rank_kwargs=rank_kwargs,
                )

        for col in (0, 1):
            _title = Title()
            _title.text = make_label(var_name, selection)
            axes[idx, col].title = _title
            axes[idx, col].y_range = DataRange1d(
                bounds=backend_config["bounds_y_range"], min_interval=0.1
            )

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

    show_layout(axes, show)

    return axes


def _plot_chains_bokeh(
    ax_density,
    ax_trace,
    data,
    x_name,
    y_name,
    chain_prop,
    combined,
    rug,
    kind,
    legend,
    trace_kwargs,
    hist_kwargs,
    plot_kwargs,
    fill_kwargs,
    rug_kwargs,
    rank_kwargs,
):
    marker = trace_kwargs.pop("marker", True)
    for chain_idx, cds in data.items():
        if kind == "trace":
            if legend:
                trace_kwargs["legend_label"] = "chain {}".format(chain_idx)
            ax_trace.line(
                x=x_name,
                y=y_name,
                source=cds,
                **_dealiase_sel_kwargs(trace_kwargs, chain_prop, chain_idx)
            )
            if marker:
                ax_trace.circle(
                    x=x_name,
                    y=y_name,
                    source=cds,
                    radius=0.30,
                    alpha=0.5,
                    **_dealiase_sel_kwargs({}, chain_prop, chain_idx)
                )
        if not combined:
            rug_kwargs["cds"] = cds
            if legend:
                plot_kwargs["legend_label"] = "chain {}".format(chain_idx)
            plot_dist(
                cds.data[y_name],
                ax=ax_density,
                rug=rug,
                hist_kwargs=hist_kwargs,
                plot_kwargs=_dealiase_sel_kwargs(plot_kwargs, chain_prop, chain_idx),
                fill_kwargs=fill_kwargs,
                rug_kwargs=rug_kwargs,
                backend="bokeh",
                backend_kwargs={},
                show=False,
            )

    if kind == "rank_bars":
        value = np.array([item.data[y_name] for item in data.values()])
        plot_rank(value, kind="bars", ax=ax_trace, backend="bokeh", show=False, **rank_kwargs)
    elif kind == "rank_vlines":
        value = np.array([item.data[y_name] for item in data.values()])
        plot_rank(value, kind="vlines", ax=ax_trace, backend="bokeh", show=False, **rank_kwargs)

    if combined:
        rug_kwargs["cds"] = data
        if legend:
            plot_kwargs["legend_label"] = "combined chains"
        plot_dist(
            np.concatenate([item.data[y_name] for item in data.values()]).flatten(),
            ax=ax_density,
            rug=rug,
            hist_kwargs=hist_kwargs,
            plot_kwargs=_dealiase_sel_kwargs(plot_kwargs, chain_prop, -1),
            fill_kwargs=fill_kwargs,
            rug_kwargs=rug_kwargs,
            backend="bokeh",
            backend_kwargs={},
            show=False,
        )
