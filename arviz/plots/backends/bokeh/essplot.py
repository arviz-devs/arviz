# pylint: disable=all
"""Bokeh ESS plots."""
import bokeh.plotting as bkp
import numpy as np
from bokeh.layouts import gridplot
from bokeh.models import Dash, Span, ColumnDataSource
from bokeh.models.annotations import Title
from scipy.stats import rankdata

from . import backend_kwarg_defaults, backend_show
from ...plot_utils import (
    make_label,
    _create_axes_grid,
)


def plot_ess(
    ax,
    plotters,
    xdata,
    ess_tail_dataset,
    mean_ess,
    sd_ess,
    idata,
    data,
    text_x,
    text_va,
    kind,
    extra_methods,
    rows,
    cols,
    figsize,
    kwargs,
    extra_kwargs,
    text_kwargs,
    _linewidth,
    _markersize,
    n_samples,
    relative,
    min_ess,
    xt_labelsize,
    titlesize,
    ax_labelsize,
    ylabel,
    rug,
    rug_kind,
    rug_kwargs,
    hline_kwargs,
    backend_kwargs,
    show,
):
    """Bokeh essplot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }
    if ax is None:
        _, ax = _create_axes_grid(
            len(plotters),
            rows,
            cols,
            figsize=figsize,
            squeeze=False,
            constrained_layout=True,
            backend="bokeh",
            backend_kwargs=backend_kwargs,
        )
    else:
        ax = np.atleast_2d(ax)

    for (var_name, selection, x), ax_ in zip(
        plotters, (item for item in ax.flatten() if item is not None)
    ):
        ax_.circle(np.asarray(xdata), np.asarray(x), size=6)
        if kind == "evolution":
            ax_.line(np.asarray(xdata), np.asarray(x), legend_label="bulk")
            ess_tail = ess_tail_dataset[var_name].sel(**selection)
            ax_.line(np.asarray(xdata), np.asarray(ess_tail), color="orange", legend_label="tail")
            ax_.circle(np.asarray(xdata), np.asarray(ess_tail), size=6, color="orange")
        elif rug:
            if rug_kwargs is None:
                rug_kwargs = {}
            if not hasattr(idata, "sample_stats"):
                raise ValueError("InferenceData object must contain sample_stats for rug plot")
            if not hasattr(idata.sample_stats, rug_kind):
                raise ValueError("InferenceData does not contain {} data".format(rug_kind))

            rug_kwargs.setdefault("space", 0.1)
            _rug_kwargs = {}
            _rug_kwargs.setdefault("size", 8)
            _rug_kwargs.setdefault("line_color", rug_kwargs.get("line_color", "black"))
            _rug_kwargs.setdefault("line_width", 1)
            _rug_kwargs.setdefault("line_alpha", 0.35)
            _rug_kwargs.setdefault("angle", np.pi / 2)

            values = data[var_name].sel(**selection).values.flatten()
            mask = idata.sample_stats[rug_kind].values.flatten()
            values = rankdata(values)[mask]
            rug_space = np.max(x) * rug_kwargs.pop("space")
            rug_x, rug_y = values / (len(mask) - 1), np.zeros_like(values) - rug_space

            glyph = Dash(x="rug_x", y="rug_y", **_rug_kwargs)
            cds_rug = ColumnDataSource({"rug_x": np.asarray(rug_x), "rug_y": np.asarray(rug_y)})
            ax_.add_glyph(cds_rug, glyph)

            hline = Span(
                location=0,
                dimension="width",
                line_color="black",
                line_width=_linewidth,
                line_alpha=0.7,
            )

            ax_.renderers.append(hline)

        if extra_methods:
            mean_ess_i = mean_ess[var_name].sel(**selection).values.item()
            sd_ess_i = sd_ess[var_name].sel(**selection).values.item()

            hline = Span(
                location=mean_ess_i,
                dimension="width",
                line_color="black",
                line_width=2,
                line_dash="dashed",
                line_alpha=1.0,
            )

            ax_.renderers.append(hline)

            hline = Span(
                location=sd_ess_i,
                dimension="width",
                line_color="black",
                line_width=1,
                line_dash="dashed",
                line_alpha=1.0,
            )

            ax_.renderers.append(hline)

        hline = Span(
            location=400 / n_samples if relative else min_ess,
            dimension="width",
            line_color="red",
            line_width=3,
            line_dash="dashed",
            line_alpha=1.0,
        )

        ax_.renderers.append(hline)

        title = Title()
        title.text = make_label(var_name, selection)
        ax_.title = title

        ax_.xaxis.axis_label = "Total number of draws" if kind == "evolution" else "Quantile"
        ax_.yaxis.axis_label = ylabel.format("Relative ESS" if relative else "ESS")

    if backend_show(show):
        grid = gridplot(ax.tolist(), toolbar_location="above")
        bkp.show(grid)

    return ax
