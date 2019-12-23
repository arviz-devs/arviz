"""Bokeh mcseplot."""

import bokeh.plotting as bkp
import numpy as np
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Dash, Span
from bokeh.models.annotations import Title
from scipy.stats import rankdata

from . import backend_kwarg_defaults, backend_show
from ...plot_utils import (
    make_label,
    _create_axes_grid,
)
from ....stats.stats_utils import quantile as _quantile


def plot_mcse(
    ax,
    plotters,
    length_plotters,
    rows,
    cols,
    figsize,
    errorbar,
    rug,
    data,
    probs,
    extra_kwargs,
    extra_methods,
    mean_mcse,
    sd_mcse,
    rug_kwargs,
    idata,
    rug_kind,
    _markersize,
    _linewidth,
    backend_kwargs,
    show,
):
    """Bokeh mcse plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }
    if ax is None:
        _, ax = _create_axes_grid(
            length_plotters,
            rows,
            cols,
            figsize=figsize,
            backend="bokeh",
            backend_kwargs=backend_kwargs,
        )
    else:
        ax = np.atleast_2d(ax)

    for (var_name, selection, x), ax_ in zip(
        plotters, (item for item in ax.flatten() if item is not None)
    ):
        if errorbar or rug:
            values = data[var_name].sel(**selection).values.flatten()
        if errorbar:
            quantile_values = _quantile(values, probs)
            ax_.dash(probs, quantile_values)
            ax_.multi_line(
                list(zip(probs, probs)),
                [(quant - err, quant + err) for quant, err in zip(quantile_values, x)],
            )
        else:
            ax_.circle(probs, x)
            if extra_methods:
                mean_mcse_i = mean_mcse[var_name].sel(**selection).values.item()
                sd_mcse_i = sd_mcse[var_name].sel(**selection).values.item()
                hline_mean = Span(
                    location=mean_mcse_i,
                    dimension="width",
                    line_color="black",
                    line_width=extra_kwargs["linewidth"] * 2,
                    line_alpha=extra_kwargs["alpha"],
                )

                ax_.renderers.append(hline_mean)

                hline_sd = Span(
                    location=sd_mcse_i,
                    dimension="width",
                    line_color="black",
                    line_width=extra_kwargs["linewidth"],
                    line_alpha=extra_kwargs["alpha"],
                )

                ax_.renderers.append(hline_sd)

        if rug:
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

            mask = idata.sample_stats[rug_kind].values.flatten()
            values = rankdata(values)[mask]
            if errorbar:
                rug_x, rug_y = (
                    values / (len(mask) - 1),
                    np.full_like(
                        values,
                        min(
                            0,
                            min(quantile_values)
                            - (max(quantile_values) - min(quantile_values)) * 0.05,
                        ),
                    ),
                )

                hline = Span(
                    location=min(
                        0,
                        min(quantile_values) - (max(quantile_values) - min(quantile_values)) * 0.05,
                    ),
                    dimension="width",
                    line_color="black",
                    line_width=_linewidth,
                    line_alpha=0.7,
                )

            else:
                rug_x, rug_y = (
                    values / (len(mask) - 1),
                    np.full_like(values, 0,),
                )

                hline = Span(
                    location=0,
                    dimension="width",
                    line_color="black",
                    line_width=_linewidth,
                    line_alpha=0.7,
                )

            ax_.renderers.append(hline)

            glyph = Dash(x="rug_x", y="rug_y", **_rug_kwargs)
            cds_rug = ColumnDataSource({"rug_x": np.asarray(rug_x), "rug_y": np.asarray(rug_y)})
            ax_.add_glyph(cds_rug, glyph)

        title = Title()
        title.text = make_label(var_name, selection)
        ax_.title = title

        ax_.xaxis.axis_label = "Quantile"
        ax_.yaxis.axis_label = (
            r"Value $\pm$ MCSE for quantiles" if errorbar else "MCSE for quantiles"
        )

        if not errorbar:
            ax_.y_range._property_values["start"] = -0.05  # pylint: disable=protected-access
            ax_.y_range._property_values["end"] = 1  # pylint: disable=protected-access

    if backend_show(show):
        grid = gridplot(ax.tolist(), toolbar_location="above")
        bkp.show(grid)

    return ax
