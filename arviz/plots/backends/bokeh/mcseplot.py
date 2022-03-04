"""Bokeh mcseplot."""
import numpy as np
from bokeh.models import ColumnDataSource, Span
from bokeh.models.glyphs import Scatter
from bokeh.models.annotations import Title
from scipy.stats import rankdata

from ....stats.stats_utils import quantile as _quantile
from ...plot_utils import _scale_fig_size
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid


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
    kwargs,  # pylint: disable=unused-argument
    extra_methods,
    mean_mcse,
    sd_mcse,
    textsize,
    labeller,
    text_kwargs,  # pylint: disable=unused-argument
    rug_kwargs,
    extra_kwargs,
    idata,
    rug_kind,
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

    (figsize, *_, _linewidth, _markersize) = _scale_fig_size(figsize, textsize, rows, cols)

    extra_kwargs = {} if extra_kwargs is None else extra_kwargs
    extra_kwargs.setdefault("linewidth", _linewidth / 2)
    extra_kwargs.setdefault("color", "black")
    extra_kwargs.setdefault("alpha", 0.5)

    if ax is None:
        ax = create_axes_grid(
            length_plotters,
            rows,
            cols,
            figsize=figsize,
            backend_kwargs=backend_kwargs,
        )
    else:
        ax = np.atleast_2d(ax)

    for (var_name, selection, isel, x), ax_ in zip(
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
                    line_color=extra_kwargs["color"],
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
                raise ValueError(f"InferenceData does not contain {rug_kind} data")
            rug_kwargs.setdefault("space", 0.1)

            _rug_kwargs = {}
            _rug_kwargs.setdefault("size", 8)
            _rug_kwargs.setdefault("line_color", rug_kwargs.get("line_color", "black"))
            _rug_kwargs.setdefault("line_width", 1)
            _rug_kwargs.setdefault("line_alpha", 0.35)
            _rug_kwargs.setdefault("angle", np.pi / 2)

            mask = idata.sample_stats[rug_kind].values.flatten()
            values = rankdata(values, method="average")[mask]
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
                    np.full_like(
                        values,
                        0,
                    ),
                )

                hline = Span(
                    location=0,
                    dimension="width",
                    line_color="black",
                    line_width=_linewidth,
                    line_alpha=0.7,
                )

            ax_.renderers.append(hline)

            glyph = Scatter(x="rug_x", y="rug_y", marker="dash", **_rug_kwargs)
            cds_rug = ColumnDataSource({"rug_x": np.asarray(rug_x), "rug_y": np.asarray(rug_y)})
            ax_.add_glyph(cds_rug, glyph)

        title = Title()
        title.text = labeller.make_label_vert(var_name, selection, isel)
        ax_.title = title

        ax_.xaxis.axis_label = "Quantile"
        ax_.yaxis.axis_label = (
            r"Value $\pm$ MCSE for quantiles" if errorbar else "MCSE for quantiles"
        )

        if not errorbar:
            ax_.y_range._property_values["start"] = -0.05  # pylint: disable=protected-access
            ax_.y_range._property_values["end"] = 1  # pylint: disable=protected-access

    show_layout(ax, show)

    return ax
