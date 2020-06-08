"""Bokeh energyplot."""
import bokeh.plotting as bkp
from bokeh.models import Label
from bokeh.models.annotations import Legend

from . import backend_kwarg_defaults
from .. import show_layout
from .distplot import _histplot_bokeh_op
from ...kdeplot import plot_kde
from ....stats import bfmi as e_bfmi


def plot_energy(
    ax,
    series,
    energy,
    kind,
    bfmi,
    figsize,
    line_width,
    fill_kwargs,
    plot_kwargs,
    bw,
    legend,
    backend_kwargs,
    show,
):
    """Bokeh energy plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(("dpi", "plot.bokeh.figure.dpi"),),
        **backend_kwargs,
    }
    dpi = backend_kwargs.pop("dpi")
    if ax is None:
        backend_kwargs.setdefault("width", int(figsize[0] * dpi))
        backend_kwargs.setdefault("height", int(figsize[1] * dpi))
        ax = bkp.figure(**backend_kwargs)

    labels = []
    if kind == "kde":
        for alpha, color, label, value in series:
            fill_kwargs["fill_alpha"] = alpha
            fill_kwargs["fill_color"] = color
            plot_kwargs["line_color"] = color
            plot_kwargs["line_alpha"] = alpha
            plot_kwargs.setdefault("line_width", line_width)
            _, glyph = plot_kde(
                value,
                bw=bw,
                label=label,
                fill_kwargs=fill_kwargs,
                plot_kwargs=plot_kwargs,
                ax=ax,
                legend=legend,
                backend="bokeh",
                backend_kwargs={},
                show=False,
                return_glyph=True,
            )
            labels.append((label, glyph,))

    elif kind in {"hist", "histogram"}:
        hist_kwargs = plot_kwargs.copy()
        hist_kwargs.update(**fill_kwargs)

        for alpha, color, label, value in series:
            hist_kwargs["fill_alpha"] = alpha
            hist_kwargs["fill_color"] = color
            hist_kwargs["line_color"] = None
            hist_kwargs["line_alpha"] = alpha
            _histplot_bokeh_op(
                value.flatten(), values2=None, rotated=False, ax=ax, hist_kwargs=hist_kwargs
            )

    else:
        raise ValueError("Plot type {} not recognized.".format(kind))

    if bfmi:
        for idx, val in enumerate(e_bfmi(energy)):
            bfmi_info = Label(
                x=int(figsize[0] * dpi * 0.58),
                y=int(figsize[1] * dpi * 0.73) - 20 * idx,
                x_units="screen",
                y_units="screen",
                text="chain {:>2} BFMI = {:.2f}".format(idx, val),
                render_mode="css",
                border_line_color=None,
                border_line_alpha=0.0,
                background_fill_color="white",
                background_fill_alpha=1.0,
            )

            ax.add_layout(bfmi_info)

    if legend and label is not None:
        legend = Legend(items=labels, location="center_right", orientation="horizontal",)
        ax.add_layout(legend, "above")
        ax.legend.click_policy = "hide"

    show_layout(ax, show)

    return ax
