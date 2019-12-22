"""Bokeh Compareplot."""
import bokeh.plotting as bkp
from bokeh.models import Span

from . import backend_kwarg_defaults, backend_show


def plot_compare(
    ax,
    comp_df,
    figsize,
    plot_ic_diff,
    plot_standard_error,
    insample_dev,
    yticks_pos,
    yticks_labels,
    line_width,
    plot_kwargs,
    information_criterion,
    step,
    backend_kwargs,
    show,
):
    """Bokeh compareplot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(
            ("tools", "plot.bokeh.tools"),
            ("output_backend", "plot.bokeh.output_backend"),
            ("dpi", "plot.bokeh.figure.dpi"),
        ),
        **backend_kwargs,
    }
    dpi = backend_kwargs.pop("dpi")

    if ax is None:
        ax = bkp.figure(width=figsize[0] * dpi, height=figsize[1] * dpi, **backend_kwargs)

    yticks_pos = list(yticks_pos)

    if plot_ic_diff:
        yticks_labels[0] = comp_df.index[0]
        yticks_labels[2::2] = comp_df.index[1:]

        ax.yaxis.ticker = yticks_pos
        ax.yaxis.major_label_overrides = {
            dtype(key): value
            for key, value in zip(yticks_pos, yticks_labels)
            for dtype in (int, float)
            if (dtype(key) - key == 0)
        }

        # create the coordinates for the errorbars
        err_xs = []
        err_ys = []

        for x, y, xerr in zip(
            comp_df[information_criterion].iloc[1:], yticks_pos[1::2], comp_df.dse[1:]
        ):
            err_xs.append((x - xerr, x + xerr))
            err_ys.append((y, y))

        # plot them
        ax.triangle(
            comp_df[information_criterion].iloc[1:],
            yticks_pos[1::2],
            line_color=plot_kwargs.get("color_dse", "grey"),
            fill_color=plot_kwargs.get("color_dse", "grey"),
            line_width=2,
            size=6,
        )
        ax.multi_line(err_xs, err_ys, line_color=plot_kwargs.get("color_dse", "grey"))

    else:
        yticks_labels = comp_df.index
        ax.yaxis.ticker = yticks_pos[::2]
        ax.yaxis.major_label_overrides = {
            key: value for key, value in zip(yticks_pos[::2], yticks_labels)
        }

    ax.circle(
        comp_df[information_criterion],
        yticks_pos[::2],
        line_color=plot_kwargs.get("color_ic", "black"),
        fill_color=None,
        line_width=2,
        size=6,
    )

    if plot_standard_error:
        # create the coordinates for the errorbars
        err_xs = []
        err_ys = []

        for x, y, xerr in zip(comp_df[information_criterion], yticks_pos[::2], comp_df.se):
            err_xs.append((x - xerr, x + xerr))
            err_ys.append((y, y))

        # plot them
        ax.multi_line(err_xs, err_ys, line_color=plot_kwargs.get("color_ic", "black"))

    if insample_dev:
        ax.circle(
            comp_df[information_criterion] - (2 * comp_df["p_" + information_criterion]),
            yticks_pos[::2],
            line_color=plot_kwargs.get("color_insample_dev", "black"),
            fill_color=plot_kwargs.get("color_insample_dev", "black"),
            line_width=2,
            size=6,
        )

    vline = Span(
        location=comp_df[information_criterion].iloc[0],
        dimension="height",
        line_color=plot_kwargs.get("color_ls_min_ic", "grey"),
        line_width=line_width,
        line_dash=plot_kwargs.get("ls_min_ic", "dashed"),
    )

    ax.renderers.append(vline)

    scale_col = information_criterion + "_scale"
    if scale_col in comp_df:
        scale = comp_df[scale_col].iloc[0].capitalize()
    else:
        scale = "Deviance"
    ax.xaxis.axis_label = scale
    ax.y_range._property_values["start"] = -1 + step  # pylint: disable=protected-access
    ax.y_range._property_values["end"] = 0 - step  # pylint: disable=protected-access

    if backend_show(show):
        bkp.show(ax, toolbar_location="above")

    return ax
