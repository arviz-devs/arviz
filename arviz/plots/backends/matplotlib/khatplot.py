"""Matplotlib khatplot."""
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from . import backend_kwarg_defaults, backend_show
from ...plot_utils import set_xticklabels
from ....stats.stats_utils import histogram


def plot_khat(
    hover_label,
    hover_format,
    ax,
    figsize,
    xdata,
    khats,
    rgba_c,
    kwargs,
    annotate,
    coord_labels,
    ax_labelsize,
    xt_labelsize,
    show_bins,
    linewidth,
    hlines_kwargs,
    xlabels,
    legend,
    color_mapping,
    cmap,
    color,
    n_data_points,
    bin_format,
    backend_kwargs,
    show,
):
    """Matplotlib khat plot."""
    if hover_label and mpl.get_backend() not in mpl.rcsetup.interactive_bk:
        hover_label = False
        warnings.warn(
            "hover labels are only available with interactive backends. To switch to an "
            "interactive backend from ipython or jupyter, use `%matplotlib` there should be "
            "no need to restart the kernel. For other cases, see "
            "https://matplotlib.org/3.1.0/tutorials/introductory/usage.html#backends",
            UserWarning,
        )

    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(constrained_layout=(not xlabels)),
        **backend_kwargs,
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, **backend_kwargs)
    else:
        fig = ax.get_figure()

    sc_plot = ax.scatter(xdata, khats, c=rgba_c, **kwargs)

    if annotate:
        idxs = xdata[khats > 1]
        for idx in idxs:
            ax.text(
                idx,
                khats[idx],
                coord_labels[idx],
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=0.8 * xt_labelsize,
            )

    xmin, xmax = ax.get_xlim()
    if show_bins:
        xmax += n_data_points / 12
    ylims1 = ax.get_ylim()
    ax.hlines([0, 0.5, 0.7, 1], xmin=xmin, xmax=xmax, linewidth=linewidth, **hlines_kwargs)
    ylims2 = ax.get_ylim()
    ymin = min(ylims1[0], ylims2[0])
    ymax = min(ylims1[1], ylims2[1])
    if show_bins:
        bin_edges = np.array([ymin, 0.5, 0.7, 1, ymax])
        bin_edges = bin_edges[(bin_edges >= ymin) & (bin_edges <= ymax)]
        hist, _, _ = histogram(khats, bin_edges)
        for idx, count in enumerate(hist):
            ax.text(
                (n_data_points - 1 + xmax) / 2,
                np.mean(bin_edges[idx : idx + 2]),
                bin_format.format(count, count / n_data_points * 100),
                horizontalalignment="center",
                verticalalignment="center",
            )
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)

    ax.set_xlabel("Data Point", fontsize=ax_labelsize)
    ax.set_ylabel(r"Shape parameter k", fontsize=ax_labelsize)
    ax.tick_params(labelsize=xt_labelsize)
    if xlabels:
        set_xticklabels(ax, coord_labels)
        fig.autofmt_xdate()
        fig.tight_layout()
    if legend:
        ncols = len(color_mapping) // 6 + 1
        for label, float_color in color_mapping.items():
            ax.scatter([], [], c=[cmap(float_color)], label=label, **kwargs)
        ax.legend(ncol=ncols, title=color)

    if hover_label and mpl.get_backend() in mpl.rcsetup.interactive_bk:
        _make_hover_annotation(fig, ax, sc_plot, coord_labels, rgba_c, hover_format)

    if backend_show(show):
        plt.show()

    return ax


def _make_hover_annotation(fig, ax, sc_plot, coord_labels, rgba_c, hover_format):
    """Show data point label when hovering over it with mouse."""
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(0, 0),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w", alpha=0.4),
        arrowprops=dict(arrowstyle="->"),
    )
    annot.set_visible(False)
    xmid = np.mean(ax.get_xlim())
    ymid = np.mean(ax.get_ylim())
    offset = 10

    def update_annot(ind):

        idx = ind["ind"][0]
        pos = sc_plot.get_offsets()[idx]
        annot_text = hover_format.format(idx, coord_labels[idx])
        annot.xy = pos
        annot.set_position(
            (-offset if pos[0] > xmid else offset, -offset if pos[1] > ymid else offset)
        )
        annot.set_text(annot_text)
        annot.get_bbox_patch().set_facecolor(rgba_c[idx])
        annot.set_ha("right" if pos[0] > xmid else "left")
        annot.set_va("top" if pos[1] > ymid else "bottom")

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc_plot.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
