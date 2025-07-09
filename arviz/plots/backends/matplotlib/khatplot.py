"""Matplotlib khatplot."""

import warnings

import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba_array
from packaging import version

from ....stats.density_utils import histogram
from ...plot_utils import _scale_fig_size, color_from_dim, set_xticklabels, vectorized_to_hex
from . import backend_show, create_axes_grid, matplotlib_kwarg_dealiaser


def plot_khat(
    hover_label,
    hover_format,
    ax,
    figsize,
    xdata,
    khats,
    good_k,
    kwargs,
    threshold,
    coord_labels,
    show_hlines,
    show_bins,
    hlines_kwargs,
    xlabels,
    legend,
    color,
    dims,
    textsize,
    markersize,
    n_data_points,
    bin_format,
    backend_kwargs,
    show,
):
    """Matplotlib khat plot."""
    if version.parse(mpl.__version__) >= version.parse("3.9.0.dev0"):
        interactive_backends = mpl.backends.backend_registry.list_builtin(
            mpl.backends.BackendFilter.INTERACTIVE
        )
    else:
        interactive_backends = mpl.rcsetup.interactive_bk
    if hover_label and mpl.get_backend() not in interactive_backends:
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
        **backend_kwargs,
    }

    (figsize, ax_labelsize, _, xt_labelsize, linewidth, scaled_markersize) = _scale_fig_size(
        figsize, textsize
    )
    backend_kwargs.setdefault("figsize", figsize)
    backend_kwargs["squeeze"] = True

    if good_k is None:
        good_k = 0.7

    hlines_kwargs = matplotlib_kwarg_dealiaser(hlines_kwargs, "hlines")
    hlines_kwargs.setdefault("hlines", [0, good_k, 1])
    hlines_kwargs.setdefault("linestyle", [":", "-.", "--", "-"])
    hlines_kwargs.setdefault("alpha", 0.7)
    hlines_kwargs.setdefault("zorder", -1)
    hlines_kwargs.setdefault("color", "C1")
    hlines_kwargs["color"] = vectorized_to_hex(hlines_kwargs["color"])

    if markersize is None:
        markersize = scaled_markersize**2  # s in scatter plot mus be markersize square
        # for dots to have the same size

    kwargs = matplotlib_kwarg_dealiaser(kwargs, "scatter")
    kwargs.setdefault("s", markersize)
    kwargs.setdefault("marker", "+")

    c_kwarg = kwargs.get("c", None)

    if c_kwarg is None:
        color_mapping = None
        cmap = None
        if isinstance(color, str):
            if color in dims:
                colors, color_mapping = color_from_dim(khats, color)
                cmap_name = kwargs.get("cmap", plt.rcParams["image.cmap"])
                cmap = getattr(cm, cmap_name)
                rgba_c = cmap(colors)
            else:
                legend = False
                rgba_c = to_rgba_array(np.full(n_data_points, color))
        else:
            legend = False
            try:
                rgba_c = to_rgba_array(color)
            except ValueError:
                cmap_name = kwargs.get("cmap", plt.rcParams["image.cmap"])
                cmap = getattr(cm, cmap_name)
                norm_fun = kwargs.get("norm", mpl.colors.Normalize(color.min(), color.max()))
                rgba_c = cmap(norm_fun(color))

        khats = khats if isinstance(khats, np.ndarray) else khats.values.flatten()
        alphas = 0.5 + 0.2 * (khats > good_k) + 0.3 * (khats > 1)
        rgba_c[:, 3] = alphas
        rgba_c = vectorized_to_hex(rgba_c)
        kwargs["c"] = rgba_c
    else:
        if isinstance(c_kwarg, str):
            if c_kwarg in dims:
                colors, color_mapping = color_from_dim(khats, c_kwarg)
            else:
                legend = False
        else:
            legend = False

    if ax is None:
        fig, ax = create_axes_grid(
            1,
            backend_kwargs=backend_kwargs,
        )
    else:
        fig = ax.get_figure()

    sc_plot = ax.scatter(xdata, khats, **kwargs)

    if threshold is not None:
        idxs = xdata[khats > threshold]
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
    ylims2 = ax.get_ylim()
    ymin = min(ylims1[0], ylims2[0])
    ymax = min(ylims1[1], ylims2[1])

    if show_hlines:
        ax.hlines(
            hlines_kwargs.pop("hlines"), xmin=xmin, xmax=xmax, linewidth=linewidth, **hlines_kwargs
        )

    if show_bins:
        bin_edges = np.array([ymin, good_k, 1, ymax])
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

    ax.set_xlabel("Data Point", fontsize=ax_labelsize)
    ax.set_ylabel(r"Shape parameter k", fontsize=ax_labelsize)
    ax.tick_params(labelsize=xt_labelsize)
    if xlabels:
        set_xticklabels(ax, coord_labels)
        fig.autofmt_xdate()
        fig.tight_layout()
    if legend:
        kwargs.pop("c")
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
