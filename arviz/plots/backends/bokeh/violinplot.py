"""Bokeh Violinplot."""
import bokeh.plotting as bkp
import numpy as np
from bokeh.layouts import gridplot
from bokeh.models.annotations import Title

from . import backend_kwarg_defaults, backend_show
from ...kdeplot import _fast_kde
from ...plot_utils import get_bins, make_label, _create_axes_grid
from ....stats import hpd
from ....stats.stats_utils import histogram


def plot_violin(
    ax,
    plotters,
    figsize,
    rows,
    cols,
    sharey,
    kwargs_shade,
    shade,
    bw,
    credible_interval,
    linewidth,
    quartiles,
    backend_kwargs,
    show,
):
    """Bokeh violin plot."""
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
            sharey=sharey,
            figsize=figsize,
            squeeze=False,
            backend="bokeh",
            backend_kwargs=backend_kwargs,
        )
    else:
        ax = np.atleast_2d(ax)

    for (var_name, selection, x), ax_ in zip(
        plotters, (item for item in ax.flatten() if item is not None)
    ):
        val = x.flatten()
        if val[0].dtype.kind == "i":
            cat_hist(val, shade, ax_, **kwargs_shade)
        else:
            _violinplot(val, shade, bw, ax_, **kwargs_shade)

        per = np.percentile(val, [25, 75, 50])
        hpd_intervals = hpd(val, credible_interval, multimodal=False)

        if quartiles:
            ax_.line([0, 0], per[:2], line_width=linewidth * 3, line_color="black")
        ax_.line([0, 0], hpd_intervals, line_width=linewidth, line_color="black")
        ax_.circle(0, per[-1])

        _title = Title()
        _title.text = make_label(var_name, selection)
        ax_.title = _title
        ax_.xaxis.major_tick_line_color = None
        ax_.xaxis.minor_tick_line_color = None
        ax_.xaxis.major_label_text_font_size = "0pt"

    if backend_show(show):
        grid = gridplot(ax.tolist(), toolbar_location="above")
        bkp.show(grid)

    return ax


def _violinplot(val, shade, bw, ax, **kwargs_shade):
    """Auxiliary function to plot violinplots."""
    density, low_b, up_b = _fast_kde(val, bw=bw)
    x = np.linspace(low_b, up_b, len(density))

    x = np.concatenate([x, x[::-1]])
    density = np.concatenate([-density, density[::-1]])

    ax.patch(density, x, fill_alpha=shade, line_width=0, **kwargs_shade)


def cat_hist(val, shade, ax, **kwargs_shade):
    """Auxiliary function to plot discrete-violinplots."""
    bins = get_bins(val)
    _, binned_d, _ = histogram(val, bins=bins)

    bin_edges = np.linspace(np.min(val), np.max(val), len(bins))
    centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
    heights = np.diff(bin_edges)

    lefts = -0.5 * binned_d

    ax.hbar(
        y=centers,
        left=lefts,
        right=-lefts,
        height=heights,
        fill_alpha=shade,
        line_alpha=shade,
        line_color=None,
        **kwargs_shade
    )
