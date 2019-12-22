"""Matplotlib forestplot."""
from collections import defaultdict, OrderedDict
from itertools import tee

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np

from . import backend_kwarg_defaults, backend_show
from ....stats import hpd
from ....stats.diagnostics import _ess, _rhat
from ....stats.stats_utils import histogram
from ...plot_utils import _scale_fig_size, xarray_var_iter, make_label, get_bins
from ...kdeplot import _fast_kde
from ....utils import conditional_jit


def pairwise(iterable):
    """From itertools cookbook. [a, b, c, ...] -> (a, b), (b, c), ..."""
    first, second = tee(iterable)
    next(second, None)
    return zip(first, second)


def plot_forest(
    ax,
    datasets,
    var_names,
    model_names,
    combined,
    colors,
    figsize,
    width_ratios,
    linewidth,
    markersize,
    kind,
    ncols,
    credible_interval,
    quartiles,
    rope,
    ridgeplot_overlap,
    ridgeplot_alpha,
    ridgeplot_kind,
    textsize,
    ess,
    r_hat,
    backend_kwargs,
    show,
):
    """Matplotlib forest plot."""
    plot_handler = PlotHandler(
        datasets, var_names=var_names, model_names=model_names, combined=combined, colors=colors
    )

    if figsize is None:
        figsize = (min(12, sum(width_ratios) * 2), plot_handler.fig_height())

    (figsize, _, titlesize, xt_labelsize, auto_linewidth, auto_markersize) = _scale_fig_size(
        figsize, textsize, 1.1, 1
    )

    if linewidth is None:
        linewidth = auto_linewidth

    if markersize is None:
        markersize = auto_markersize

    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    if ax is None:
        _, axes = plt.subplots(
            nrows=1,
            ncols=ncols,
            figsize=figsize,
            gridspec_kw={"width_ratios": width_ratios},
            sharey=True,
            **backend_kwargs,
        )
    else:
        axes = ax

    axes = np.atleast_1d(axes)
    if kind == "forestplot":
        plot_handler.forestplot(
            credible_interval,
            quartiles,
            xt_labelsize,
            titlesize,
            linewidth,
            markersize,
            axes[0],
            rope,
        )
    elif kind == "ridgeplot":
        plot_handler.ridgeplot(
            ridgeplot_overlap, linewidth, ridgeplot_alpha, ridgeplot_kind, axes[0]
        )
    else:
        raise TypeError(
            "Argument 'kind' must be one of 'forestplot' or "
            "'ridgeplot' (you provided {})".format(kind)
        )

    idx = 1
    if ess:
        plot_handler.plot_neff(axes[idx], xt_labelsize, titlesize, markersize)
        idx += 1

    if r_hat:
        plot_handler.plot_rhat(axes[idx], xt_labelsize, titlesize, markersize)
        idx += 1

    for ax_ in axes:
        if kind == "ridgeplot":
            ax_.grid(False)
        else:
            ax_.grid(False, axis="y")
        # Remove ticklines on y-axes
        ax_.tick_params(axis="y", left=False, right=False)

        for loc, spine in ax_.spines.items():
            if loc in ["left", "right"]:
                spine.set_visible(False)

        if len(plot_handler.data) > 1:
            plot_handler.make_bands(ax_)

    labels, ticks = plot_handler.labels_and_ticks()
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels(labels)
    all_plotters = list(plot_handler.plotters.values())
    y_max = plot_handler.y_max() - all_plotters[-1].group_offset
    if kind == "ridgeplot":  # space at the top
        y_max += ridgeplot_overlap
    axes[0].set_ylim(-all_plotters[0].group_offset, y_max)

    if backend_show(show):
        plt.show()

    return axes


class PlotHandler:
    """Class to handle logic from ForestPlot."""

    # pylint: disable=inconsistent-return-statements

    def __init__(self, datasets, var_names, model_names, combined, colors):
        self.data = datasets

        if model_names is None:
            if len(self.data) > 1:
                model_names = ["Model {}".format(idx) for idx, _ in enumerate(self.data)]
            else:
                model_names = [""]
        elif len(model_names) != len(self.data):
            raise ValueError("The number of model names does not match the number of models")

        self.model_names = list(reversed(model_names))  # y-values are upside down

        if var_names is None:
            if len(self.data) > 1:
                self.var_names = list(
                    set().union(*[OrderedDict(datum.data_vars) for datum in self.data])
                )
            else:
                self.var_names = list(
                    reversed(*[OrderedDict(datum.data_vars) for datum in self.data])
                )
        else:
            self.var_names = list(reversed(var_names))  # y-values are upside down

        self.combined = combined

        if colors == "cycle":
            colors = ["C{}".format(idx) for idx, _ in enumerate(self.data)]
        elif isinstance(colors, str):
            colors = [colors for _ in self.data]

        self.colors = list(reversed(colors))  # y-values are upside down

        self.plotters = self.make_plotters()

    def make_plotters(self):
        """Initialize an object for each variable to be plotted."""
        plotters, y = {}, 0
        for var_name in self.var_names:
            plotters[var_name] = VarHandler(
                var_name,
                self.data,
                y,
                model_names=self.model_names,
                combined=self.combined,
                colors=self.colors,
            )
            y = plotters[var_name].y_max()
        return plotters

    def labels_and_ticks(self):
        """Collect labels and ticks from plotters."""
        val = self.plotters.values()

        @conditional_jit(forceobj=True)
        def label_idxs():
            labels, idxs = [], []
            for plotter in val:
                sub_labels, sub_idxs, _, _ = plotter.labels_ticks_and_vals()
                labels.append(sub_labels)
                idxs.append(sub_idxs)
            return np.concatenate(labels), np.concatenate(idxs)

        return label_idxs()

    def display_multiple_ropes(self, rope, ax, y, linewidth, rope_var):
        """Display ROPE when more than one interval is provided."""
        vals = dict(rope[rope_var][0])["rope"]
        ax.plot(
            vals,
            (y + 0.05, y + 0.05),
            lw=linewidth * 2,
            color="C2",
            solid_capstyle="round",
            zorder=0,
            alpha=0.7,
        )
        return ax

    def ridgeplot(self, mult, linewidth, alpha, ridgeplot_kind, ax):
        """Draw ridgeplot for each plotter.

        Parameters
        ----------
        mult : float
            How much to multiply height by. Set this to greater than 1 to have some overlap.
        linewidth : float
            Width of line on border of ridges
        alpha : float
            Transparency of ridges
        kind : string
            By default ("auto") continuous variables are plotted using KDEs and discrete ones using
            histograms. To override this use "hist" to plot histograms and "density" for KDEs
        ax : Axes
            Axes to draw on
        """
        if alpha is None:
            alpha = 1.0
        zorder = 0
        for plotter in self.plotters.values():
            for x, y_min, y_max, color in plotter.ridgeplot(mult, ridgeplot_kind):
                if alpha == 0:
                    border = color
                    facecolor = "None"
                else:
                    border = "k"
                    facecolor = to_rgba(color, alpha)
                if x.dtype.kind == "i":
                    ax.bar(
                        x,
                        y_max - y_min,
                        bottom=y_min,
                        linewidth=linewidth,
                        ec=border,
                        fc=facecolor,
                        zorder=zorder,
                    )
                else:
                    ax.plot(x, y_max, "-", linewidth=linewidth, color=border, zorder=zorder)
                    ax.plot(x, y_min, "-", linewidth=linewidth, color=border, zorder=zorder)
                    ax.fill_between(x, y_min, y_max, alpha=alpha, color=color, zorder=zorder)
                zorder -= 1
        return ax

    def forestplot(
        self, credible_interval, quartiles, xt_labelsize, titlesize, linewidth, markersize, ax, rope
    ):
        """Draw forestplot for each plotter.

        Parameters
        ----------
        credible_interval : float
            How wide each line should be
        quartiles : bool
            Whether to mark quartiles
        xt_textsize : float
            Size of tick text
        titlesize : float
            Size of title text
        linewidth : float
            Width of forestplot line
        markersize : float
            Size of marker in center of forestplot line
        ax : Axes
            Axes to draw on
        """
        # Quantiles to be calculated
        endpoint = 100 * (1 - credible_interval) / 2
        if quartiles:
            qlist = [endpoint, 25, 50, 75, 100 - endpoint]
        else:
            qlist = [endpoint, 50, 100 - endpoint]

        for plotter in self.plotters.values():
            for y, rope_var, values, color in plotter.treeplot(qlist, credible_interval):
                if isinstance(rope, dict):
                    self.display_multiple_ropes(rope, ax, y, linewidth, rope_var)

                mid = len(values) // 2
                param_iter = zip(
                    np.linspace(2 * linewidth, linewidth, mid, endpoint=True)[-1::-1], range(mid)
                )
                for width, j in param_iter:
                    ax.hlines(y, values[j], values[-(j + 1)], linewidth=width, color=color)
                ax.plot(
                    values[mid],
                    y,
                    "o",
                    mfc=ax.get_facecolor(),
                    markersize=markersize * 0.75,
                    color=color,
                )
        ax.tick_params(labelsize=xt_labelsize)
        ax.set_title(
            "{:.1%} Credible Interval".format(credible_interval), fontsize=titlesize, wrap=True
        )
        if rope is None or isinstance(rope, dict):
            return
        elif len(rope) == 2:
            ax.axvspan(rope[0], rope[1], 0, self.y_max(), color="C2", alpha=0.5)
        else:
            raise ValueError(
                "Argument `rope` must be None, a dictionary like"
                '{"var_name": {"rope": (lo, hi)}}, or an '
                "iterable of length 2"
            )
        return ax

    def plot_neff(self, ax, xt_labelsize, titlesize, markersize):
        """Draw effective n for each plotter."""
        for plotter in self.plotters.values():
            for y, ess, color in plotter.ess():
                if ess is not None:
                    ax.plot(
                        ess,
                        y,
                        "o",
                        color=color,
                        clip_on=False,
                        markersize=markersize,
                        markeredgecolor="k",
                    )
        ax.set_xlim(left=0)
        ax.set_title("ess", fontsize=titlesize, wrap=True)
        ax.tick_params(labelsize=xt_labelsize)
        return ax

    def plot_rhat(self, ax, xt_labelsize, titlesize, markersize):
        """Draw r-hat for each plotter."""
        for plotter in self.plotters.values():
            for y, r_hat, color in plotter.r_hat():
                if r_hat is not None:
                    ax.plot(r_hat, y, "o", color=color, markersize=markersize, markeredgecolor="k")
        ax.set_xlim(left=0.9, right=2.1)
        ax.set_xticks([1, 2])
        ax.tick_params(labelsize=xt_labelsize)
        ax.set_title("r_hat", fontsize=titlesize, wrap=True)
        return ax

    def make_bands(self, ax):
        """Draw shaded horizontal bands for each plotter."""
        y_vals, y_prev, is_zero = [0], None, False
        prev_color_index = 0
        for plotter in self.plotters.values():
            for y, *_, color in plotter.iterator():
                if self.colors.index(color) < prev_color_index:
                    if not is_zero and y_prev is not None:
                        y_vals.append((y + y_prev) * 0.5)
                        is_zero = True
                else:
                    is_zero = False
                prev_color_index = self.colors.index(color)
                y_prev = y

        offset = plotter.group_offset  # pylint: disable=undefined-loop-variable

        y_vals.append(y_prev + offset)
        for idx, (y_start, y_stop) in enumerate(pairwise(y_vals)):
            ax.axhspan(y_start, y_stop, color="k", alpha=0.1 * (idx % 2))
        return ax

    def fig_height(self):
        """Figure out the height of this plot."""
        # hand-tuned
        return (
            4
            + len(self.data) * len(self.var_names)
            - 1
            + 0.1 * sum(1 for j in self.plotters.values() for _ in j.iterator())
        )

    def y_max(self):
        """Get maximum y value for the plot."""
        return max(p.y_max() for p in self.plotters.values())


class VarHandler:
    """Handle individual variable logic."""

    def __init__(self, var_name, data, y_start, model_names, combined, colors):
        self.var_name = var_name
        self.data = data
        self.y_start = y_start
        self.model_names = model_names
        self.combined = combined
        self.colors = colors
        self.model_color = dict(zip(self.model_names, self.colors))
        max_chains = max(datum.chain.max().values for datum in data)
        self.chain_offset = len(data) * 0.45 / max(1, max_chains)
        self.var_offset = 1.5 * self.chain_offset
        self.group_offset = 2 * self.var_offset

    def iterator(self):
        """Iterate over models and chains for each variable."""
        if self.combined:
            grouped_data = [[(0, datum)] for datum in self.data]
            skip_dims = {"chain"}
        else:
            grouped_data = [datum.groupby("chain") for datum in self.data]
            skip_dims = set()

        label_dict = OrderedDict()
        for name, grouped_datum in zip(self.model_names, grouped_data):
            for _, sub_data in grouped_datum:
                datum_iter = xarray_var_iter(
                    sub_data,
                    var_names=[self.var_name],
                    skip_dims=skip_dims,
                    reverse_selections=True,
                )
                for _, selection, values in datum_iter:
                    label = make_label(self.var_name, selection, position="beside")
                    if label not in label_dict:
                        label_dict[label] = OrderedDict()
                    if name not in label_dict[label]:
                        label_dict[label][name] = []
                    label_dict[label][name].append(values)

        y = self.y_start
        for label, model_data in label_dict.items():
            for model_name, value_list in model_data.items():
                if model_name:
                    row_label = "{}: {}".format(model_name, label)
                else:
                    row_label = label
                for values in value_list:
                    yield y, row_label, label, values, self.model_color[model_name]
                    y += self.chain_offset
                y += self.var_offset
            y += self.group_offset

    def labels_ticks_and_vals(self):
        """Get labels, ticks, values, and colors for the variable."""
        y_ticks = defaultdict(list)
        for y, label, _, vals, color in self.iterator():
            y_ticks[label].append((y, vals, color))
        labels, ticks, vals, colors = [], [], [], []
        for label, data in y_ticks.items():
            labels.append(label)
            ticks.append(np.mean([j[0] for j in data]))
            vals.append(np.vstack([j[1] for j in data]))
            colors.append(data[0][2])  # the colors are all the same
        return labels, ticks, vals, colors

    def treeplot(self, qlist, credible_interval):
        """Get data for each treeplot for the variable."""
        for y, _, label, values, color in self.iterator():
            ntiles = np.percentile(values.flatten(), qlist)
            ntiles[0], ntiles[-1] = hpd(values.flatten(), credible_interval, multimodal=False)
            yield y, label, ntiles, color

    def ridgeplot(self, mult, ridgeplot_kind):
        """Get data for each ridgeplot for the variable."""
        xvals, yvals, pdfs, colors = [], [], [], []
        for y, *_, values, color in self.iterator():
            yvals.append(y)
            colors.append(color)
            values = values.flatten()
            values = values[np.isfinite(values)]

            if ridgeplot_kind == "auto":
                kind = "hist" if np.all(np.mod(values, 1) == 0) else "density"
            else:
                kind = ridgeplot_kind

            if kind == "hist":
                bins = get_bins(values)
                _, density, x = histogram(values, bins=bins)
                x = x[:-1]
            elif kind == "density":
                density, lower, upper = _fast_kde(values)
                x = np.linspace(lower, upper, len(density))

            xvals.append(x)
            pdfs.append(density)

        scaling = max(np.max(j) for j in pdfs)
        for y, x, pdf, color in zip(yvals, xvals, pdfs, colors):
            y = y * np.ones_like(x)
            yield x, y, mult * pdf / scaling + y, color

    def ess(self):
        """Get effective n data for the variable."""
        _, y_vals, values, colors = self.labels_ticks_and_vals()
        for y, value, color in zip(y_vals, values, colors):
            yield y, _ess(value), color

    def r_hat(self):
        """Get rhat data for the variable."""
        _, y_vals, values, colors = self.labels_ticks_and_vals()
        for y, value, color in zip(y_vals, values, colors):
            if value.ndim != 2 or value.shape[0] < 2:
                yield y, None, color
            else:
                yield y, _rhat(value), color

    def y_max(self):
        """Get max y value for the variable."""
        end_y = max(y for y, *_ in self.iterator())

        if self.combined:
            end_y += self.group_offset

        return end_y + 2 * self.group_offset
