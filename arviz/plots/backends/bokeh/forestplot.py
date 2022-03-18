# pylint: disable=all
"""Bokeh forestplot."""
from collections import OrderedDict, defaultdict
from itertools import cycle, tee

import bokeh.plotting as bkp
import matplotlib.pyplot as plt
import numpy as np
from bokeh.models import Band, ColumnDataSource, DataRange1d
from bokeh.models.annotations import Title, Legend
from bokeh.models.tickers import FixedTicker

from ....sel_utils import xarray_var_iter
from ....rcparams import rcParams
from ....stats import hdi
from ....stats.density_utils import get_bins, histogram, kde
from ....stats.diagnostics import _ess, _rhat
from ....utils import conditional_jit
from ...plot_utils import _scale_fig_size
from .. import show_layout
from . import backend_kwarg_defaults


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
    combine_dims,
    colors,
    figsize,
    width_ratios,
    linewidth,
    markersize,
    kind,
    ncols,
    hdi_prob,
    quartiles,
    rope,
    ridgeplot_overlap,
    ridgeplot_alpha,
    ridgeplot_kind,
    ridgeplot_truncate,
    ridgeplot_quantiles,
    textsize,
    legend,
    labeller,
    ess,
    r_hat,
    backend_config,
    backend_kwargs,
    show,
):
    """Bokeh forest plot."""
    plot_handler = PlotHandler(
        datasets,
        var_names=var_names,
        model_names=model_names,
        combined=combined,
        combine_dims=combine_dims,
        colors=colors,
        labeller=labeller,
    )

    if figsize is None:
        if kind == "ridgeplot":
            figsize = (min(14, sum(width_ratios) * 3), plot_handler.fig_height() * 3)
        else:
            figsize = (min(12, sum(width_ratios) * 2), plot_handler.fig_height())

    (figsize, _, _, _, auto_linewidth, auto_markersize) = _scale_fig_size(figsize, textsize, 1.1, 1)

    if linewidth is None:
        linewidth = auto_linewidth

    if markersize is None:
        markersize = auto_markersize

    if backend_config is None:
        backend_config = {}

    backend_config = {
        **backend_kwarg_defaults(
            ("bounds_x_range", "plot.bokeh.bounds_x_range"),
            ("bounds_y_range", "plot.bokeh.bounds_y_range"),
        ),
        **backend_config,
    }

    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(
            ("dpi", "plot.bokeh.figure.dpi"),
        ),
        **backend_kwargs,
    }
    dpi = backend_kwargs.pop("dpi")

    if ax is None:
        axes = []

        for i, width_r in zip(range(ncols), width_ratios):
            backend_kwargs_i = backend_kwargs.copy()
            backend_kwargs_i.setdefault("height", int(figsize[1] * dpi))
            backend_kwargs_i.setdefault(
                "width", int(figsize[0] * (width_r / sum(width_ratios)) * dpi * 1.25)
            )
            if i == 0:
                ax = bkp.figure(
                    **backend_kwargs_i,
                )
                backend_kwargs.setdefault("y_range", ax.y_range)
            else:
                ax = bkp.figure(**backend_kwargs_i)
            axes.append(ax)
    else:
        axes = ax

    axes = np.atleast_2d(axes)

    plotted = defaultdict(list)

    if kind == "forestplot":
        plot_handler.forestplot(
            hdi_prob,
            quartiles,
            linewidth,
            markersize,
            axes[0, 0],
            rope,
            plotted,
        )
    elif kind == "ridgeplot":
        plot_handler.ridgeplot(
            hdi_prob,
            ridgeplot_overlap,
            linewidth,
            markersize,
            ridgeplot_alpha,
            ridgeplot_kind,
            ridgeplot_truncate,
            ridgeplot_quantiles,
            axes[0, 0],
            plotted,
        )
    else:
        raise TypeError(
            "Argument 'kind' must be one of 'forestplot' or "
            "'ridgeplot' (you provided {})".format(kind)
        )

    idx = 1
    if ess:
        plotted_ess = defaultdict(list)
        plot_handler.plot_neff(axes[0, idx], markersize, plotted_ess)
        if legend:
            plot_handler.legend(axes[0, idx], plotted_ess)
        idx += 1

    if r_hat:
        plotted_r_hat = defaultdict(list)
        plot_handler.plot_rhat(axes[0, idx], markersize, plotted_r_hat)
        if legend:
            plot_handler.legend(axes[0, idx], plotted_r_hat)
        idx += 1

    all_plotters = list(plot_handler.plotters.values())
    y_max = plot_handler.y_max() - all_plotters[-1].group_offset
    if kind == "ridgeplot":  # space at the top
        y_max += ridgeplot_overlap

    for i, ax_ in enumerate(axes.ravel()):
        if kind == "ridgeplot":
            ax_.xgrid.grid_line_color = None
            ax_.ygrid.grid_line_color = None
        else:
            ax_.ygrid.grid_line_color = None

        if i != 0:
            ax_.yaxis.visible = False

        ax_.outline_line_color = None
        ax_.x_range = DataRange1d(bounds=backend_config["bounds_x_range"], min_interval=1)
        ax_.y_range = DataRange1d(bounds=backend_config["bounds_y_range"], min_interval=2)

        ax_.y_range._property_values["start"] = -all_plotters[  # pylint: disable=protected-access
            0
        ].group_offset
        ax_.y_range._property_values["end"] = y_max  # pylint: disable=protected-access

    labels, ticks = plot_handler.labels_and_ticks()
    ticks = [int(tick) if (tick).is_integer() else tick for tick in ticks]

    axes[0, 0].yaxis.ticker = FixedTicker(ticks=ticks)
    axes[0, 0].yaxis.major_label_overrides = dict(zip(map(str, ticks), map(str, labels)))

    if legend:
        plot_handler.legend(axes[0, 0], plotted)
    show_layout(axes, show)

    return axes


class PlotHandler:
    """Class to handle logic from ForestPlot."""

    # pylint: disable=inconsistent-return-statements

    def __init__(self, datasets, var_names, model_names, combined, combine_dims, colors, labeller):
        self.data = datasets

        if model_names is None:
            if len(self.data) > 1:
                model_names = [f"Model {idx}" for idx, _ in enumerate(self.data)]
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
        self.combine_dims = combine_dims

        if colors == "cycle":
            colors = [
                prop
                for _, prop in zip(
                    range(len(self.data)), cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
                )
            ]
        elif isinstance(colors, str):
            colors = [colors for _ in self.data]

        self.colors = list(reversed(colors))  # y-values are upside down
        self.labeller = labeller

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
                combine_dims=self.combine_dims,
                colors=self.colors,
                labeller=self.labeller,
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
                sub_labels, sub_idxs, _, _, _ = plotter.labels_ticks_and_vals()
                labels_to_idxs = defaultdict(list)
                for label, idx in zip(sub_labels, sub_idxs):
                    labels_to_idxs[label].append(idx)
                sub_idxs = []
                sub_labels = []
                for label, all_idx in labels_to_idxs.items():
                    sub_labels.append(label)
                    sub_idxs.append(np.mean([j for j in all_idx]))
                labels.append(sub_labels)
                idxs.append(sub_idxs)
            return np.concatenate(labels), np.concatenate(idxs)

        return label_idxs()

    def legend(self, ax, plotted):
        """Add interactive legend with colorcoded model info."""
        legend_it = []
        for (model_name, glyphs) in plotted.items():
            legend_it.append((model_name, glyphs))

        legend = Legend(items=legend_it, orientation="vertical", location="top_left")
        ax.add_layout(legend, "above")
        ax.legend.click_policy = "hide"

    def display_multiple_ropes(
        self, rope, ax, y, linewidth, var_name, selection, plotted, model_name
    ):
        """Display ROPE when more than one interval is provided."""
        for sel in rope.get(var_name, []):
            # pylint: disable=line-too-long
            if all(k in selection and selection[k] == v for k, v in sel.items() if k != "rope"):
                vals = sel["rope"]
                plotted[model_name].append(
                    ax.line(
                        vals,
                        (y + 0.05, y + 0.05),
                        line_width=linewidth * 2,
                        color=[
                            color
                            for _, color in zip(
                                range(3), cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
                            )
                        ][2],
                        line_alpha=0.7,
                    )
                )
                return ax

    def ridgeplot(
        self,
        hdi_prob,
        mult,
        linewidth,
        markersize,
        alpha,
        ridgeplot_kind,
        ridgeplot_truncate,
        ridgeplot_quantiles,
        ax,
        plotted,
    ):
        """Draw ridgeplot for each plotter.

        Parameters
        ----------
        hdi_prob : float
            Probability for the highest density interval.
        mult : float
            How much to multiply height by. Set this to greater than 1 to have some overlap.
        linewidth : float
            Width of line on border of ridges
        markersize : float
            Size of marker in center of forestplot line
        alpha : float
            Transparency of ridges
        ridgeplot_kind : string
            By default ("auto") continuous variables are plotted using KDEs and discrete ones using
            histograms. To override this use "hist" to plot histograms and "density" for KDEs
        ridgeplot_truncate: bool
            Whether to truncate densities according to the value of hdi_prop. Defaults to True
        ridgeplot_quantiles: list
            Quantiles in ascending order used to segment the KDE. Use [.25, .5, .75] for quartiles.
            Defaults to None.
        ax : Axes
            Axes to draw on
        plotted : dict
            Contains glyphs for each model
        """
        if alpha is None:
            alpha = 1.0
        for plotter in list(self.plotters.values())[::-1]:
            for x, y_min, y_max, hdi_, y_q, color, model_name in plotter.ridgeplot(
                hdi_prob, mult, ridgeplot_kind
            ):
                if alpha == 0:
                    border = color
                    facecolor = None
                else:
                    border = "black"
                    facecolor = color
                if x.dtype.kind == "i":
                    if ridgeplot_truncate:
                        y_max = y_max[(x >= hdi_[0]) & (x <= hdi_[1])]
                        x = x[(x >= hdi_[0]) & (x <= hdi_[1])]
                    else:
                        facecolor = color
                        alpha = [alpha if ci else 0 for ci in ((x >= hdi_[0]) & (x <= hdi_[1]))]
                    y_min = np.ones_like(x) * y_min
                    plotted[model_name].append(
                        ax.vbar(
                            x=x,
                            top=y_max - y_min,
                            bottom=y_min,
                            width=0.9,
                            line_color=border,
                            color=facecolor,
                            fill_alpha=alpha,
                        )
                    )
                else:
                    tr_x = x[(x >= hdi_[0]) & (x <= hdi_[1])]
                    tr_y_min = np.ones_like(tr_x) * y_min
                    tr_y_max = y_max[(x >= hdi_[0]) & (x <= hdi_[1])]
                    y_min = np.ones_like(x) * y_min
                    patch = ax.patch(
                        np.concatenate([tr_x, tr_x[::-1]]),
                        np.concatenate([tr_y_min, tr_y_max[::-1]]),
                        fill_color=color,
                        fill_alpha=alpha,
                        line_width=0,
                    )
                    patch.level = "overlay"
                    plotted[model_name].append(patch)
                    if ridgeplot_truncate:
                        plotted[model_name].append(
                            ax.line(
                                x, y_max, line_dash="solid", line_width=linewidth, line_color=border
                            )
                        )
                        plotted[model_name].append(
                            ax.line(
                                x, y_min, line_dash="solid", line_width=linewidth, line_color=border
                            )
                        )
                    else:
                        plotted[model_name].append(
                            ax.line(
                                tr_x,
                                tr_y_max,
                                line_dash="solid",
                                line_width=linewidth,
                                line_color=border,
                            )
                        )
                        plotted[model_name].append(
                            ax.line(
                                tr_x,
                                tr_y_min,
                                line_dash="solid",
                                line_width=linewidth,
                                line_color=border,
                            )
                        )
                if ridgeplot_quantiles is not None:
                    quantiles = [x[np.sum(y_q < quant)] for quant in ridgeplot_quantiles]
                    plotted[model_name].append(
                        ax.diamond(
                            quantiles,
                            np.ones_like(quantiles) * y_min[0],
                            line_color="black",
                            fill_color="black",
                            size=markersize,
                        )
                    )

        return ax

    def forestplot(self, hdi_prob, quartiles, linewidth, markersize, ax, rope, plotted):
        """Draw forestplot for each plotter.

        Parameters
        ----------
        hdi_prob : float
            Probability for the highest density interval. Width of each line.
        quartiles : bool
            Whether to mark quartiles
        linewidth : float
            Width of forestplot line
        markersize : float
            Size of marker in center of forestplot line
        ax : Axes
            Axes to draw on
        plotted : dict
            Contains glyphs for each model
        """
        if rope is None or isinstance(rope, dict):
            pass
        elif len(rope) == 2:
            cds = ColumnDataSource(
                {
                    "x": rope,
                    "lower": [-2 * self.y_max(), -2 * self.y_max()],
                    "upper": [self.y_max() * 2, self.y_max() * 2],
                }
            )

            band = Band(
                base="x",
                lower="lower",
                upper="upper",
                fill_color=[
                    color
                    for _, color in zip(
                        range(4), cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
                    )
                ][2],
                line_alpha=0.5,
                source=cds,
            )

            ax.renderers.append(band)
        else:
            raise ValueError(
                "Argument `rope` must be None, a dictionary like"
                '{"var_name": {"rope": (lo, hi)}}, or an '
                "iterable of length 2"
            )
        # Quantiles to be calculated
        endpoint = 100 * (1 - hdi_prob) / 2
        if quartiles:
            qlist = [endpoint, 25, 50, 75, 100 - endpoint]
        else:
            qlist = [endpoint, 50, 100 - endpoint]

        for plotter in self.plotters.values():
            for y, model_name, selection, values, color in plotter.treeplot(qlist, hdi_prob):
                if isinstance(rope, dict):
                    self.display_multiple_ropes(
                        rope, ax, y, linewidth, plotter.var_name, selection, plotted, model_name
                    )

                mid = len(values) // 2
                param_iter = zip(
                    np.linspace(2 * linewidth, linewidth, mid, endpoint=True)[-1::-1], range(mid)
                )
                for width, j in param_iter:
                    plotted[model_name].append(
                        ax.line(
                            [values[j], values[-(j + 1)]],
                            [y, y],
                            line_width=width,
                            line_color=color,
                        )
                    )
                plotted[model_name].append(
                    ax.circle(
                        x=values[mid],
                        y=y,
                        size=markersize * 0.75,
                        fill_color=color,
                    )
                )
        _title = Title()
        _title.text = f"{hdi_prob:.1%} HDI"
        ax.title = _title

        return ax

    def plot_neff(self, ax, markersize, plotted):
        """Draw effective n for each plotter."""
        max_ess = 0
        for plotter in self.plotters.values():
            for y, ess, color, model_name in plotter.ess():
                if ess is not None:
                    plotted[model_name].append(
                        ax.circle(
                            x=ess,
                            y=y,
                            fill_color=color,
                            size=markersize,
                            line_color="black",
                        )
                    )
                if ess > max_ess:
                    max_ess = ess
        ax.x_range._property_values["start"] = 0  # pylint: disable=protected-access
        ax.x_range._property_values["end"] = 1.07 * max_ess  # pylint: disable=protected-access

        _title = Title()
        _title.text = "ess"
        ax.title = _title

        ax.xaxis[0].ticker.desired_num_ticks = 3

        return ax

    def plot_rhat(self, ax, markersize, plotted):
        """Draw r-hat for each plotter."""
        for plotter in self.plotters.values():
            for y, r_hat, color, model_name in plotter.r_hat():
                if r_hat is not None:
                    plotted[model_name].append(
                        ax.circle(
                            x=r_hat, y=y, fill_color=color, size=markersize, line_color="black"
                        )
                    )
        ax.x_range._property_values["start"] = 0.9  # pylint: disable=protected-access
        ax.x_range._property_values["end"] = 2.1  # pylint: disable=protected-access

        _title = Title()
        _title.text = "r_hat"
        ax.title = _title

        ax.xaxis[0].ticker.desired_num_ticks = 3

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

    def __init__(
        self, var_name, data, y_start, model_names, combined, combine_dims, colors, labeller
    ):
        self.var_name = var_name
        self.data = data
        self.y_start = y_start
        self.model_names = model_names
        self.combined = combined
        self.combine_dims = combine_dims
        self.colors = colors
        self.labeller = labeller
        self.model_color = dict(zip(self.model_names, self.colors))
        max_chains = max(datum.chain.max().values for datum in data)
        self.chain_offset = len(data) * 0.45 / max(1, max_chains)
        self.var_offset = 1.5 * self.chain_offset
        self.group_offset = 2 * self.var_offset

    def iterator(self):
        """Iterate over models and chains for each variable."""
        if self.combined:
            grouped_data = [[(0, datum)] for datum in self.data]
            skip_dims = self.combine_dims.union({"chain"})
        else:
            grouped_data = [datum.groupby("chain") for datum in self.data]
            skip_dims = self.combine_dims

        label_dict = OrderedDict()
        selection_list = []
        for name, grouped_datum in zip(self.model_names, grouped_data):
            for _, sub_data in grouped_datum:
                datum_iter = xarray_var_iter(
                    sub_data,
                    var_names=[self.var_name],
                    skip_dims=skip_dims,
                    reverse_selections=True,
                )
                datum_list = list(datum_iter)
                for _, selection, isel, values in datum_list:
                    selection_list.append(selection)
                    if not selection or not len(selection_list) % len(datum_list):
                        var_name = self.var_name
                    else:
                        var_name = ""
                    label = self.labeller.make_label_flat(var_name, selection, isel)
                    if label not in label_dict:
                        label_dict[label] = OrderedDict()
                    if name not in label_dict[label]:
                        label_dict[label][name] = []
                    label_dict[label][name].append(values)

        y = self.y_start
        for idx, (label, model_data) in enumerate(label_dict.items()):
            for model_name, value_list in model_data.items():
                row_label = self.labeller.make_model_label(model_name, label)
                for values in value_list:
                    yield y, row_label, model_name, label, selection_list[
                        idx
                    ], values, self.model_color[model_name]
                    y += self.chain_offset
                y += self.var_offset
            y += self.group_offset

    def labels_ticks_and_vals(self):
        """Get labels, ticks, values, and colors for the variable."""
        y_ticks = defaultdict(list)
        for y, label, model_name, _, _, vals, color in self.iterator():
            y_ticks[label].append((y, vals, color, model_name))
        labels, ticks, vals, colors, model_names = [], [], [], [], []
        for label, all_data in y_ticks.items():
            for data in all_data:
                labels.append(label)
                ticks.append(data[0])
                vals.append(np.array(data[1]))
                model_names.append(data[3])
                colors.append(data[2])  # the colors are all the same
        return labels, ticks, vals, colors, model_names

    def treeplot(self, qlist, hdi_prob):
        """Get data for each treeplot for the variable."""
        for y, _, model_name, _, selection, values, color in self.iterator():
            ntiles = np.percentile(values.flatten(), qlist)
            ntiles[0], ntiles[-1] = hdi(values.flatten(), hdi_prob, multimodal=False)
            yield y, model_name, selection, ntiles, color

    def ridgeplot(self, hdi_prob, mult, ridgeplot_kind):
        """Get data for each ridgeplot for the variable."""
        xvals, hdi_vals, yvals, pdfs, pdfs_q, colors, model_names = [], [], [], [], [], [], []

        for y, _, model_name, *_, values, color in self.iterator():
            yvals.append(y)
            colors.append(color)
            model_names.append(model_name)
            values = values.flatten()
            values = values[np.isfinite(values)]

            if hdi_prob != 1:
                hdi_ = hdi(values, hdi_prob, multimodal=False)
            else:
                hdi_ = min(values), max(values)

            if ridgeplot_kind == "auto":
                kind = "hist" if np.all(np.mod(values, 1) == 0) else "density"
            else:
                kind = ridgeplot_kind

            if kind == "hist":
                bins = get_bins(values)
                _, density, x = histogram(values, bins=bins)
                x = x[:-1]
            elif kind == "density":
                x, density = kde(values)

            density_q = density.cumsum() / density.sum()

            xvals.append(x)
            pdfs.append(density)
            pdfs_q.append(density_q)
            hdi_vals.append(hdi_)

        scaling = max(np.max(j) for j in pdfs)
        for y, x, hdi_val, pdf, pdf_q, color, model_name in zip(
            yvals, xvals, hdi_vals, pdfs, pdfs_q, colors, model_names
        ):
            yield x, y, mult * pdf / scaling + y, hdi_val, pdf_q, color, model_name

    def ess(self):
        """Get effective n data for the variable."""
        _, y_vals, values, colors, model_names = self.labels_ticks_and_vals()
        for y, value, color, model_name in zip(y_vals, values, colors, model_names):
            yield y, _ess(value), color, model_name

    def r_hat(self):
        """Get rhat data for the variable."""
        _, y_vals, values, colors, model_names = self.labels_ticks_and_vals()
        for y, value, color, model_name in zip(y_vals, values, colors, model_names):
            if value.ndim != 2 or value.shape[0] < 2:
                yield y, None, color, model_name
            else:
                yield y, _rhat(value), color, model_name

    def y_max(self):
        """Get max y value for the variable."""
        end_y = max(y for y, *_ in self.iterator())

        if self.combined:
            end_y += self.group_offset

        return end_y + 2 * self.group_offset
