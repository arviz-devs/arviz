"""Forest plot."""
from collections import defaultdict, OrderedDict
from itertools import tee

import numpy as np
import matplotlib.pyplot as plt

from ..data import convert_to_dataset
from ..stats import hpd
from ..stats.diagnostics import _ess, _rhat
from .plot_utils import _scale_fig_size, xarray_var_iter, make_label
from .kdeplot import _fast_kde
from ..utils import _var_names


def pairwise(iterable):
    """From itertools cookbook. [a, b, c, ...] -> (a, b), (b, c), ..."""
    first, second = tee(iterable)
    next(second, None)
    return zip(first, second)


def plot_forest(
    data,
    kind="forestplot",
    model_names=None,
    var_names=None,
    combined=False,
    credible_interval=0.94,
    rope=None,
    quartiles=True,
    ess=False,
    r_hat=False,
    colors="cycle",
    textsize=None,
    linewidth=None,
    markersize=None,
    ridgeplot_alpha=None,
    ridgeplot_overlap=2,
    figsize=None,
):
    """Forest plot to compare credible intervals from a number of distributions.

    Generates a forest plot of 100*(credible_interval)% credible intervals from
    a trace or list of traces.

    Parameters
    ----------
    data : obj or list[obj]
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    kind : str
        Choose kind of plot for main axis. Supports "forestplot" or "ridgeplot"
    model_names : list[str], optional
        List with names for the models in the list of data. Useful when
        plotting more that one dataset
    var_names: list[str], optional
        List of variables to plot (defaults to None, which results in all
        variables plotted)
    combined : bool
        Flag for combining multiple chains into a single chain. If False (default),
        chains will be plotted separately.
    credible_interval : float, optional
        Credible interval to plot. Defaults to 0.94.
    rope: tuple or dictionary of tuples
        Lower and upper values of the Region Of Practical Equivalence. If a list with one
        interval only is provided, the ROPE will be displayed across the y-axis. If more than one
        interval is provided the length of the list should match the number of variables.
    quartiles : bool, optional
        Flag for plotting the interquartile range, in addition to the credible_interval intervals.
        Defaults to True
    r_hat : bool, optional
        Flag for plotting Split R-hat statistics. Requires 2 or more chains. Defaults to False
    ess : bool, optional
        Flag for plotting the effective sample size. Defaults to False
    colors : list or string, optional
        list with valid matplotlib colors, one color per model. Alternative a string can be passed.
        If the string is `cycle`, it will automatically chose a color per model from the
        matplotlibs cycle. If a single color is passed, eg 'k', 'C2', 'red' this color will be used
        for all models. Defauls to 'cycle'.
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.
    linewidth : int
        Line width throughout. If None it will be autoscaled based on figsize.
    markersize : int
        Markersize throughout. If None it will be autoscaled based on figsize.
    ridgeplot_alpha : float
        Transparency for ridgeplot fill.  If 0, border is colored by model, otherwise
        a black outline is used.
    ridgeplot_overlap : float
        Overlap height for ridgeplots.
    figsize : tuple
        Figure size. If None it will be defined automatically.

    Returns
    -------
    gridspec : matplotlib GridSpec

    Examples
    --------
    Forestpĺot

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> non_centered_data = az.load_arviz_data('non_centered_eight')
        >>> fig, axes = az.plot_forest(non_centered_data,
        >>>                            kind='forestplot',
        >>>                            var_names=['theta'],
        >>>                            combined=True,
        >>>                            ridgeplot_overlap=3,
        >>>                            figsize=(9, 7))
        >>> axes[0].set_title('Estimated theta for 8 schools model')

    Ridgeplot

    .. plot::
        :context: close-figs

        >>> fig, axes = az.plot_forest(non_centered_data,
        >>>                            kind='ridgeplot',
        >>>                            var_names=['theta'],
        >>>                            combined=True,
        >>>                            ridgeplot_overlap=3,
        >>>                            colors='white',
        >>>                            figsize=(9, 7))
        >>> axes[0].set_title('Estimated theta for 8 schools model')
    """
    if not isinstance(data, (list, tuple)):
        data = [data]

    datasets = [convert_to_dataset(datum) for datum in reversed(data)]

    var_names = _var_names(var_names, datasets)

    ncols, width_ratios = 1, [3]

    if ess:
        ncols += 1
        width_ratios.append(1)

    if r_hat:
        ncols += 1
        width_ratios.append(1)

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

    fig, axes = plt.subplots(
        nrows=1,
        ncols=ncols,
        figsize=figsize,
        gridspec_kw={"width_ratios": width_ratios},
        sharey=True,
        constrained_layout=True,
    )

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
        plot_handler.ridgeplot(ridgeplot_overlap, linewidth, ridgeplot_alpha, axes[0])
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

    for ax in axes:
        ax.grid(False)
        # Remove ticklines on y-axes
        for ticks in ax.yaxis.get_major_ticks():
            ticks.tick1On = False
            ticks.tick2On = False

        for loc, spine in ax.spines.items():
            if loc in ["left", "right"]:
                spine.set_visible(False)

        if len(plot_handler.data) > 1:
            plot_handler.make_bands(ax)

    labels, ticks = plot_handler.labels_and_ticks()
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels(labels)
    all_plotters = list(plot_handler.plotters.values())
    y_max = plot_handler.y_max() - all_plotters[-1].group_offset
    if kind == "ridgeplot":  # space at the top
        y_max += ridgeplot_overlap
    axes[0].set_ylim(-all_plotters[0].group_offset, y_max)

    return fig, axes


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
        labels, idxs = [], []
        for plotter in self.plotters.values():
            sub_labels, sub_idxs, _, _ = plotter.labels_ticks_and_vals()
            labels.append(sub_labels)
            idxs.append(sub_idxs)
        return np.concatenate(labels), np.concatenate(idxs)

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

    def ridgeplot(self, mult, linewidth, alpha, ax):
        """Draw ridgeplot for each plotter.

        Parameters
        ----------
        mult : float
            How much to multiply height by. Set this to greater than 1 to have some overlap.
        linewidth : float
            Width of line on border of ridges
        alpha : float
            Transparency of ridges
        ax : Axes
            Axes to draw on
        """
        if alpha is None:
            alpha = 1.0
        zorder = 0
        for plotter in self.plotters.values():
            for x, y_min, y_max, color in plotter.ridgeplot(mult):
                if alpha == 0:
                    border = color
                else:
                    border = "k"
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
            ntiles[0], ntiles[-1] = hpd(values.flatten(), credible_interval)
            yield y, label, ntiles, color

    def ridgeplot(self, mult):
        """Get data for each ridgeplot for the variable."""
        xvals, yvals, pdfs, colors = [], [], [], []
        for y, *_, values, color in self.iterator():
            yvals.append(y)
            colors.append(color)
            values = values.flatten()
            density, lower, upper = _fast_kde(values)
            xvals.append(np.linspace(lower, upper, len(density)))
            pdfs.append(density)

        scaling = max(j.max() for j in pdfs)
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
