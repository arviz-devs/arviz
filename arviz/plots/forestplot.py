from collections import defaultdict
from itertools import tee

import numpy as np
import matplotlib.pyplot as plt

from ..stats.diagnostics import _get_neff, _get_rhat
from ..stats import hpd
from .plot_utils import _scale_text, xarray_var_iter, make_label
from ..utils import convert_to_xarray
from .kdeplot import fast_kde


def pairwise(iterable):
    """From itertools cookbook. [a, b, c, ...] -> (a, b), (b, c), ..."""
    first, second = tee(iterable)
    next(second, None)
    return zip(first, second)


def forestplot(data, model_names=None, var_names=None, combined=False, credible_interval=0.95,
               quartiles=True, joyplot=False, r_hat=True, n_eff=True, colors='cycle', textsize=None,
               linewidth=None, markersize=None, joyplot_alpha=None, figsize=None):
    """
    Forest plot

    Generates a forest plot of 100*(credible_interval)% credible intervals from
    a trace or list of traces.

    Parameters
    ----------
    data : xarray.Dataset or list of compatible
        Samples from a model posterior
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
        Credible interval to plot. Defaults to 0.95.
    quartiles : bool, optional
        Flag for plotting the interquartile range, in addition to the credible_interval intervals.
        Defaults to True
    r_hat : bool, optional
        Flag for plotting Gelman-Rubin statistics. Requires 2 or more chains. Defaults to True
    n_eff : bool, optional
        Flag for plotting the effective sample size. Requires 2 or more chains. Defaults to True
    colors : list or string, optional
        list with valid matplotlib colors, one color per model. Alternative a string can be passed.
        If the string is `cycle`, it will automatically chose a color per model from the
        matyplolibs cycle. If a single color is passed, eg 'k', 'C2', 'red' this color will be used
        for all models. Defauls to 'cycle'.
    textsize: int
        Text size for labels. If None it will be autoscaled based on figsize.
    linewidth : int
        Line width throughout. If None it will be autoscaled based on figsize.
    markersize : int
        Markersize throughout. If None it will be autoscaled based on figsize.
    joyplot_alpha : float
        Transparency for joyplot fill.  If 0, border is colored by model, otherwise
        a black outline is used.
    figsize : tuple, optional
        Figure size. Defaults to None

    Returns
    -------
    gridspec : matplotlib GridSpec

    """

    ncols, width_ratios = 1, [3]

    if n_eff:
        ncols += 1
        width_ratios.append(1)

    if r_hat:
        ncols += 1
        width_ratios.append(1)

    plot_handler = PlotHandler(data, var_names=var_names, model_names=model_names,
                               combined=combined, colors=colors)

    if figsize is None:
        figsize = (min(12, sum(width_ratios) * 2), plot_handler.fig_height())

    textsize, auto_linewidth, auto_markersize = _scale_text(figsize, textsize=textsize)
    if linewidth is None:
        linewidth = auto_linewidth

    if markersize is None:
        markersize = auto_markersize

    fig, axes = plt.subplots(nrows=1,
                             ncols=ncols,
                             figsize=figsize,
                             gridspec_kw={'width_ratios': width_ratios},
                             sharey=True
                            )

    axes = np.atleast_1d(axes)
    if joyplot:
        plot_handler.joyplot(textsize, linewidth, joyplot_alpha, axes[0])

    else:
        plot_handler.forestplot(credible_interval, quartiles, textsize,
                                linewidth, markersize, axes[0])

    idx = 1
    if r_hat:
        plot_handler.plot_rhat(axes[idx], textsize, markersize)
        idx += 1

    if n_eff:
        plot_handler.plot_neff(axes[idx], textsize, markersize)
        idx += 1

    for ax in axes:
        ax.grid(False)
        # Remove ticklines on y-axes
        for ticks in ax.yaxis.get_major_ticks():
            ticks.tick1On = False
            ticks.tick2On = False

        for loc, spine in ax.spines.items():
            if loc in ['left', 'right']:
                spine.set_color('none')  # don't draw spine

        if len(plot_handler.data) > 1:
            plot_handler.make_bands(ax)

    labels, ticks = plot_handler.labels_and_ticks()
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels(labels)
    all_plotters = list(plot_handler.plotters.values())
    y_max = plot_handler.y_max() - all_plotters[-1].group_offset
    if joyplot:
        y_max += 1
    axes[0].set_ylim(-all_plotters[0].group_offset, y_max)

    return fig, axes


class PlotHandler(object):
    def __init__(self, data, var_names, model_names, combined, colors):
        if not isinstance(data, (list, tuple)):
            data = [data]

        self.data = [convert_to_xarray(datum) for datum in reversed(data)]  # y-values upside down

        if model_names is None:
            if len(self.data) > 1:
                model_names = [f'Model {idx}' for idx, _ in enumerate(self.data)]
            else:
                model_names = ['']
        elif len(model_names) != len(self.data):
            raise ValueError("The number of model names does not match the number of models")

        self.model_names = list(reversed(model_names))  # y-values are upside down

        if var_names is None:
            self.var_names = list(set.union(*[set(datum.data_vars) for datum in self.data]))
        else:
            self.var_names = list(reversed(var_names))  # y-values are upside down

        self.combined = combined

        if colors == 'cycle':
            colors = [f'C{idx % 10}' for idx, _ in enumerate(self.data)]
        elif isinstance(colors, str):
            colors = [colors for _ in self.data]

        self.colors = list(reversed(colors))  # y-values are upside down

        self.plotters = self.make_plotters()

    def make_plotters(self):
        plotters, y = {}, 0
        for var_name in self.var_names:
            plotters[var_name] = VarHandler(var_name, self.data, y,
                                            model_names=self.model_names,
                                            combined=self.combined,
                                            colors=self.colors,
                                           )
            y = plotters[var_name].y_max()
        return plotters

    def labels_and_ticks(self):
        labels, idxs = [], []
        for plotter in self.plotters.values():
            sub_labels, sub_idxs, _, _ = plotter.labels_ticks_and_vals()
            labels.append(sub_labels)
            idxs.append(sub_idxs)
        return np.concatenate(labels), np.concatenate(idxs)

    def joyplot(self, textsize, linewidth, alpha, ax):
        if self.combined:
            mult = 1.8
        else:
            mult = 1.

        if alpha is None:
            alpha = 1.
        zorder = 0
        for plotter in self.plotters.values():
            for x, y_min, y_max, color in plotter.joyplot(mult):
                if alpha == 0:
                    border = color
                else:
                    border = 'k'
                ax.plot(x, y_max, '-', linewidth=linewidth, color=border, zorder=zorder)
                ax.plot(x, y_min, '-', linewidth=linewidth, color=border, zorder=zorder)
                ax.fill_between(x, y_min, y_max, alpha=alpha, color=color, zorder=zorder)
                zorder -= 1

        ax.tick_params(labelsize=textsize)

        return ax
    def forestplot(self, credible_interval, quartiles, textsize, linewidth, markersize, ax):
        # Quantiles to be calculated
        endpoint = 100 * (1 - credible_interval) / 2
        if quartiles:
            qlist = [endpoint, 25, 50, 75, 100 - endpoint]
        else:
            qlist = [endpoint, 50, 100 - endpoint]


        for plotter in self.plotters.values():
            for y, values, color in plotter.treeplot(qlist, credible_interval):
                mid = len(values) // 2
                param_iter = zip(
                    np.linspace(2 * linewidth, linewidth, mid, endpoint=True)[-1::-1],
                    range(mid))
                for width, j in param_iter:
                    ax.hlines(y, values[j], values[-(j + 1)], linewidth=width, color=color)
                ax.plot(values[mid], y, 'o',
                        mfc=ax.get_facecolor(),
                        markersize=markersize * 0.75,
                        color=color)
        ax.tick_params(labelsize=textsize)
        ax.set_title(f'{credible_interval:.1%} Credible Interval', fontsize=textsize)

        return ax

    def plot_neff(self, ax, textsize, markersize):
        for plotter in self.plotters.values():
            for y, n_eff, color in plotter.n_eff():
                if n_eff is not None:
                    ax.plot(n_eff, y, 'o', color=color, markersize=markersize)
        ax.set_xlim(left=0)
        ax.set_title('Effective n', fontsize=textsize)
        ax.tick_params(labelsize=textsize)
        return ax

    def plot_rhat(self, ax, textsize, markersize):
        for plotter in self.plotters.values():
            for y, r_hat, color in plotter.r_hat():
                if r_hat is not None:
                    ax.plot(r_hat, y, 'o', color=color, markersize=markersize)
        ax.set_xlim(left=0.9, right=2.1)
        ax.set_xticks([1, 2])
        ax.tick_params(labelsize=textsize)
        ax.set_title('R-hat', fontsize=textsize)
        return ax

    def make_bands(self, ax):
        y_vals, y_prev, is_zero = [0], None, False
        prev_color_index = 0
        plotter = None  # To make sure it is defined
        for plotter in self.plotters.values():
            for y, _, _, color in plotter.iterator():
                if self.colors.index(color) < prev_color_index:
                    if not is_zero and y_prev is not None:
                        y_vals.append((y + y_prev) * 0.5)
                        is_zero = True
                else:
                    is_zero = False
                prev_color_index = self.colors.index(color)
                y_prev = y

        if plotter is None:
            offset = 1
        else:
            offset = plotter.group_offset

        y_vals.append(y_prev + offset)
        for idx, (y_start, y_stop) in enumerate(pairwise(y_vals)):
            ax.axhspan(y_start, y_stop, color='k', alpha=0.1 * (idx % 2))
        return ax

    def fig_height(self):
        # hand-tuned
        return (len(self.data) * len(self.var_names) - 1 +
                0.25 * sum(1 for j in self.plotters.values() for _ in j.iterator())
               )

    def y_max(self):
        return max(p.y_max() for p in self.plotters.values())


class VarHandler(object):
    def __init__(self, var_name, data, y_start, model_names, combined, colors):
        self.var_name = var_name
        self.data = data
        self.y_start = y_start
        self.model_names = model_names
        self.combined = combined
        self.colors = colors
        self.model_color = dict(zip(self.model_names, self.colors))
        self.chain_offset = len(data) * 0.45 / max(datum.chain.max().values for datum in data)
        self.var_offset = 1.5 * self.chain_offset
        self.group_offset = 2 * self.var_offset

    def iterator(self):
        if self.combined:
            grouped_data = [[(0, datum)] for datum in self.data]
            skip_dims = {'chain'}
        else:
            grouped_data = [datum.groupby('chain') for datum in self.data]
            skip_dims = set()

        label_dict = {}
        for name, grouped_datum in zip(self.model_names, grouped_data):
            for _, sub_data in grouped_datum:
                datum_iter = xarray_var_iter(sub_data,
                                             var_names=[self.var_name],
                                             skip_dims=skip_dims)
                for _, selection, values in datum_iter:
                    label = make_label(self.var_name, selection)
                    if label not in label_dict:
                        label_dict[label] = {}
                    if name not in label_dict[label]:
                        label_dict[label][name] = []
                    label_dict[label][name].append(values)

        y = self.y_start
        for label, model_data in label_dict.items():
            for model_name, value_list in model_data.items():
                if model_name:
                    row_label = f'{model_name}: {label}'
                else:
                    row_label = label
                for values in value_list:
                    yield y, row_label, values, self.model_color[model_name]
                    y += self.chain_offset
                y += self.var_offset
            y += self.group_offset

    def labels_ticks_and_vals(self):
        y_ticks = defaultdict(list)
        for y, label, vals, color in self.iterator():
            y_ticks[label].append((y, vals, color))
        labels, ticks, vals, colors = [], [], [], []
        for label, data in y_ticks.items():
            labels.append(label)
            ticks.append(np.mean([j[0] for j in data]))
            vals.append(np.vstack([j[1] for j in data]))
            colors.append(data[0][2])  # the colors are all the same
        return labels, ticks, vals, colors

    def treeplot(self, qlist, credible_interval):
        for y, _, values, color in self.iterator():
            ntiles = np.percentile(values.flatten(), qlist)
            ntiles[0], ntiles[-1] = hpd(values.flatten(), alpha=1-credible_interval)
            yield y, ntiles, color

    def joyplot(self, mult):
        xvals, yvals, pdfs, colors = [], [], [], []
        for y, _, values, color in self.iterator():
            yvals.append(y)
            colors.append(color)
            values = values.flatten()
            density, lower, upper = fast_kde(values)
            xvals.append(np.linspace(lower, upper, len(density)))
            pdfs.append(density)

        scaling = max(j.max() for j in pdfs)
        for y, x, pdf, color in zip(yvals, xvals, pdfs, colors):
            y = y * np.ones_like(x)
            yield x, y, mult * pdf / scaling + y, color

    def n_eff(self):
        _, y_vals, values, colors = self.labels_ticks_and_vals()
        for y, value, color in zip(y_vals, values, colors):
            if value.ndim != 2 or value.shape[0] < 2:
                yield y, None, color
            else:
                yield y, _get_neff(value), color

    def r_hat(self):
        _, y_vals, values, colors = self.labels_ticks_and_vals()
        for y, value, color in zip(y_vals, values, colors):
            if value.ndim != 2 or value.shape[0] < 2:
                yield y, None, color
            else:
                yield y, _get_rhat(value), color

    def y_max(self):
        end_y = max(y for y, *_ in self.iterator())

        if self.combined:
            end_y += self.group_offset

        return end_y + 2 * self.group_offset
