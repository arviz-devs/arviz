"""Matplotlib traceplot."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from . import backend_kwarg_defaults, backend_show
from ...distplot import plot_dist
from ...plot_utils import _scale_fig_size, get_bins, make_label


def plot_trace(
    data,
    var_names,  # pylint: disable=unused-argument
    divergences,
    figsize,
    rug,
    lines,
    combined,
    legend,
    plot_kwargs,
    fill_kwargs,
    rug_kwargs,
    hist_kwargs,
    trace_kwargs,
    plotters,
    divergence_data,
    colors,
    backend_kwargs,
    show,
):
    """Plot distribution (histogram or kernel density estimates) and sampled values.

    If `divergences` data is available in `sample_stats`, will plot the location of divergences as
    dashed vertical lines.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names : string, or list of strings
        One or more variables to be plotted.
    divergences : {"bottom", "top", None, False}
        Plot location of divergences on the traceplots. Options are "bottom", "top", or False-y.
    figsize : figure size tuple
        If None, size is (12, variables * 2)
    rug : bool
        If True adds a rugplot. Defaults to False. Ignored for 2D KDE. Only affects continuous
        variables.
    lines : tuple
        Tuple of (var_name, {'coord': selection}, [line, positions]) to be overplotted as
        vertical lines on the density and horizontal lines on the trace.
    combined : bool
        Flag for combining multiple chains into a single line. If False (default), chains will be
        plotted separately.
    legend : bool
        Add a legend to the figure with the chain color code.
    plot_kwargs : dict
        Extra keyword arguments passed to `arviz.plot_dist`. Only affects continuous variables.
    fill_kwargs : dict
        Extra keyword arguments passed to `arviz.plot_dist`. Only affects continuous variables.
    rug_kwargs : dict
        Extra keyword arguments passed to `arviz.plot_dist`. Only affects continuous variables.
    hist_kwargs : dict
        Extra keyword arguments passed to `arviz.plot_dist`. Only affects discrete variables.
    trace_kwargs : dict
        Extra keyword arguments passed to `plt.plot`
    Returns
    -------
    axes : matplotlib axes


    Examples
    --------
    Plot a subset variables

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('non_centered_eight')
        >>> coords = {'school': ['Choate', 'Lawrenceville']}
        >>> az.plot_trace(data, var_names=('theta_t', 'theta'), coords=coords)

    Show all dimensions of multidimensional variables in the same plot

    .. plot::
        :context: close-figs

        >>> az.plot_trace(data, compact=True)

    Combine all chains into one distribution

    .. plot::
        :context: close-figs

        >>> az.plot_trace(data, var_names=('theta_t', 'theta'), coords=coords, combined=True)


    Plot reference lines against distribution and trace

    .. plot::
        :context: close-figs

        >>> lines = (('theta_t',{'school': "Choate"}, [-1]),)
        >>> az.plot_trace(data, var_names=('theta_t', 'theta'), coords=coords, lines=lines)

    """
    # Set plot default backend kwargs
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {**backend_kwarg_defaults(), **backend_kwargs}

    textsize = plot_kwargs.pop("textsize", 10)

    figsize, _, titlesize, xt_labelsize, linewidth, _ = _scale_fig_size(
        figsize, textsize, rows=len(plotters), cols=2
    )

    trace_kwargs.setdefault("linewidth", linewidth)
    plot_kwargs.setdefault("linewidth", linewidth)

    _, axes = plt.subplots(len(plotters), 2, squeeze=False, figsize=figsize, **backend_kwargs)

    for idx, (var_name, selection, value) in enumerate(plotters):
        value = np.atleast_2d(value)

        if len(value.shape) == 2:
            _plot_chains_mpl(
                axes,
                idx,
                value,
                data,
                colors,
                combined,
                xt_labelsize,
                rug,
                trace_kwargs,
                hist_kwargs,
                plot_kwargs,
                fill_kwargs,
                rug_kwargs,
            )
        else:
            value = value.reshape((value.shape[0], value.shape[1], -1))
            for sub_idx in range(value.shape[2]):
                _plot_chains_mpl(
                    axes,
                    idx,
                    value[..., sub_idx],
                    data,
                    colors,
                    combined,
                    xt_labelsize,
                    rug,
                    trace_kwargs,
                    hist_kwargs,
                    plot_kwargs,
                    fill_kwargs,
                    rug_kwargs,
                )

        if value[0].dtype.kind == "i":
            xticks = get_bins(value)
            axes[idx, 0].set_xticks(xticks[:-1])
        axes[idx, 0].set_yticks([])
        for col in (0, 1):
            axes[idx, col].set_title(make_label(var_name, selection), fontsize=titlesize, wrap=True)
            axes[idx, col].tick_params(labelsize=xt_labelsize)

        xlims = [ax.get_xlim() for ax in axes[idx, :]]
        ylims = [ax.get_ylim() for ax in axes[idx, :]]

        if divergences:
            div_selection = {k: v for k, v in selection.items() if k in divergence_data.dims}
            divs = divergence_data.sel(**div_selection).values
            # if combined:
            #     divs = divs.flatten()
            divs = np.atleast_2d(divs)

            for chain, chain_divs in enumerate(divs):
                div_draws = data.draw.values[chain_divs]
                div_idxs = np.arange(len(chain_divs))[chain_divs]
                if div_idxs.size > 0:
                    if divergences == "top":
                        ylocs = [ylim[1] for ylim in ylims]
                    else:
                        ylocs = [ylim[0] for ylim in ylims]
                    values = value[chain, div_idxs]
                    axes[idx, 1].plot(
                        div_draws,
                        np.zeros_like(div_idxs) + ylocs[1],
                        marker="|",
                        color="black",
                        markeredgewidth=1.5,
                        markersize=30,
                        linestyle="None",
                        alpha=hist_kwargs["alpha"],
                        zorder=-5,
                    )
                    axes[idx, 1].set_ylim(*ylims[1])
                    axes[idx, 0].plot(
                        values,
                        np.zeros_like(values) + ylocs[0],
                        marker="|",
                        color="black",
                        markeredgewidth=1.5,
                        markersize=30,
                        linestyle="None",
                        alpha=trace_kwargs["alpha"],
                        zorder=-5,
                    )
                    axes[idx, 0].set_ylim(*ylims[0])

        for _, _, vlines in (j for j in lines if j[0] == var_name and j[1] == selection):
            if isinstance(vlines, (float, int)):
                line_values = [vlines]
            else:
                line_values = np.atleast_1d(vlines).ravel()
            axes[idx, 0].vlines(line_values, *ylims[0], colors="black", linewidth=1.5, alpha=0.75)
            axes[idx, 1].hlines(
                line_values, *xlims[1], colors="black", linewidth=1.5, alpha=trace_kwargs["alpha"]
            )
        axes[idx, 0].set_ylim(bottom=0, top=ylims[0][1])
        axes[idx, 1].set_xlim(left=data.draw.min(), right=data.draw.max())
        axes[idx, 1].set_ylim(*ylims[1])
    if legend:
        handles = [
            Line2D([], [], color=color, label=chain_id)
            for chain_id, color in zip(data.chain.values, colors)
        ]
        if combined:
            handles.insert(0, Line2D([], [], color=colors[-1], label="combined"))
        axes[0, 1].legend(handles=handles, title="chain")

    if backend_show(show):
        plt.show()

    return axes


def _plot_chains_mpl(
    axes,
    idx,
    value,
    data,
    colors,
    combined,
    xt_labelsize,
    rug,
    trace_kwargs,
    hist_kwargs,
    plot_kwargs,
    fill_kwargs,
    rug_kwargs,
):
    for chain_idx, row in enumerate(value):
        axes[idx, 1].plot(data.draw.values, row, color=colors[chain_idx], **trace_kwargs)

        if not combined:
            plot_kwargs["color"] = colors[chain_idx]
            plot_dist(
                values=row,
                textsize=xt_labelsize,
                rug=rug,
                ax=axes[idx, 0],
                hist_kwargs=hist_kwargs,
                plot_kwargs=plot_kwargs,
                fill_kwargs=fill_kwargs,
                rug_kwargs=rug_kwargs,
                backend="matplotlib",
                backend_kwargs={},
                show=False,
            )

    if combined:
        plot_kwargs["color"] = colors[-1]
        plot_dist(
            values=value.flatten(),
            textsize=xt_labelsize,
            rug=rug,
            ax=axes[idx, 0],
            hist_kwargs=hist_kwargs,
            plot_kwargs=plot_kwargs,
            fill_kwargs=fill_kwargs,
            rug_kwargs=rug_kwargs,
            backend="matplotlib",
            backend_kwargs={},
            show=False,
        )
