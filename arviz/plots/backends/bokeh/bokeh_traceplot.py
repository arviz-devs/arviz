import bokeh.plotting as bkp
from bokeh.models.annotations import Title
from bokeh.layouts import gridplot
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np

from ....data import convert_to_dataset
from ...distplot import plot_dist
from ...plot_utils import _scale_fig_size, get_bins, xarray_var_iter, make_label, get_coords
from ....utils import _var_names
from ....rcparams import rcParams

def _plot_trace_bokeh(
    data,
    var_names=None,
    coords=None,
    divergences="bottom",
    figsize=None,
    textsize=None,
    lines=None,
    compact=False,
    combined=False,
    legend=False,
    plot_kwargs=None,
    fill_kwargs=None,
    rug_kwargs=None,
    hist_kwargs=None,
    trace_kwargs=None,
    backend_kwargs=None,
    show=True,
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
    coords : mapping, optional
        Coordinates of var_names to be plotted. Passed to `Dataset.sel`
    divergences : {"bottom", "top", None, False}
        Plot location of divergences on the traceplots. Options are "bottom", "top", or False-y.
    figsize : figure size tuple
        If None, size is (12, variables * 2)
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.
    lines : tuple
        Tuple of (var_name, {'coord': selection}, [line, positions]) to be overplotted as
        vertical lines on the density and horizontal lines on the trace.
    compact : bool
        Plot multidimensional variables in a single plot.
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
        Extra keyword arguments passed to `bokeh.plotting.lines`
    Returns
    -------
    axes : bokeh figures


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

    if divergences:
        try:
            divergence_data = convert_to_dataset(data, group="sample_stats").diverging
        except (ValueError, AttributeError):  # No sample_stats, or no `.diverging`
            divergences = False

    if coords is None:
        coords = {}

    data = get_coords(convert_to_dataset(data, group="posterior"), coords)
    var_names = _var_names(var_names, data)

    if divergences:
        divergence_data = get_coords(
            divergence_data, {k: v for k, v in coords.items() if k in ("chain", "draw")}
        )

    if lines is None:
        lines = ()

    num_colors = len(data.chain) + 1 if combined else len(data.chain)
    colors = [
        prop
        for _, prop in zip(
            range(num_colors), cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        )
    ]

    if compact:
        skip_dims = set(data.dims) - {"chain", "draw"}
    else:
        skip_dims = set()

    plotters = list(xarray_var_iter(data, var_names=var_names, combined=True, skip_dims=skip_dims))
    max_plots = rcParams["plot.max_subplots"]
    max_plots = len(plotters) if max_plots is None else max_plots
    if len(plotters) > max_plots:
        warnings.warn(
            "rcParams['plot.max_subplots'] ({max_plots}) is smaller than the number "
            "of variables to plot ({len_plotters}), generating only {max_plots} "
            "plots".format(max_plots=max_plots, len_plotters=len(plotters)),
            SyntaxWarning,
        )
        plotters = plotters[:max_plots]

    if figsize is None:
        figsize = (12, len(plotters) * 2)

    if trace_kwargs is None:
        trace_kwargs = {}

    trace_kwargs.setdefault("alpha", 0.35)

    if hist_kwargs is None:
        hist_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    if fill_kwargs is None:
        fill_kwargs = {}
    if rug_kwargs is None:
        rug_kwargs = {}

    hist_kwargs.setdefault("alpha", 0.35)

    figsize, _, titlesize, xt_labelsize, linewidth, _ = _scale_fig_size(
        figsize, textsize, rows=len(plotters), cols=2
    )
    figsize = int(figsize[0] * 90 // 2), int(figsize[1] * 90 // len(plotters))

    trace_kwargs.setdefault("line_width", linewidth)
    plot_kwargs.setdefault("line_width", linewidth)

    if backend_kwargs is None:
        backend_kwargs = dict()

    backend_kwargs.setdefault("tools",("pan,wheel_zoom,box_zoom,"
                                       "lasso_select,poly_select,"
                                       "undo,redo,reset,save,hover"))
    backend_kwargs.setdefault("output_backend", "webgl")
    backend_kwargs.setdefault("height", figsize[1])
    backend_kwargs.setdefault("width", figsize[0])

    axes = []
    for i in range(len(plotters)):
        if i != 0:
            _axes = [
                bkp.figure(
                    **backend_kwargs
                ),
                bkp.figure(
                    x_range=axes[0][1].x_range,
                    **backend_kwargs
                ),
            ]
        else:
            _axes = [
                bkp.figure(
                    **backend_kwargs
                ),
                bkp.figure(
                    **backend_kwargs
                ),
            ]
        axes.append(_axes)

    axes = np.array(axes)

    for idx, (var_name, selection, value) in enumerate(plotters):
        value = np.atleast_2d(value)

        if len(value.shape) == 2:
            _plot_chains_bokeh(
                axes,
                idx,
                value,
                data,
                colors,
                combined,
                legend,
                xt_labelsize,
                trace_kwargs,
                hist_kwargs,
                plot_kwargs,
                fill_kwargs,
                rug_kwargs,
            )
        else:
            value = value.reshape((value.shape[0], value.shape[1], -1))
            for sub_idx in range(value.shape[2]):
                _plot_chains_bokeh(
                    axes,
                    idx,
                    value[..., sub_idx],
                    data,
                    colors,
                    combined,
                    xt_labelsize,
                    trace_kwargs,
                    hist_kwargs,
                    plot_kwargs,
                    fill_kwargs,
                    rug_kwargs,
                )

        for col in (0, 1):
            _title = Title()
            _title.text = make_label(var_name, selection)
            axes[idx, col].title = _title

        # TODO
        if False:  # divergences:
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
                    axes[idx, 1].circle(
                        div_draws,
                        np.zeros_like(div_idxs) + ylocs[1],
                        # marker="|",
                        line_color="black",
                        line_alpha=hist_kwargs["alpha"],
                    )
                    axes[idx, 0].circle(
                        values,
                        np.zeros_like(values) + ylocs[0],
                        # marker="|",
                        line_color="black",
                        line_alpha=trace_kwargs["alpha"],
                        # zorder=-5,
                    )

        # TODO
        # for _, _, vlines in (j for j in lines if j[0] == var_name and j[1] == selection):
        #    if isinstance(vlines, (float, int)):
        #        line_values = [vlines]
        #    else:
        #        line_values = np.atleast_1d(vlines).ravel()
        #    axes[idx, 0].vlines(line_values, *ylims[0], colors="black", linewidth=1.5, alpha=0.75)
        #    axes[idx, 1].hlines(
        #        line_values, *xlims[1], colors="black", linewidth=1.5, alpha=trace_kwargs["alpha"]
        #    )

        if legend:
            for col in (0, 1):
                axes[idx, col].legend.location = "top_left"
                axes[idx, col].legend.click_policy="hide"

    if show:
        grid = gridplot([list(item) for item in axes], toolbar_location="above")
        bkp.show(grid)

    return axes


def _plot_chains_bokeh(
    axes,
    idx,
    value,
    data,
    colors,
    combined,
    legend,
    xt_labelsize,
    trace_kwargs,
    hist_kwargs,
    plot_kwargs,
    fill_kwargs,
    rug_kwargs,
):
    for chain_idx, row in enumerate(value):
        # do this manually?
        # https://stackoverflow.com/questions/36561476/change-color-of-non-selected-bokeh-lines
        if legend:
            axes[idx, 1].line(data.draw.values, row, line_color=colors[chain_idx], legend_label="chain {} - line".format(chain_idx), **trace_kwargs)
            axes[idx, 1].circle(
                data.draw.values,
                row,
                radius=1,
                line_color=colors[chain_idx],
                fill_color=colors[chain_idx],
                alpha=0.5,
                legend_label="chain {} - scatter".format(chain_idx)
            )
        else:
            # tmp hack
            axes[idx, 1].line(data.draw.values, row, line_color=colors[chain_idx], **trace_kwargs)
            axes[idx, 1].circle(
                data.draw.values,
                row,
                radius=1,
                line_color=colors[chain_idx],
                fill_color=colors[chain_idx],
                alpha=0.5,
            )
        if not combined:
            if legend:
                plot_kwargs["legend_label"] = "chain {}".format(chain_idx)
            plot_kwargs["line_color"] = colors[chain_idx]
            plot_dist(
                row,
                textsize=xt_labelsize,
                ax=axes[idx, 0],
                hist_kwargs=hist_kwargs,
                plot_kwargs=plot_kwargs,
                fill_kwargs=fill_kwargs,
                rug_kwargs=rug_kwargs,
                backend="bokeh",
                show=False,
            )

    if combined:
        plot_kwargs["line_color"] = colors[-1]
        plot_dist(
            value.flatten(),
            textsize=xt_labelsize,
            ax=axes[idx, 0],
            hist_kwargs=hist_kwargs,
            plot_kwargs=plot_kwargs,
            fill_kwargs=fill_kwargs,
            rug_kwargs=rug_kwargs,
            backend="bokeh",
            show=False,
        )
