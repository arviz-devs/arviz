"""Plot pointwise elpd estimations of inference data."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import NullFormatter
from matplotlib.lines import Line2D

from ..data import convert_to_inference_data
from .plot_utils import (
    _scale_fig_size,
    get_coords,
    color_from_dim,
    format_coords_as_labels,
    set_xticklabels,
)
from ..stats import waic, loo, ELPDData


def plot_elpd(
    compare_dict,
    color=None,
    xlabels=False,
    figsize=None,
    textsize=None,
    coords=None,
    legend=False,
    threshold=None,
    ax=None,
    ic="waic",
    scale="deviance",
    plot_kwargs=None,
):
    """
    Plot a scatter or hexbin matrix of the sampled parameters.

    Parameters
    ----------
    compare_dict : mapping, str -> ELPDData or InferenceData
        A dictionary mapping the model name to the object containing its inference data or
        the result of `waic`/`loo` functions.
        Refer to az.convert_to_inference_data for details on possible dict items
    color : str or array_like, optional
        Colors of the scatter plot, if color is a str all dots will have the same color,
        if it is the size of the observations, each dot will have the specified color,
        otherwise, it will be interpreted as a list of the dims to be used for the color code
    xlabels : bool, optional
        Use coords as xticklabels
    figsize : figure size tuple, optional
        If None, size is (8 + numvars, 8 + numvars)
    textsize: int, optional
        Text size for labels. If None it will be autoscaled based on figsize.
    coords : mapping, optional
        Coordinates of points to plot. **All** values are used for computation, but only a
        a subset can be plotted for convenience.
    legend : bool, optional
        Include a legend to the plot. Only taken into account when color argument is a dim name.
    threshold : float
        If some elpd difference is larger than `threshold * elpd.std()`, show its label. If
        `None`, no observations will be highlighted.
    ax: axes, optional
        Matplotlib axes
    ic : str, optional
        Information Criterion (WAIC or LOO) used to compare models. Default WAIC. Only taken
        into account when input is InferenceData.
    scale : str, optional
        scale argument passed to az.waic or az.loo, see their docs for details. Only taken
        into account when input is InferenceData.
    plot_kwargs : dicts, optional
        Additional keywords passed to ax.plot

    Returns
    -------
    ax : matplotlib axes

    Examples
    --------
    Compare pointwise WAIC for centered and non centered models of the 8school problem

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> idata1 = az.load_arviz_data("centered_eight")
        >>> idata2 = az.load_arviz_data("non_centered_eight")
        >>> az.plot_elpd(
        >>>     {"centered model": idata1, "non centered model": idata2},
        >>>     xlabels=True
        >>> )

    """
    valid_ics = ["waic", "loo"]
    ic = ic.lower()
    if ic not in valid_ics:
        raise ValueError(
            ("Information Criteria type {} not recognized." "IC must be in {}").format(
                ic, valid_ics
            )
        )
    ic_fun = waic if ic == "waic" else loo

    # Make sure all object are ELPDData
    for k, item in compare_dict.items():
        if not isinstance(item, ELPDData):
            compare_dict[k] = ic_fun(convert_to_inference_data(item), pointwise=True, scale=scale)
    ics = [elpd_data.index[0] for elpd_data in compare_dict.values()]
    if not all(x == ics[0] for x in ics):
        raise SyntaxError(
            "All Information Criteria must be of the same kind, but both loo and waic data present"
        )
    ic = ics[0]
    scales = [elpd_data["{}_scale".format(ic)] for elpd_data in compare_dict.values()]
    if not all(x == scales[0] for x in scales):
        raise SyntaxError(
            "All Information Criteria must be on the same scale, but {} are present".format(
                set(scales)
            )
        )
    numvars = len(compare_dict)
    models = list(compare_dict.keys())

    if coords is None:
        coords = {}

    if plot_kwargs is None:
        plot_kwargs = {}
    plot_kwargs.setdefault("marker", "+")

    pointwise_data = [
        get_coords(compare_dict[model]["{}_i".format(ic)], coords) for model in models
    ]
    xdata = np.arange(pointwise_data[0].size)

    if isinstance(color, str):
        if color in pointwise_data[0].dims:
            colors, color_mapping = color_from_dim(pointwise_data[0], color)
            if legend:
                cmap_name = plot_kwargs.pop("cmap", plt.rcParams["image.cmap"])
                markersize = plot_kwargs.pop("s", plt.rcParams["lines.markersize"])
                cmap = getattr(cm, cmap_name)
                handles = [
                    Line2D(
                        [],
                        [],
                        color=cmap(float_color),
                        label=coord,
                        ms=markersize,
                        lw=0,
                        **plot_kwargs
                    )
                    for coord, float_color in color_mapping.items()
                ]
                plot_kwargs.setdefault("cmap", cmap_name)
                plot_kwargs.setdefault("s", markersize ** 2)
            plot_kwargs.setdefault("c", colors)
        else:
            plot_kwargs.setdefault("c", color)
            legend = False
    else:
        legend = False
        plot_kwargs.setdefault("c", color)

    if xlabels:
        coord_labels = format_coords_as_labels(pointwise_data[0])

    if numvars < 2:
        raise Exception("Number of models to compare must be 2 or greater.")

    if numvars == 2:
        (figsize, ax_labelsize, titlesize, xt_labelsize, _, markersize) = _scale_fig_size(
            figsize, textsize, numvars - 1, numvars - 1
        )
        plot_kwargs.setdefault("s", markersize ** 2)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=(not xlabels and not legend))

        ydata = pointwise_data[0] - pointwise_data[1]
        ax.scatter(xdata, ydata, **plot_kwargs)
        if threshold is not None:
            ydata = ydata.values.flatten()
            diff_abs = np.abs(ydata - ydata.mean())
            bool_ary = diff_abs > threshold * ydata.std()
            try:
                coord_labels
            except NameError:
                coord_labels = xdata.astype(str)
            outliers = np.argwhere(bool_ary).squeeze()
            for outlier in outliers:
                label = coord_labels[outlier]
                ax.text(
                    outlier,
                    ydata[outlier],
                    label,
                    horizontalalignment="center",
                    verticalalignment="bottom" if ydata[outlier] > 0 else "top",
                    fontsize=0.8 * xt_labelsize,
                )

        ax.set_title("{} - {}".format(*models), fontsize=titlesize, wrap=True)
        ax.set_ylabel("ELPD difference", fontsize=ax_labelsize, wrap=True)
        ax.tick_params(labelsize=xt_labelsize)
        if xlabels:
            set_xticklabels(ax, coord_labels)
            fig.autofmt_xdate()
        if legend:
            ncols = len(handles) // 6 + 1
            ax.legend(handles=handles, ncol=ncols, title=color)

    else:
        (figsize, ax_labelsize, titlesize, xt_labelsize, _, markersize) = _scale_fig_size(
            figsize, textsize, numvars - 2, numvars - 2
        )
        plot_kwargs.setdefault("s", markersize ** 2)

        if ax is None:
            fig, ax = plt.subplots(
                numvars - 1,
                numvars - 1,
                figsize=figsize,
                constrained_layout=(not xlabels and not legend),
            )

        for i in range(0, numvars - 1):
            var1 = pointwise_data[i]

            for j in range(0, numvars - 1):
                if j < i:
                    ax[j, i].axis("off")
                    continue

                var2 = pointwise_data[j + 1]
                ax[j, i].scatter(xdata, var1 - var2, **plot_kwargs)
                if threshold is not None:
                    ydata = (var1 - var2).values.flatten()
                    diff_abs = np.abs(ydata - ydata.mean())
                    bool_ary = diff_abs > threshold * ydata.std()
                    try:
                        coord_labels
                    except NameError:
                        coord_labels = xdata.astype(str)
                    outliers = np.argwhere(bool_ary).squeeze()
                    for outlier in outliers:
                        label = coord_labels[outlier]
                        ax[j, i].text(
                            outlier,
                            ydata[outlier],
                            label,
                            horizontalalignment="center",
                            verticalalignment="bottom" if ydata[outlier] > 0 else "top",
                            fontsize=0.8 * xt_labelsize,
                        )

                if j + 1 != numvars - 1:
                    ax[j, i].axes.get_xaxis().set_major_formatter(NullFormatter())
                    ax[j, i].set_xticks([])
                elif xlabels:
                    set_xticklabels(ax[j, i], coord_labels)

                if i != 0:
                    ax[j, i].axes.get_yaxis().set_major_formatter(NullFormatter())
                    ax[j, i].set_yticks([])
                else:
                    ax[j, i].set_ylabel("ELPD difference", fontsize=ax_labelsize, wrap=True)

                ax[j, i].tick_params(labelsize=xt_labelsize)
                ax[j, i].set_title(
                    "{} - {}".format(models[i], models[j + 1]), fontsize=titlesize, wrap=True
                )
        if xlabels:
            fig.autofmt_xdate()
        if legend:
            ncols = len(handles) // 6 + 1
            ax[0, 1].legend(
                handles=handles, ncol=ncols, title=color, bbox_to_anchor=(0, 1), loc="upper left"
            )
    return ax
