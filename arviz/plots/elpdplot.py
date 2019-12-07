"""Plot pointwise elpd estimations of inference data."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

from .backends import check_bokeh_version
from ..data import convert_to_inference_data
from .plot_utils import (
    get_coords,
    format_coords_as_labels,
    color_from_dim,
)
from ..stats import waic, loo, ELPDData
from ..rcparams import rcParams


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
    ic=None,
    scale="deviance",
    plot_kwargs=None,
    backend=None,
    show=True,
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
        Information Criterion (WAIC or LOO) used to compare models. Defaults to
        ``rcParams["stats.information_criterion"]``. Only taken
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
    ic = rcParams["stats.information_criterion"] if ic is None else ic.lower()
    if ic not in valid_ics:
        raise ValueError(
            ("Information Criteria type {} not recognized." "IC must be in {}").format(
                ic, valid_ics
            )
        )
    ic_fun = waic if ic == "waic" else loo
    handles = None
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
    backend = rcParams["plot.backend"] if backend is None else backend.lower()
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
    coord_labels = format_coords_as_labels(pointwise_data[0]) if xlabels else None

    markersize = None
    if isinstance(color, str):
        if color in pointwise_data[0].dims:
            colors, color_mapping = color_from_dim(pointwise_data[0], color)
            if legend and backend != "bokeh":
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

    if numvars < 2:
        raise Exception("Number of models to compare must be 2 or greater.")

    # flatten data (data must be flattened after selecting, labeling and coloring)
    pointwise_data = [pointwise.values.flatten() for pointwise in pointwise_data]

    elpd_plot_kwargs = dict(
        ax=ax,
        models=models,
        pointwise_data=pointwise_data,
        numvars=numvars,
        figsize=figsize,
        textsize=textsize,
        plot_kwargs=plot_kwargs,
        markersize=markersize,
        xlabels=xlabels,
        coord_labels=coord_labels,
        xdata=xdata,
        threshold=threshold,
        legend=legend,
        handles=handles,
        color=color,
    )

    if backend == "bokeh":
        check_bokeh_version()
        from .backends.bokeh.bokeh_elpdplot import _plot_elpd

        elpd_plot_kwargs.pop("legend")
        elpd_plot_kwargs.pop("handles")
        elpd_plot_kwargs.pop("color")
        elpd_plot_kwargs["show"] = show
        ax = _plot_elpd(**elpd_plot_kwargs)  # pylint: disable=unexpected-keyword-arg
    elif backend == "matplotlib":
        from .backends.matplotlib.mpl_elpdplot import _plot_elpd

        ax = _plot_elpd(**elpd_plot_kwargs)

    return ax
