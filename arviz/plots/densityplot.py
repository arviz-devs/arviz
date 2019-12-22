"""KDE and histogram plots for multiple variables."""
from itertools import cycle
import warnings

import matplotlib.pyplot as plt

from ..data import convert_to_dataset
from .plot_utils import (
    _scale_fig_size,
    make_label,
    xarray_var_iter,
    default_grid,
    get_plotting_function,
)
from ..utils import _var_names
from ..rcparams import rcParams


# pylint:disable-msg=too-many-function-args
def plot_density(
    data,
    group="posterior",
    data_labels=None,
    var_names=None,
    credible_interval=0.94,
    point_estimate="mean",
    colors="cycle",
    outline=True,
    hpd_markers="",
    shade=0.0,
    bw=4.5,
    figsize=None,
    textsize=None,
    ax=None,
    backend=None,
    backend_kwargs=None,
    show=None,
):
    """Generate KDE plots for continuous variables and histograms for discrete ones.

    Plots are truncated at their 100*(1-alpha)% credible intervals. Plots are grouped per variable
    and colors assigned to models.

    Parameters
    ----------
    data : Union[Object, Iterator[Object]]
        Any object that can be converted to an az.InferenceData object, or an Iterator returning
        a sequence of such objects.
        Refer to documentation of az.convert_to_dataset for details about such objects.
    group: Optional[str]
        Specifies which InferenceData group should be plotted.  Defaults to 'posterior'.
        Alternative values include 'prior' and any other strings used as dataset keys in the
        InferenceData.
    data_labels : Optional[List[str]]
        List with names for the datasets passed as "data." Useful when plotting more than one
        dataset.  Must be the same shape as the data parameter.  Defaults to None.
    var_names: Optional[List[str]]
        List of variables to plot.  If multiple datasets are supplied and var_names is not None,
        will print the same set of variables for each dataset.  Defaults to None, which results in
        all the variables being plotted.
    credible_interval : float
        Credible intervals. Should be in the interval (0, 1]. Defaults to 0.94.
    point_estimate : Optional[str]
        Plot point estimate per variable. Values should be 'mean', 'median' or None.
        Defaults to 'mean'.
    colors : Optional[Union[List[str],str]]
        List with valid matplotlib colors, one color per model. Alternative a string can be passed.
        If the string is `cycle`, it will automatically choose a color per model from matplotlib's
        cycle. If a single color is passed, e.g. 'k', 'C2' or 'red' this color will be used for all
        models. Defaults to `cycle`.
    outline : bool
        Use a line to draw KDEs and histograms. Default to True
    hpd_markers : str
        A valid `matplotlib.markers` like 'v', used to indicate the limits of the hpd interval.
        Defaults to empty string (no marker).
    shade : Optional[float]
        Alpha blending value for the shaded area under the curve, between 0 (no shade) and 1
        (opaque). Defaults to 0.
    bw : Optional[float]
        Bandwidth scaling factor for the KDE. Should be larger than 0. The higher this number the
        smoother the KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule
        of thumb (the default rule used by SciPy).
    figsize : Optional[Tuple[int, int]]
        Figure size. If None it will be defined automatically.
    textsize: Optional[float]
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.
    ax: axes, optional
        Matplotlib axes or Bokeh figures.
    backend: str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    backend_kwargs: bool, optional
        These are kwargs specific to the backend being used. For additional documentation
        check the plotting method of the backend.
    show : bool, optional
        Call backend show function.

    Returns
    -------
    axes : matplotlib axes or bokeh figures


    Examples
    --------
    Plot default density plot

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> centered = az.load_arviz_data('centered_eight')
        >>> non_centered = az.load_arviz_data('non_centered_eight')
        >>> az.plot_density([centered, non_centered])

    Plot subset variables by specifying variable name exactly

    .. plot::
        :context: close-figs

        >>> az.plot_density([centered, non_centered], var_names=["mu"])

    Plot a specific `az.InferenceData` group

    .. plot::
        :context: close-figs

        >>> az.plot_density([centered, non_centered], var_names=["mu"], group="prior")

    Specify credible interval

    .. plot::
        :context: close-figs

        >>> az.plot_density([centered, non_centered], var_names=["mu"], credible_interval=.5)

    Shade plots and/or remove outlines

    .. plot::
        :context: close-figs

        >>> az.plot_density([centered, non_centered], var_names=["mu"], outline=False, shade=.8)

    Specify binwidth for kernel density estimation

    .. plot::
        :context: close-figs

        >>> az.plot_density([centered, non_centered], var_names=["mu"], bw=.9)
    """
    if not isinstance(data, (list, tuple)):
        datasets = [convert_to_dataset(data, group=group)]
    else:
        datasets = [convert_to_dataset(datum, group=group) for datum in data]

    var_names = _var_names(var_names, datasets)

    if point_estimate not in ("mean", "median", None):
        raise ValueError(
            "Point estimate should be 'mean'," "median' or None, not {}".format(point_estimate)
        )

    n_data = len(datasets)

    if data_labels is None:
        if n_data > 1:
            data_labels = ["{}".format(idx) for idx in range(n_data)]
        else:
            data_labels = [""]
    elif len(data_labels) != n_data:
        raise ValueError(
            "The number of names for the models ({}) "
            "does not match the number of models ({})".format(len(data_labels), n_data)
        )

    if colors == "cycle":
        colors = [
            prop
            for _, prop in zip(
                range(n_data), cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
            )
        ]
    elif isinstance(colors, str):
        colors = [colors for _ in range(n_data)]

    if not 1 >= credible_interval > 0:
        raise ValueError("The value of credible_interval should be in the interval (0, 1]")

    to_plot = [list(xarray_var_iter(data, var_names, combined=True)) for data in datasets]
    all_labels = []
    length_plotters = []
    for plotters in to_plot:
        length_plotters.append(len(plotters))
        for var_name, selection, _ in plotters:
            label = make_label(var_name, selection)
            if label not in all_labels:
                all_labels.append(label)
    length_plotters = max(length_plotters)
    max_plots = rcParams["plot.max_subplots"]
    max_plots = length_plotters if max_plots is None else max_plots
    if length_plotters > max_plots:
        warnings.warn(
            "rcParams['plot.max_subplots'] ({max_plots}) is smaller than the number "
            "of variables to plot ({len_plotters}) in plot_density, generating only "
            "{max_plots} plots".format(max_plots=max_plots, len_plotters=length_plotters),
            SyntaxWarning,
        )
        all_labels = all_labels[:max_plots]
        to_plot = [
            [
                (var_name, selection, values)
                for var_name, selection, values in plotters
                if make_label(var_name, selection) in all_labels
            ]
            for plotters in to_plot
        ]
        length_plotters = max_plots
    rows, cols = default_grid(length_plotters, max_cols=3)

    (figsize, _, titlesize, xt_labelsize, linewidth, markersize) = _scale_fig_size(
        figsize, textsize, rows, cols
    )

    plot_density_kwargs = dict(
        ax=ax,
        all_labels=all_labels,
        to_plot=to_plot,
        colors=colors,
        bw=bw,
        figsize=figsize,
        length_plotters=length_plotters,
        rows=rows,
        cols=cols,
        titlesize=titlesize,
        xt_labelsize=xt_labelsize,
        linewidth=linewidth,
        markersize=markersize,
        credible_interval=credible_interval,
        point_estimate=point_estimate,
        hpd_markers=hpd_markers,
        outline=outline,
        shade=shade,
        n_data=n_data,
        data_labels=data_labels,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend == "bokeh":

        plot_density_kwargs["line_width"] = plot_density_kwargs.pop("linewidth")
        plot_density_kwargs.pop("titlesize")
        plot_density_kwargs.pop("xt_labelsize")
        plot_density_kwargs.pop("n_data")

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_density", "densityplot", backend)
    ax = plot(**plot_density_kwargs)
    return ax
