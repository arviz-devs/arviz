"""KDE and histogram plots for multiple variables."""
import warnings

from ..data import convert_to_dataset
from ..labels import BaseLabeller
from ..sel_utils import (
    xarray_var_iter,
)
from ..rcparams import rcParams
from ..utils import _var_names
from .plot_utils import default_grid, get_plotting_function


# pylint:disable-msg=too-many-function-args
def plot_density(
    data,
    group="posterior",
    data_labels=None,
    var_names=None,
    filter_vars=None,
    combine_dims=None,
    transform=None,
    hdi_prob=None,
    point_estimate="auto",
    colors="cycle",
    outline=True,
    hdi_markers="",
    shade=0.0,
    bw="default",
    circular=False,
    grid=None,
    figsize=None,
    textsize=None,
    labeller=None,
    ax=None,
    backend=None,
    backend_kwargs=None,
    show=None,
):
    r"""Generate KDE plots for continuous variables and histograms for discrete ones.

    Plots are truncated at their 100*(1-alpha)% highest density intervals. Plots are grouped per
    variable and colors assigned to models.

    Parameters
    ----------
    data : InferenceData or iterable of InferenceData
        Any object that can be converted to an :class:`arviz.InferenceData` object, or an Iterator
        returning a sequence of such objects.
        Refer to documentation of :func:`arviz.convert_to_dataset` for details.
    group : {"posterior", "prior"}, default "posterior"
        Specifies which InferenceData group should be plotted. If "posterior", then the values
        in `posterior_predictive` group are compared to the ones in `observed_data`, if "prior" then
        the same comparison happens, but with the values in `prior_predictive` group.
    data_labels : list of str, default None
        List with names for the datasets passed as "data." Useful when plotting more than one
        dataset.  Must be the same shape as the data parameter.
    var_names : list of str, optional
        List of variables to plot. If multiple datasets are supplied and `var_names` is not None,
        will print the same set of variables for each dataset. Defaults to None, which results in
        all the variables being plotted.
    filter_vars : {None, "like", "regex"}, default None
        If `None` (default), interpret `var_names` as the real variables names. If "like",
        interpret `var_names` as substrings of the real variables names. If "regex",
        interpret `var_names` as regular expressions on the real variables names. See
        :ref:`this section <common_filter_vars>` for usage examples.
    combine_dims : set_like of str, optional
        List of dimensions to reduce. Defaults to reducing only the "chain" and "draw" dimensions.
        See :ref:`this section <common_combine_dims>` for usage examples.
    transform : callable
        Function to transform data (defaults to `None` i.e. the identity function).
    hdi_prob : float, default 0.94
        Probability for the highest density interval. Should be in the interval (0, 1].
        See :ref:`this section <common_hdi_prob>` for usage examples.
    point_estimate : str, optional
        Plot point estimate per variable. Values should be 'mean', 'median', 'mode' or None.
        Defaults to 'auto' i.e. it falls back to default set in ``rcParams``.
    colors : str or list of str, optional
        List with valid matplotlib colors, one color per model. Alternative a string can be passed.
        If the string is `cycle`, it will automatically choose a color per model from matplotlib's
        cycle. If a single color is passed, e.g. 'k', 'C2' or 'red' this color will be used for all
        models. Defaults to `cycle`.
    outline : bool, default True
        Use a line to draw KDEs and histograms.
    hdi_markers : str
        A valid `matplotlib.markers` like 'v', used to indicate the limits of the highest density
        interval. Defaults to empty string (no marker).
    shade : float, default 0
        Alpha blending value for the shaded area under the curve, between 0 (no shade) and 1
        (opaque).
    bw : float or str, optional
        If numeric, indicates the bandwidth and must be positive.
        If str, indicates the method to estimate the bandwidth and must be
        one of "scott", "silverman", "isj" or "experimental" when `circular` is False
        and "taylor" (for now) when `circular` is True.
        Defaults to "default" which means "experimental" when variable is not circular
        and "taylor" when it is.
    circular : bool, default False
        If True, it interprets the values passed are from a circular variable measured in radians
        and a circular KDE is used. Only valid for 1D KDE.
    grid : tuple, optional
        Number of rows and columns. Defaults to ``None``, the rows and columns are
        automatically inferred. See :ref:`this section <common_grid>` for usage examples.
    figsize : (float, float), optional
        Figure size. If `None` it will be defined automatically.
    textsize : float, optional
        Text size scaling factor for labels, titles and lines. If `None` it will be autoscaled based
        on `figsize`.
    labeller : Labeller, optional
        Class providing the method ``make_label_vert`` to generate the labels in the plot titles.
        Read the :ref:`label_guide` for more details and usage examples.
    ax : 2D array-like of matplotlib_axes or bokeh_figure, optional
        A 2D array of locations into which to plot the densities. If not supplied, ArviZ will create
        its own array of plot areas (and return it).
    backend : {"matplotlib", "bokeh"}, default "matplotlib"
        Select plotting backend.
    backend_kwargs : dict, optional
        These are kwargs specific to the backend being used, passed to
        :func:`matplotlib.pyplot.subplots` or :class:`bokeh.plotting.figure`.
        For additional documentation check the plotting method of the backend.
    show : bool, optional
        Call backend show function.

    Returns
    -------
    axes : 2D ndarray of matplotlib_axes or bokeh_figure

    See Also
    --------
    plot_dist : Plot distribution as histogram or kernel density estimates.
    plot_posterior : Plot Posterior densities in the style of John K. Kruschke's book.

    Examples
    --------
    Plot default density plot

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> centered = az.load_arviz_data('centered_eight')
        >>> non_centered = az.load_arviz_data('non_centered_eight')
        >>> az.plot_density([centered, non_centered])

    Plot variables in a 4x5 grid

    .. plot::
        :context: close-figs

        >>> az.plot_density([centered, non_centered], grid=(4, 5))

    Plot subset variables by specifying variable name exactly

    .. plot::
        :context: close-figs

        >>> az.plot_density([centered, non_centered], var_names=["mu"])

    Plot a specific `az.InferenceData` group

    .. plot::
        :context: close-figs

        >>> az.plot_density([centered, non_centered], var_names=["mu"], group="prior")

    Specify highest density interval

    .. plot::
        :context: close-figs

        >>> az.plot_density([centered, non_centered], var_names=["mu"], hdi_prob=.5)

    Shade plots and/or remove outlines

    .. plot::
        :context: close-figs

        >>> az.plot_density([centered, non_centered], var_names=["mu"], outline=False, shade=.8)

    Specify binwidth for kernel density estimation

    .. plot::
        :context: close-figs

        >>> az.plot_density([centered, non_centered], var_names=["mu"], bw=.9)
    """
    if isinstance(data, (list, tuple)):
        datasets = [convert_to_dataset(datum, group=group) for datum in data]
    else:
        datasets = [convert_to_dataset(data, group=group)]

    if transform is not None:
        datasets = [transform(dataset) for dataset in datasets]

    if labeller is None:
        labeller = BaseLabeller()

    var_names = _var_names(var_names, datasets, filter_vars)

    n_data = len(datasets)

    if data_labels is None:
        data_labels = [f"{idx}" for idx in range(n_data)] if n_data > 1 else [""]
    elif len(data_labels) != n_data:
        raise ValueError(
            f"The number of names for the models ({len(data_labels)}) "
            f"does not match the number of models ({n_data})"
        )

    if hdi_prob is None:
        hdi_prob = rcParams["stats.hdi_prob"]
    elif not 1 >= hdi_prob > 0:
        raise ValueError("The value of hdi_prob should be in the interval (0, 1]")

    to_plot = [
        list(xarray_var_iter(data, var_names, combined=True, skip_dims=combine_dims))
        for data in datasets
    ]
    all_labels = []
    length_plotters = []
    for plotters in to_plot:
        length_plotters.append(len(plotters))
        for var_name, selection, isel, _ in plotters:
            label = labeller.make_label_vert(var_name, selection, isel)
            if label not in all_labels:
                all_labels.append(label)
    length_plotters = len(all_labels)
    max_plots = rcParams["plot.max_subplots"]
    max_plots = length_plotters if max_plots is None else max_plots
    if length_plotters > max_plots:
        warnings.warn(
            "rcParams['plot.max_subplots'] ({max_plots}) is smaller than the number "
            "of variables to plot ({len_plotters}) in plot_density, generating only "
            "{max_plots} plots".format(max_plots=max_plots, len_plotters=length_plotters),
            UserWarning,
        )
        all_labels = all_labels[:max_plots]
        to_plot = [
            [
                (var_name, selection, values)
                for var_name, selection, isel, values in plotters
                if labeller.make_label_vert(var_name, selection, isel) in all_labels
            ]
            for plotters in to_plot
        ]
        length_plotters = max_plots
    rows, cols = default_grid(length_plotters, grid=grid, max_cols=3)

    if bw == "default":
        bw = "taylor" if circular else "experimental"

    plot_density_kwargs = dict(
        ax=ax,
        all_labels=all_labels,
        to_plot=to_plot,
        colors=colors,
        bw=bw,
        circular=circular,
        figsize=figsize,
        length_plotters=length_plotters,
        rows=rows,
        cols=cols,
        textsize=textsize,
        labeller=labeller,
        hdi_prob=hdi_prob,
        point_estimate=point_estimate,
        hdi_markers=hdi_markers,
        outline=outline,
        shade=shade,
        n_data=n_data,
        data_labels=data_labels,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_density", "densityplot", backend)
    ax = plot(**plot_density_kwargs)
    return ax
