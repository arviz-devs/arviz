"""Forest plot."""
from ..data import convert_to_dataset
from ..labels import BaseLabeller, NoModelLabeller
from ..rcparams import rcParams
from ..utils import _var_names, get_coords
from .plot_utils import get_plotting_function


def plot_forest(
    data,
    kind="forestplot",
    model_names=None,
    var_names=None,
    filter_vars=None,
    transform=None,
    coords=None,
    combined=False,
    combine_dims=None,
    hdi_prob=None,
    rope=None,
    quartiles=True,
    ess=False,
    r_hat=False,
    colors="cycle",
    textsize=None,
    linewidth=None,
    markersize=None,
    legend=True,
    labeller=None,
    ridgeplot_alpha=None,
    ridgeplot_overlap=2,
    ridgeplot_kind="auto",
    ridgeplot_truncate=True,
    ridgeplot_quantiles=None,
    figsize=None,
    ax=None,
    backend=None,
    backend_config=None,
    backend_kwargs=None,
    show=None,
):
    r"""Forest plot to compare HDI intervals from a number of distributions.

    Generate forest or ridge plots to compare distributions from a model or list of models.
    Additionally, the function can display effective sample sizes (ess) and Rhats to visualize
    convergence diagnostics alongside the distributions.

    Parameters
    ----------
    data : InferenceData
        Any object that can be converted to an :class:`arviz.InferenceData` object
        Refer to documentation of :func:`arviz.convert_to_dataset` for details.
    kind : {"foresplot", "ridgeplot"}, default "forestplot"
        Specify the kind of plot:

        * The ``kind="forestplot"`` generates credible intervals, where the central points are the
          estimated posterior means, the thick lines are the central quartiles, and the thin lines
          represent the :math:`100\times`(`hdi_prob`)% highest density intervals.
        * The ``kind="ridgeplot"`` option generates density plots (kernel density estimate or
          histograms) in the same graph. Ridge plots can be configured to have different overlap,
          truncation bounds and quantile markers.

    model_names : list of str, optional
        List with names for the models in the list of data. Useful when plotting more that one
        dataset.
    var_names : list of str, optional
        Variables to be plotted. Prefix the variables by ``~`` when you want to exclude
        them from the plot. See :ref:`this section <common_var_names>` for usage examples.
    combine_dims : set_like of str, optional
        List of dimensions to reduce. Defaults to reducing only the "chain" and "draw" dimensions.
        See :ref:`this section <common_combine_dims>` for usage examples.
    filter_vars : {None, "like", "regex"}, default None
        If `None` (default), interpret `var_names` as the real variables names. If "like",
        interpret `var_names` as substrings of the real variables names. If "regex",
        interpret `var_names` as regular expressions on the real variables names. See
        :ref:`this section <common_filter_vars>` for usage examples.
    transform : callable, optional
        Function to transform data (defaults to None i.e.the identity function).
    coords : dict, optional
        Coordinates of ``var_names`` to be plotted. Passed to :meth:`xarray.Dataset.sel`.
        See :ref:`this section <common_coords>` for usage examples.
    combined : bool, default False
        Flag for combining multiple chains into a single chain. If False, chains will
        be plotted separately. See :ref:`this section <common_combine>` for usage examples.
    hdi_prob : float, default 0.94
        Plots highest posterior density interval for chosen percentage of density.
        See :ref:`this section <common_ hdi_prob>` for usage examples.
    rope : tuple or dictionary of tuples
        Lower and upper values of the Region of Practical Equivalence. If a list with one interval
        only is provided, the ROPE will be displayed across the y-axis. If more than one
        interval is provided the length of the list should match the number of variables.
    quartiles : bool, default True
        Flag for plotting the interquartile range, in addition to the ``hdi_prob`` intervals.
    r_hat : bool, default False
        Flag for plotting Split R-hat statistics. Requires 2 or more chains.
    ess : bool, default False
        Flag for plotting the effective sample size.
    colors : list or string, optional
        list with valid matplotlib colors, one color per model. Alternative a string can be passed.
        If the string is `cycle`, it will automatically chose a color per model from the matplotlibs
        cycle. If a single color is passed, eg 'k', 'C2', 'red' this color will be used for all
        models. Defaults to 'cycle'.
    textsize : float, optional
        Text size scaling factor for labels, titles and lines. If `None` it will be autoscaled based
        on ``figsize``.
    linewidth : int, optional
        Line width throughout. If `None` it will be autoscaled based on ``figsize``.
    markersize : int, optional
        Markersize throughout. If `None` it will be autoscaled based on ``figsize``.
    legend : bool, optional
        Show a legend with the color encoded model information.
        Defaults to True, if there are multiple models.
    labeller : Labeller, optional
        Class providing the method ``make_label_vert`` to generate the labels in the plot titles.
        Read the :ref:`label_guide` for more details and usage examples.
    ridgeplot_alpha: float, optional
        Transparency for ridgeplot fill.  If ``ridgeplot_alpha=0``, border is colored by model,
        otherwise a `black` outline is used.
    ridgeplot_overlap : float, default 2
        Overlap height for ridgeplots.
    ridgeplot_kind : string, optional
        By default ("auto") continuous variables are plotted using KDEs and discrete ones using
        histograms. To override this use "hist" to plot histograms and "density" for KDEs.
    ridgeplot_truncate : bool, default True
        Whether to truncate densities according to the value of ``hdi_prob``.
    ridgeplot_quantiles : list, optional
        Quantiles in ascending order used to segment the KDE. Use [.25, .5, .75] for quartiles.
    figsize : (float, float), optional
        Figure size. If `None`, it will be defined automatically.
    ax : axes, optional
        :class:`matplotlib.axes.Axes` or :class:`bokeh.plotting.Figure`.
    backend : {"matplotlib", "bokeh"}, default "matplotlib"
        Select plotting backend.
    backend_config : dict, optional
        Currently specifies the bounds to use for bokeh axes. Defaults to value set in ``rcParams``.
    backend_kwargs : dict, optional
        These are kwargs specific to the backend being used, passed to
        :func:`matplotlib.pyplot.subplots` or :class:`bokeh.plotting.figure`.
        For additional documentation check the plotting method of the backend.
    show : bool, optional
        Call backend show function.

    Returns
    -------
    1D ndarray of matplotlib_axes or bokeh_figures

    See Also
    --------
    plot_posterior : Plot Posterior densities in the style of John K. Kruschke's book.
    plot_density : Generate KDE plots for continuous variables and histograms for discrete ones.
    summary : Create a data frame with summary statistics.

    Examples
    --------
    Forestplot

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> non_centered_data = az.load_arviz_data('non_centered_eight')
        >>> axes = az.plot_forest(non_centered_data,
        >>>                            kind='forestplot',
        >>>                            var_names=["^the"],
        >>>                            filter_vars="regex",
        >>>                            combined=True,
        >>>                            figsize=(9, 7))
        >>> axes[0].set_title('Estimated theta for 8 schools model')

    Forestplot with multiple datasets

    .. plot::
        :context: close-figs

        >>> centered_data = az.load_arviz_data('centered_eight')
        >>> axes = az.plot_forest([non_centered_data, centered_data],
        >>>                            model_names = ["non centered eight", "centered eight"],
        >>>                            kind='forestplot',
        >>>                            var_names=["^the"],
        >>>                            filter_vars="regex",
        >>>                            combined=True,
        >>>                            figsize=(9, 7))
        >>> axes[0].set_title('Estimated theta for 8 schools models')

    Forestplot with ropes

    .. plot::
        :context: close-figs

        >>> rope = {'theta': [{'school': 'Choate', 'rope': (2, 4)}], 'mu': [{'rope': (-2, 2)}]}
        >>> axes = az.plot_forest(non_centered_data,
        >>>                            rope=rope,
        >>>                            var_names='~tau',
        >>>                            combined=True,
        >>>                            figsize=(9, 7))
        >>> axes[0].set_title('Estimated theta for 8 schools model')


    Ridgeplot

    .. plot::
        :context: close-figs

        >>> axes = az.plot_forest(non_centered_data,
        >>>                            kind='ridgeplot',
        >>>                            var_names=['theta'],
        >>>                            combined=True,
        >>>                            ridgeplot_overlap=3,
        >>>                            colors='white',
        >>>                            figsize=(9, 7))
        >>> axes[0].set_title('Estimated theta for 8 schools model')

    Ridgeplot non-truncated and with quantiles

    .. plot::
        :context: close-figs

        >>> axes = az.plot_forest(non_centered_data,
        >>>                            kind='ridgeplot',
        >>>                            var_names=['theta'],
        >>>                            combined=True,
        >>>                            ridgeplot_truncate=False,
        >>>                            ridgeplot_quantiles=[.25, .5, .75],
        >>>                            ridgeplot_overlap=0.7,
        >>>                            colors='white',
        >>>                            figsize=(9, 7))
        >>> axes[0].set_title('Estimated theta for 8 schools model')
    """
    if not isinstance(data, (list, tuple)):
        data = [data]
    if len(data) == 1:
        legend = False

    if coords is None:
        coords = {}

    if combine_dims is None:
        combine_dims = set()

    if labeller is None:
        labeller = NoModelLabeller() if legend else BaseLabeller()

    datasets = [convert_to_dataset(datum) for datum in reversed(data)]
    if transform is not None:
        datasets = [transform(dataset) for dataset in datasets]
    datasets = get_coords(
        datasets, list(reversed(coords)) if isinstance(coords, (list, tuple)) else coords
    )

    var_names = _var_names(var_names, datasets, filter_vars)

    ncols, width_ratios = 1, [3]

    if ess:
        ncols += 1
        width_ratios.append(1)

    if r_hat:
        ncols += 1
        width_ratios.append(1)

    if hdi_prob is None:
        hdi_prob = rcParams["stats.hdi_prob"]
    elif not 1 >= hdi_prob > 0:
        raise ValueError("The value of hdi_prob should be in the interval (0, 1]")

    plot_forest_kwargs = dict(
        ax=ax,
        datasets=datasets,
        var_names=var_names,
        model_names=model_names,
        combined=combined,
        combine_dims=combine_dims,
        colors=colors,
        figsize=figsize,
        width_ratios=width_ratios,
        linewidth=linewidth,
        markersize=markersize,
        kind=kind,
        ncols=ncols,
        hdi_prob=hdi_prob,
        quartiles=quartiles,
        rope=rope,
        ridgeplot_overlap=ridgeplot_overlap,
        ridgeplot_alpha=ridgeplot_alpha,
        ridgeplot_kind=ridgeplot_kind,
        ridgeplot_truncate=ridgeplot_truncate,
        ridgeplot_quantiles=ridgeplot_quantiles,
        textsize=textsize,
        legend=legend,
        labeller=labeller,
        ess=ess,
        r_hat=r_hat,
        backend_kwargs=backend_kwargs,
        backend_config=backend_config,
        show=show,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_forest", "forestplot", backend)
    axes = plot(**plot_forest_kwargs)
    return axes
