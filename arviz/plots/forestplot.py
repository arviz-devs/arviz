"""Forest plot."""
from ..data import convert_to_dataset
from .plot_utils import get_coords, get_plotting_function
from ..utils import _var_names


def plot_forest(
    data,
    kind="forestplot",
    model_names=None,
    var_names=None,
    coords=None,
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
    ridgeplot_kind="auto",
    figsize=None,
    ax=None,
    backend=None,
    backend_kwargs=None,
    show=None,
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
    coords : dict, optional
        Coordinates of var_names to be plotted. Passed to `Dataset.sel`
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
    ridgeplot_kind : string
        By default ("auto") continuous variables are plotted using KDEs and discrete ones using
        histograms. To override this use "hist" to plot histograms and "density" for KDEs
    figsize : tuple
        Figure size. If None it will be defined automatically.
    ax: axes, optional
        Matplotlib axes or bokeh figures.
    backend: str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    backend_kwargs: bool, optional
        These are kwargs specific to the backend being used. For additional documentation
        check the plotting method of the backend.
    show : bool, optional
        Call backend show function.

    Returns
    -------
    gridspec : matplotlib GridSpec or bokeh figures

    Examples
    --------
    Forestpĺot

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> non_centered_data = az.load_arviz_data('non_centered_eight')
        >>> axes = az.plot_forest(non_centered_data,
        >>>                            kind='forestplot',
        >>>                            var_names=['theta'],
        >>>                            combined=True,
        >>>                            ridgeplot_overlap=3,
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
    """
    if not isinstance(data, (list, tuple)):
        data = [data]

    if coords is None:
        coords = {}
    datasets = get_coords(
        [convert_to_dataset(datum) for datum in reversed(data)],
        list(reversed(coords)) if isinstance(coords, (list, tuple)) else coords,
    )

    var_names = _var_names(var_names, datasets)

    ncols, width_ratios = 1, [3]

    if ess:
        ncols += 1
        width_ratios.append(1)

    if r_hat:
        ncols += 1
        width_ratios.append(1)

    plot_forest_kwargs = dict(
        ax=ax,
        datasets=datasets,
        var_names=var_names,
        model_names=model_names,
        combined=combined,
        colors=colors,
        figsize=figsize,
        width_ratios=width_ratios,
        linewidth=linewidth,
        markersize=markersize,
        kind=kind,
        ncols=ncols,
        credible_interval=credible_interval,
        quartiles=quartiles,
        rope=rope,
        ridgeplot_overlap=ridgeplot_overlap,
        ridgeplot_alpha=ridgeplot_alpha,
        ridgeplot_kind=ridgeplot_kind,
        textsize=textsize,
        ess=ess,
        r_hat=r_hat,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_forest", "forestplot", backend)
    axes = plot(**plot_forest_kwargs)
    return axes
