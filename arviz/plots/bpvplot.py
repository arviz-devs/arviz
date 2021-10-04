"""Bayesian p-value Posterior/Prior predictive plot."""
import numpy as np

from ..labels import BaseLabeller
from ..rcparams import rcParams
from ..utils import _var_names
from .plot_utils import default_grid, filter_plotters_list, get_plotting_function
from ..sel_utils import xarray_var_iter


def plot_bpv(
    data,
    kind="u_value",
    t_stat="median",
    bpv=True,
    plot_mean=True,
    reference="analytical",
    mse=False,
    n_ref=100,
    hdi_prob=0.94,
    color="C0",
    grid=None,
    figsize=None,
    textsize=None,
    labeller=None,
    data_pairs=None,
    var_names=None,
    filter_vars=None,
    coords=None,
    flatten=None,
    flatten_pp=None,
    ax=None,
    backend=None,
    plot_ref_kwargs=None,
    backend_kwargs=None,
    group="posterior",
    show=None,
):
    """
    Plot Bayesian p-value for observed data and Posterior/Prior predictive.

    Parameters
    ----------
    data : az.InferenceData object
        InferenceData object containing the observed and posterior/prior predictive data.
    kind : str
        Type of plot to display ("p_value", "u_value", "t_stat"). Defaults to u_value.
        For "p_value" we compute p := p(y* ≤ y | y). This is the probability of the data y being
        larger or equal than the predicted data y*. The ideal value is 0.5 (half the predictions
        below and half above the data).
        For "u_value" we compute pi := p(yi* ≤ yi | y). i.e. like a p_value but per observation yi.
        This is also known as marginal p_value. The ideal distribution is uniform. This is similar
        to the LOO-pit calculation/plot, the difference is than in LOO-pit plot we compute
        pi = p(yi* r ≤ yi | y-i ), where y-i, is all other data except yi.
        For "t_stat" we compute := p(T(y)* ≤ T(y) | y) where T is any T statistic. See t_stat
        argument below for details of available options.
    t_stat : str, float, or callable
        T statistics to compute from the observations and predictive distributions. Allowed strings
        are "mean", "median" or "std". Defaults to "median". Alternative a quantile can be passed
        as a float (or str) in the interval (0, 1). Finally a user defined function is also
        acepted, see examples section for details.
    bpv : bool
        If True (default) add the bayesian p_value to the legend when kind = t_stat.
    plot_mean : bool
        Whether or not to plot the mean T statistic. Defaults to True.
    reference : str
        How to compute the distributions used as reference for u_values or p_values. Allowed values
        are "analytical" (default) and "samples". Use `None` to do not plot any reference.
        Defaults to "samples".
    mse :bool
        Show scaled mean square error between uniform distribution and marginal p_value
        distribution. Defaults to False.
    n_ref : int, optional
        Number of reference distributions to sample when `reference=samples`. Defaults to 100.
    hdi_prob: float, optional
        Probability for the highest density interval for the analytical reference distribution when
        computing u_values. Should be in the interval (0, 1]. Defaults to
        0.94.
    color : str
        Matplotlib color
    grid : tuple
        Number of rows and columns. Defaults to None, the rows and columns are
        automatically inferred.
    figsize : tuple
        Figure size. If None it will be defined automatically.
    textsize : float
        Text size scaling factor for labels, titles and lines. If None it will be
        autoscaled based on figsize.
    data_pairs : dict
        Dictionary containing relations between observed data and posterior/prior predictive data.
        Dictionary structure:

        - key = data var_name
        - value = posterior/prior predictive var_name

        For example, `data_pairs = {'y' : 'y_hat'}`
        If None, it will assume that the observed data and the posterior/prior
        predictive data have the same variable name.
    labeller : labeller instance, optional
        Class providing the method `make_pp_label` to generate the labels in the plot titles.
        Read the :ref:`label_guide` for more details and usage examples.
    var_names : list of variable names
        Variables to be plotted, if `None` all variable are plotted. Prefix the variables by `~`
        when you want to exclude them from the plot.
    filter_vars : {None, "like", "regex"}, optional, default=None
        If `None` (default), interpret var_names as the real variables names. If "like",
        interpret var_names as substrings of the real variables names. If "regex",
        interpret var_names as regular expressions on the real variables names. A la
        `pandas.filter`.
    coords : dict
        Dictionary mapping dimensions to selected coordinates to be plotted.
        Dimensions without a mapping specified will include all coordinates for
        that dimension. Defaults to including all coordinates for all
        dimensions if None.
    flatten : list
        List of dimensions to flatten in observed_data. Only flattens across the coordinates
        specified in the coords argument. Defaults to flattening all of the dimensions.
    flatten_pp : list
        List of dimensions to flatten in posterior_predictive/prior_predictive. Only flattens
        across the coordinates specified in the coords argument. Defaults to flattening all
        of the dimensions. Dimensions should match flatten excluding dimensions for data_pairs
        parameters. If flatten is defined and flatten_pp is None, then `flatten_pp=flatten`.
    legend : bool
        Add legend to figure. By default True.
    ax : numpy array-like of matplotlib axes or bokeh figures, optional
        A 2D array of locations into which to plot the densities. If not supplied, Arviz will create
        its own array of plot areas (and return it).
    backend : str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    plot_ref_kwargs :  dict, optional
        Extra keyword arguments to control how reference is represented. Passed to `plt.plot` or
        `plt.axhspan`(when `kind=u_value` and `reference=analytical`).
    backend_kwargs : bool, optional
        These are kwargs specific to the backend being used. For additional documentation
        check the plotting method of the backend.
    group : {"prior", "posterior"}, optional
        Specifies which InferenceData group should be plotted. Defaults to 'posterior'.
        Other value can be 'prior'.
    show : bool, optional
        Call backend show function.

    Returns
    -------
    axes: matplotlib axes or bokeh figures

    References
    ----------
    * Gelman et al. (2013) see http://www.stat.columbia.edu/~gelman/book/ pages 151-153 for details

    Examples
    --------
    Plot Bayesian p_values.

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data("regression1d")
        >>> az.plot_bpv(data, kind="p_value")

    Plot custom t statistic comparison.

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data("regression1d")
        >>> az.plot_bpv(data, kind="t_stat", t_stat=lambda x:np.percentile(x, q=50, axis=-1))
    """
    if group not in ("posterior", "prior"):
        raise TypeError("`group` argument must be either `posterior` or `prior`")

    for groups in (f"{group}_predictive", "observed_data"):
        if not hasattr(data, groups):
            raise TypeError(f'`data` argument must have the group "{groups}"')

    if kind.lower() not in ("t_stat", "u_value", "p_value"):
        raise TypeError("`kind` argument must be either `t_stat`, `u_value`, or `p_value`")

    if reference is not None:
        if reference.lower() not in ("analytical", "samples"):
            raise TypeError(
                "`reference` argument must be either `analytical`, `samples`, or `None`"
            )

    if hdi_prob is None:
        hdi_prob = rcParams["stats.hdi_prob"]
    else:
        if not 1 >= hdi_prob > 0:
            raise ValueError("The value of hdi_prob should be in the interval (0, 1]")

    if data_pairs is None:
        data_pairs = {}

    if labeller is None:
        labeller = BaseLabeller()

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    observed = data.observed_data

    if group == "posterior":
        predictive_dataset = data.posterior_predictive
    elif group == "prior":
        predictive_dataset = data.prior_predictive

    if var_names is None:
        var_names = list(observed.data_vars)
    var_names = _var_names(var_names, observed, filter_vars)
    pp_var_names = [data_pairs.get(var, var) for var in var_names]
    pp_var_names = _var_names(pp_var_names, predictive_dataset, filter_vars)

    if flatten_pp is None and flatten is None:
        flatten_pp = list(predictive_dataset.dims.keys())
    elif flatten_pp is None:
        flatten_pp = flatten
    if flatten is None:
        flatten = list(observed.dims.keys())

    if coords is None:
        coords = {}

    total_pp_samples = predictive_dataset.sizes["chain"] * predictive_dataset.sizes["draw"]

    for key in coords.keys():
        coords[key] = np.where(np.in1d(observed[key], coords[key]))[0]

    obs_plotters = filter_plotters_list(
        list(
            xarray_var_iter(
                observed.isel(coords), skip_dims=set(flatten), var_names=var_names, combined=True
            )
        ),
        "plot_t_stats",
    )
    length_plotters = len(obs_plotters)
    pp_plotters = [
        tup
        for _, tup in zip(
            range(length_plotters),
            xarray_var_iter(
                predictive_dataset.isel(coords),
                var_names=pp_var_names,
                skip_dims=set(flatten_pp),
                combined=True,
            ),
        )
    ]
    rows, cols = default_grid(length_plotters, grid=grid)

    bpvplot_kwargs = dict(
        ax=ax,
        length_plotters=length_plotters,
        rows=rows,
        cols=cols,
        obs_plotters=obs_plotters,
        pp_plotters=pp_plotters,
        total_pp_samples=total_pp_samples,
        kind=kind,
        bpv=bpv,
        t_stat=t_stat,
        reference=reference,
        mse=mse,
        n_ref=n_ref,
        hdi_prob=hdi_prob,
        plot_mean=plot_mean,
        color=color,
        figsize=figsize,
        textsize=textsize,
        labeller=labeller,
        plot_ref_kwargs=plot_ref_kwargs,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_bpv", "bpvplot", backend)
    axes = plot(**bpvplot_kwargs)
    return axes
