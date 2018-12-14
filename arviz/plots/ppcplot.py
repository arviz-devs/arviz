"""Posterior predictive plot."""
import numpy as np
from .kdeplot import plot_kde, _fast_kde
from .plot_utils import (
    xarray_var_iter,
    _scale_fig_size,
    default_grid,
    make_label,
    _create_axes_grid,
)
from ..utils import _var_names


def plot_ppc(
    data,
    kind="density",
    alpha=0.2,
    mean=True,
    figsize=None,
    textsize=None,
    data_pairs=None,
    var_names=None,
    coords=None,
    flatten=None,
    num_pp_samples=None,
    random_seed=None,
):
    """
    Plot for Posterior Predictive checks.

    Parameters
    ----------
    data : az.InferenceData object
        InferenceData object containing the observed and posterior
        predictive data.
    kind : str
        Type of plot to display (density, cumulative, or scatter). Defaults to density.
    alpha : float
        Opacity of posterior predictive density curves. Defaults to 0.2.
    mean : bool
        Whether or not to plot the mean posterior predictive distribution. Defaults to True
    figsize : tuple
        Figure size. If None it will be defined automatically.
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be
        autoscaled based on figsize.
    data_pairs : dict
        Dictionary containing relations between observed data and posterior predictive data.
        Dictionary structure:
            Key = data var_name

            Value = posterior predictive var_name
        Example: `data_pairs = {'y' : 'y_hat'}`

        If None, it will assume that the observed data and the posterior
        predictive data have the same variable name.
    var_names : list
        List of variables to be plotted. Defaults to all observed variables in the
        model if None.
    coords : dict
        Dictionary mapping dimensions to selected coordinates to be plotted.
        Dimensions without a mapping specified will include all coordinates for
        that dimension. Defaults to including all coordinates for all
        dimensions if None.
    flatten : list
        List of dimensions to flatten. Only flattens across the coordinates
        specified in the coords argument. Defaults to flattening all of the dimensions.
    num_pp_samples : int
        The number of posterior predictive samples to plot.
        It defaults to a maximum of 30 samples for `kind` = 'scatter'.
        Otherwise it defaults to all provided samples.
    random_seed : int
        Random number generator seed passed to numpy.random.seed to allow
        reproducibility of the plot. By default, no seed will be provided
        and the plot will change each call if a random sample is specified
        by `num_pp_samples`.

    Returns
    -------
    axes : matplotlib axes

    Examples
    --------
    Plot the observed data KDE overlaid on posterior predictive KDEs.

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('radon')
        >>> az.plot_ppc(data)

    Plot the overlay with empirical CDFs.

    .. plot::
        :context: close-figs

        >>> az.plot_ppc(data, kind='cumulative')

    Use the coords and flatten parameters to plot selected variable dimensions
    across multiple plots.

    .. plot::
        :context: close-figs

        >>> az.plot_ppc(data, coords={'observed_county': ['ANOKA', 'BELTRAMI']}, flatten=[])

    Plot the overlay using a stacked scatter plot that is particularly useful
    when the sample sizes are small.

    .. plot::
        :context: close-figs

        >>> az.plot_ppc(data, kind='scatter', flatten=[],
        >>>             coords={'observed_county': ['AITKIN', 'BELTRAMI']})

    Plot random posterior predictive sub-samples.

    .. plot::
        :context: close-figs

        >>> az.plot_ppc(data, num_pp_samples=30, random_seed=7)
    """
    for group in ("posterior_predictive", "observed_data"):
        if not hasattr(data, group):
            raise TypeError(
                '`data` argument must have the group "{group}" for ppcplot'.format(group=group)
            )

    if kind.lower() not in ("density", "cumulative", "scatter"):
        raise TypeError("`kind` argument must be either `density`, `cumulative`, or `scatter`")

    if data_pairs is None:
        data_pairs = {}

    observed = data.observed_data
    posterior_predictive = data.posterior_predictive

    if var_names is None:
        var_names = observed.data_vars
    var_names = _var_names(var_names)
    pp_var_names = [data_pairs.get(var, var) for var in var_names]

    if flatten is None:
        flatten = list(observed.dims.keys())

    if coords is None:
        coords = {}

    if random_seed is not None:
        np.random.seed(random_seed)

    total_pp_samples = posterior_predictive.sizes["chain"] * posterior_predictive.sizes["draw"]
    if num_pp_samples is None:
        if kind == "scatter":
            num_pp_samples = min(30, total_pp_samples)
        else:
            num_pp_samples = total_pp_samples

    if (
        not isinstance(num_pp_samples, int)
        or not num_pp_samples >= 1
        or not num_pp_samples <= total_pp_samples
    ):
        raise TypeError(
            "`num_pp_samples` must be an integer between 1 and "
            + "{limit}.".format(limit=total_pp_samples)
        )

    pp_sample_ix = np.random.choice(total_pp_samples, size=num_pp_samples, replace=False)

    for key in coords.keys():
        coords[key] = np.where(np.in1d(observed[key], coords[key]))[0]

    obs_plotters = list(
        xarray_var_iter(
            observed.isel(coords), skip_dims=set(flatten), var_names=var_names, combined=True
        )
    )
    pp_plotters = list(
        xarray_var_iter(
            posterior_predictive.isel(coords),
            var_names=pp_var_names,
            skip_dims=set(flatten),
            combined=True,
        )
    )
    length_plotters = len(obs_plotters)
    rows, cols = default_grid(length_plotters)

    (figsize, ax_labelsize, _, xt_labelsize, linewidth, markersize) = _scale_fig_size(
        figsize, textsize, rows, cols
    )

    _, axes = _create_axes_grid(length_plotters, rows, cols, figsize=figsize)

    for i, ax in enumerate(axes):
        var_name, selection, obs_vals = obs_plotters[i]
        pp_var_name, _, pp_vals = pp_plotters[i]
        dtype = observed[var_name].dtype.kind

        # flatten non-specified dimensions
        obs_vals = obs_vals.flatten()
        pp_vals = pp_vals.squeeze()
        if len(pp_vals.shape) > 2:
            pp_vals = pp_vals.reshape((pp_vals.shape[0], np.prod(pp_vals.shape[1:])))
        pp_sampled_vals = pp_vals[pp_sample_ix]

        if kind == "density":
            if dtype == "f":
                plot_kde(
                    obs_vals,
                    label="Observed {}".format(var_name),
                    plot_kwargs={"color": "k", "linewidth": linewidth, "zorder": 3},
                    fill_kwargs={"alpha": 0},
                    ax=ax,
                )
            else:
                nbins = round(len(obs_vals) ** 0.5)
                hist, bin_edges = np.histogram(obs_vals, bins=nbins, density=True)
                hist = np.concatenate((hist[:1], hist))
                ax.plot(
                    bin_edges,
                    hist,
                    label="Observed {}".format(var_name),
                    color="k",
                    linewidth=linewidth,
                    zorder=3,
                    drawstyle="steps-pre",
                )
            # run plot_kde manually with one plot call
            pp_densities = []
            for vals in pp_sampled_vals:
                vals = np.array([vals]).flatten()
                if dtype == "f":
                    pp_density, lower, upper = _fast_kde(vals)
                    pp_x = np.linspace(lower, upper, len(pp_density))
                    pp_densities.extend([pp_x, pp_density])
                else:
                    nbins = round(len(vals) ** 0.5)
                    hist, bin_edges = np.histogram(vals, bins=nbins, density=True)
                    hist = np.concatenate((hist[:1], hist))
                    pp_densities.extend([bin_edges, hist])
            plot_kwargs = {"color": "C5", "alpha": alpha, "linewidth": 0.5 * linewidth}
            if dtype == "i":
                plot_kwargs["drawstyle"] = "steps-pre"
            ax.plot(*pp_densities, **plot_kwargs)
            ax.plot([], color="C5", label="Posterior predictive {}".format(pp_var_name))
            if mean:
                if dtype == "f":
                    plot_kde(
                        pp_vals.flatten(),
                        plot_kwargs={
                            "color": "C0",
                            "linestyle": "--",
                            "linewidth": linewidth,
                            "zorder": 2,
                        },
                        label="Posterior predictive mean {}".format(pp_var_name),
                        ax=ax,
                    )
                else:
                    vals = pp_vals.flatten()
                    nbins = round(len(vals) ** 0.5)
                    hist, bin_edges = np.histogram(vals, bins=nbins, density=True)
                    hist = np.concatenate((hist[:1], hist))
                    ax.plot(
                        bin_edges,
                        hist,
                        color="C0",
                        linewidth=linewidth,
                        label="Posterior predictive mean {}".format(pp_var_name),
                        zorder=2,
                        linestyle="--",
                        drawstyle="steps-pre",
                    )
            ax.tick_params(labelsize=xt_labelsize)
            ax.set_yticks([])

        elif kind == "cumulative":
            if dtype == "f":
                ax.plot(
                    *_empirical_cdf(obs_vals),
                    color="k",
                    linewidth=linewidth,
                    label="Observed {}".format(var_name),
                    zorder=3
                )
            else:
                ax.plot(
                    *_empirical_cdf(obs_vals),
                    color="k",
                    linewidth=linewidth,
                    label="Observed {}".format(var_name),
                    drawstyle="steps-pre",
                    zorder=3
                )
            # run plot_kde manually with one plot call
            pp_densities = []
            for vals in pp_sampled_vals:
                vals = np.array([vals]).flatten()
                pp_x, pp_density = _empirical_cdf(vals)
                pp_densities.extend([pp_x, pp_density])
            if dtype == "f":
                ax.plot(*pp_densities, alpha=alpha, color="C5", linewidth=linewidth)
            else:
                ax.plot(
                    *pp_densities,
                    alpha=alpha,
                    color="C5",
                    drawstyle="steps-pre",
                    linewidth=linewidth
                )
            ax.plot([], color="C5", label="Posterior predictive {}".format(pp_var_name))
            if mean:
                if dtype == "f":
                    ax.plot(
                        *_empirical_cdf(pp_vals.flatten()),
                        color="C0",
                        linestyle="--",
                        linewidth=linewidth,
                        label="Posterior predictive mean {}".format(pp_var_name)
                    )
                else:
                    ax.plot(
                        *_empirical_cdf(pp_vals.flatten()),
                        color="C0",
                        linestyle="--",
                        linewidth=linewidth,
                        drawstyle="steps-pre",
                        label="Posterior predictive mean {}".format(pp_var_name)
                    )
            ax.set_yticks([0, 0.5, 1])

        elif kind == "scatter":
            ax.plot(
                obs_vals,
                np.zeros_like(obs_vals),
                "o",
                color="C0",
                markersize=markersize,
                label="Observed {}".format(var_name),
            )

            if mean:
                if dtype == "f":
                    plot_kde(
                        pp_vals.flatten(),
                        plot_kwargs={
                            "color": "C0",
                            "linestyle": "--",
                            "linewidth": linewidth,
                            "zorder": 2,
                        },
                        label="Posterior predictive mean {}".format(pp_var_name),
                        ax=ax,
                    )
                else:
                    vals = pp_vals.flatten()
                    nbins = round(len(vals) ** 0.5)
                    hist, bin_edges = np.histogram(vals, bins=nbins, density=True)
                    hist = np.concatenate((hist[:1], hist))
                    ax.plot(
                        bin_edges,
                        hist,
                        color="C0",
                        linewidth=linewidth,
                        label="Posterior predictive mean {}".format(pp_var_name),
                        zorder=2,
                        linestyle="--",
                        drawstyle="steps-pre",
                    )

            limit = ax.get_ylim()[1] * 1.05
            y_rows = np.linspace(0, limit, num_pp_samples)
            for vals, y in zip(pp_sampled_vals, y_rows[1:]):
                vals = np.array([vals]).flatten()
                ax.plot(vals, [y] * len(vals), "o", zorder=1, color="C5", markersize=markersize)
            ax.scatter([], [], color="C5", label="Posterior predictive {}".format(pp_var_name))

            ax.set_yticks([])

        if var_name != pp_var_name:
            xlabel = "{} / {}".format(var_name, pp_var_name)
        else:
            xlabel = var_name
        ax.set_xlabel(make_label(xlabel, selection), fontsize=ax_labelsize)

        if i == 0:
            ax.legend(fontsize=xt_labelsize)
        else:
            ax.legend([])

    return axes


def _empirical_cdf(data):
    """Compute empirical cdf of a numpy array.

    Parameters
    ----------
    data : np.array
        1d array

    Returns
    -------
    np.array, np.array
        x and y coordinates for the empirical cdf of the data
    """
    return np.sort(data), np.linspace(0, 1, len(data))
