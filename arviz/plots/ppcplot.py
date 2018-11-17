"""Posterior predictive plot."""
import numpy as np
from .kdeplot import plot_kde, _fast_kde
from .plot_utils import (
    xarray_var_iter,
    _scale_fig_size,
    default_grid,
    make_label,
    _create_axes_grid,
    get_coords,
)
from ..utils import _var_names


def plot_ppc(
    data, kind="density", alpha=0.2, mean=True, figsize=None, textsize=None, data_pairs=None,
    var_names=None, coords=None, flatten=None
):
    """
    Plot for Posterior Predictive checks.

    Note that this plot will flatten out any dimensions in the posterior predictive variables.

    Parameters
    ----------
    data : az.InferenceData object
        InferenceData object containing the observed and posterior
        predictive data.
    kind : str
        Type of plot to display (density or cumulative). Defaults to density.
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

    """
    for group in ("posterior_predictive", "observed_data"):
        if not hasattr(data, group):
            raise TypeError(
                '`data` argument must have the group "{group}" for ppcplot'.format(group=group)
            )

    if kind.lower() not in ("density", "cumulative"):
        raise TypeError("`kind` argument must be either `density` or `cumulative`")

    if data_pairs is None:
        data_pairs = {}

    observed = data.observed_data
    posterior_predictive = data.posterior_predictive

    #TODO: validate inputs for new parameters
    if var_names is None:
        var_names = observed.data_vars;
    var_names = _var_names(var_names)
    pp_var_names = [data_pairs.get(var, var) for var in var_names]

    if flatten is None:
      flatten = list(observed.dims.keys())

    if coords is None:
        coords = {}

    #TODO: confirm appropriate way to index coordinates
    for key in coords.keys():
        coords[key] = np.where(np.in1d(observed[key], coords[key]))[0]

    obs_plotters = list(xarray_var_iter(observed.isel(coords),
                                        skip_dims=set(flatten),
                                        var_names=var_names,
                                        combined=True))
    pp_plotters = list(xarray_var_iter(posterior_predictive.isel(coords),
                                       var_names=pp_var_names,
                                       skip_dims=set(flatten),
                                       combined=True))
    length_plotters = len(obs_plotters)
    rows, cols = default_grid(length_plotters)

    (figsize, ax_labelsize, _, xt_labelsize, linewidth, _) = _scale_fig_size(
        figsize, textsize, rows, cols
    )

    _, axes = _create_axes_grid(length_plotters, rows, cols, figsize=figsize)

    #TODO: Fix legends?
    for i, ax in enumerate(axes):
        var_name, selection, obs_vals = obs_plotters[i]
        pp_var_name, _, pp_vals = pp_plotters[i]
        dtype = observed[var_name].dtype.kind

        # flatten non-specified dimensions
        obs_vals = obs_vals.flatten()
        pp_vals = pp_vals.squeeze()
        if len(pp_vals.shape) > 2:
            pp_vals = pp_vals.reshape((pp_vals.shape[0], np.prod(pp_vals.shape[1:])))

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
            for vals in pp_vals:
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
            if var_name != pp_var_name:
                xlabel = "{} / {}".format(var_name, pp_var_name)
            else:
                xlabel = var_name
            ax.set_xlabel(make_label(xlabel, selection), fontsize=ax_labelsize)
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
            for vals in pp_vals:
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
            if var_name != pp_var_name:
                xlabel = "{} / {}".format(var_name, pp_var_name)
            else:
                xlabel = var_name
            ax.set_xlabel(make_label(xlabel, selection), fontsize=ax_labelsize)
            ax.set_xlabel(xlabel, fontsize=ax_labelsize)
            ax.set_yticks([0, 0.5, 1])
        ax.legend(fontsize=xt_labelsize)
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
