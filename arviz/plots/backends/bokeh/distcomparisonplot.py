"""Bokeh Density Comparison plot."""
import warnings


def plot_dist_comparison(
    ax,
    nvars,
    ngroups,
    figsize,
    dc_plotters,
    legend,
    groups,
    prior_kwargs,
    posterior_kwargs,
    observed_kwargs,
    backend_kwargs,
    show,
):
    """Bokeh Density Comparison plot."""
    warnings.warn(
        "The bokeh backend is still under development. Use matplotlib bakend.", UserWarning
    )
    raise
