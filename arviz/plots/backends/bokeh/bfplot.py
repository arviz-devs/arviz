"""Bokeh Bayes Factor plot."""


def plot_bf(
    ax,
    nvars,
    ngroups,
    figsize,
    dc_plotters,
    legend,
    groups,
    textsize,
    labeller,
    prior_kwargs,
    posterior_kwargs,
    observed_kwargs,
    backend_kwargs,
    show,
):
    """Bokeh Bayes Factor plot."""
    raise NotImplementedError(
        "The bokeh backend is still under development. Use matplotlib backend."
    )
