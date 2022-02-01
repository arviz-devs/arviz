"""Prior elicitation using roulette method."""

from .backends.matplotlib import plot_roulette as plot
from ..rcparams import rcParams


def plot_roulette(
    x_min=0, x_max=10, nrows=10, ncols=10, parametrization="PyMC", figsize=None, backend=None
):
    """
    Prior elicitation for 1D distribution.

    Draw 1D distributions using a grid as input.


    Parameters
    ----------
    x_min: Optional[float]
        Minimum value for the domain of the grid and fitted distribution
    x_max: Optional[float]
        Maximum value for the domain of the grid and fitted distribution
    nrows: Optional[int]
        Number of rows for the grid. Defaults to 10.
    ncols: Optional[int]
        Number of columns for the grid. Defaults to 10.
    parametrization: Optional[str]
        Parametrization used to report the result of the fit.
        Currently only "PyMC" parametrization is supported.
    figsize: Optional[Tuple[int, int]]
        Figure size. If None it will be defined automatically.
    backend: Optional[str]
        Select plotting backend {"matplotlib", "bokeh"}. Default "matplotlib". Currently this plot
        only works with the matplotlib backend.

    Returns
    -------
    scipy.stats.distributions.rv_frozen
        Notice that the returned rv_frozen object always use the scipy parametrization,
        irrespective of the value of `parametrization` argument.
        Unlike standard rv_frozen objects this one has a name attribute

    References
    ----------
    * Morris D.E. et al. (2014) see https://doi.org/10.1016/j.envsoft.2013.10.010
    * See roulette mode http://optics.eee.nottingham.ac.uk/match/uncertainty.php
    """

    plot_roulette_args = dict(
        x_min=x_min,
        x_max=x_max,
        nrows=nrows,
        ncols=ncols,
        parametrization=parametrization,
        figsize=figsize,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    if backend == "bokeh":
        raise NotImplementedError(
            "plot_roulette is currently only supported with matplotlib backend."
        )

    rv_frozen = plot(**plot_roulette_args)

    return rv_frozen
