from scipy import stats
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from .prior_utils import (
    get_parametrization,
    check_boundaries,
    relative_error,
    sane_scipy,
    compute_xvals,
    func,
    get_normal,
    dist_dict,
)


def find_prior(
    name="normal",
    lower=-1,
    upper=1,
    mass=0.90,
    extra=None,
    parametrization="pymc",
    plot=True,
    figsize=(10, 4),
    ax=None,
):
    """
    Find parameters for a given distribution with `mass` between `lower` and `upper`.

    Parameters
    ----------
    name : str
        Name of the distribution to use as prior
    lower : float
        Lower bound
    upper : float
        Upper bound
    mass: float
        Probability mass between ``lower`` and ``upper`` bounds. Defaults to 0.9
    plot: bool
        Whether to plot the distribution, and lower and upper bounds. Defautls to True.
    figsize: tuple
        size of the figure when ``plot=True``
    ax:

    Returns
    -------
    axes: matplotlib axes
    rv_frozen : scipy.stats.distributions.rv_frozen
        Notice that the returned rv_frozen object always use the scipy parametrization,
        irrespective of the value of `parametrization` argument.
        Unlike standard rv_frozen objects this one has a name attribute
    opt: scipy.optimize.OptimizeResult
        Represents the optimization result.
    """
    check_boundaries(name, lower, upper)

    opt = get_normal(lower, upper, mass)
    mu_init, sigma_init = opt["x"]

    if name == "normal":
        rv_frozen = stats.norm(mu_init, sigma_init)
        rv_frozen.name = name
        a, b = opt["x"]
    else:
        if name == "beta":
            kappa = (mu_init * (1 - mu_init) / (sigma_init) ** 2) - 1
            a = mu_init * kappa
            b = (1 - mu_init) * kappa
            dist = stats.beta

        elif name == "lognormal":
            a = np.log(mu_init ** 2 / (sigma_init ** 2 + mu_init ** 2) ** 0.5)
            b = np.log(sigma_init ** 2 / mu_init ** 2 + 1) ** 0.5
            dist = stats.lognorm

        elif name == "exponential":
            a = mu_init
            b = sigma_init
            dist = stats.expon

        elif name == "gamma":
            a = mu_init ** 2 / sigma_init ** 2
            b = sigma_init ** 2 / mu_init
            dist = stats.gamma
        elif name == "student":
            a = mu_init
            b = sigma_init
            dist = stats.t

        opt = least_squares(func, x0=(a, b), args=(dist, lower, upper, mass, extra))
        a, b = opt["x"]
        rv_frozen = sane_scipy(dist, a, b, extra)

    if plot:
        r_error = relative_error(rv_frozen, upper, lower, mass)
        x = compute_xvals(rv_frozen)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        color = next(ax._get_lines.prop_cycler)["color"]
        ax.plot([lower, upper], [0, 0], "o", color=color, alpha=0.5)
        title = get_parametrization(name, a, b, extra, dist_dict, parametrization)
        subtitle = f"relative error = {r_error:.2f}"
        ax.plot(x, rv_frozen.pdf(x), label=title + "\n" + subtitle, color=color)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_yticks([])

    return ax, rv_frozen, opt
