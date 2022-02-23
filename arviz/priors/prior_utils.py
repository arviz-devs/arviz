import numpy as np
from scipy import stats
from scipy.optimize import least_squares


dist_dict = {
    "beta": ("alpha", "beta"),
    "exponential": ("lam",),
    "gamma": ("alpha", "beta"),
    "lognormal": ("mu", "sigma"),
    "normal": ("mu", "sigma"),
    "student": ("nu", "mu", "sigma"),
}  # This names could be different for different "parametrizations"


def get_parametrization(name, a, b, extra, dist_dict, parametrization):
    if parametrization == "pymc":
        if name == "gamma":
            title = f"{name}({dist_dict[name][0]}={a:.2f}, {dist_dict[name][1]}={1/b:.2f})"
        elif name == "exponential":
            title = f"{name}({dist_dict[name][0]}={1/a:.2f})"
        elif name == "lognormal":
            title = f"{name}({dist_dict[name][0]}={a:.2f}, {dist_dict[name][1]}={b:.2f})"
        elif name in ["normal", "beta"]:
            title = f"{name}({dist_dict[name][0]}={a:.2f}, {dist_dict[name][1]}={b:.2f})"
        elif name == "student":
            title = f"{name}({dist_dict[name][0]}={extra:.2f}, {dist_dict[name][1]}={a:.2f}, {dist_dict[name][2]}={b:.2f})"
    elif parametrization == "scipy":
        if name in ["gamma", "lognormal", "normal"]:
            title = f"{name}({dist_dict[name][0]}={a:.2f}, {dist_dict[name][1]}={b:.2f})"
        elif name == "exponential":
            title = f"{name}({dist_dict[name][0]}={a:.2f})"
        elif name == "student":
            title = f"{name}({dist_dict[name][0]}={extra:.2f}, {dist_dict[name][1]}={a:.2f}, {dist_dict[name][2]}={b:.2f})"
    return title


def check_boundaries(name, lower, upper):
    DOMAIN_ERROR = f"The provided boundaries are outside the domain of the {name} distribution"
    if name == "beta":
        if lower == 0 and upper == 1:
            raise ValueError(
                "Given the provided boundaries, mass will be always 1. Please provide other values"
            )
        if lower < 0 or upper > 1:
            raise ValueError(DOMAIN_ERROR)
    elif name in ["exponential", "gamma", "lognormal"]:
        if lower < 0:
            raise ValueError(DOMAIN_ERROR)


def relative_error(rv_frozen, upper, lower, requiered_mass):
    computed_mass = rv_frozen.cdf(upper) - rv_frozen.cdf(lower)
    return (computed_mass - requiered_mass) / requiered_mass * 100


def sane_scipy(dist, a, b, extra=None):
    dist_name = dist.name
    if dist_name in ["norm", "beta"]:
        rv_frozen = dist(a, b)
    elif dist_name == "gamma":
        rv_frozen = dist(a=a, scale=b)
    elif dist_name == "lognorm":
        rv_frozen = dist(b, scale=np.exp(a))
    elif dist_name == "expon":
        rv_frozen = dist(scale=a)
    elif dist_name == "t":
        rv_frozen = dist(df=extra, loc=a, scale=b)

    rv_frozen.name = dist_name
    return rv_frozen


def compute_xvals(rv_frozen):
    if np.isfinite(rv_frozen.a):
        lq = rv_frozen.a
    else:
        lq = 0.001

    if np.isfinite(rv_frozen.b):
        uq = rv_frozen.b
    else:
        uq = 0.999

    x = np.linspace(rv_frozen.ppf(lq), rv_frozen.ppf(uq), 1000)
    return x


def func(params, dist, lower, upper, mass, extra=None):
    a, b = params
    rv_frozen = sane_scipy(dist, a, b, extra)
    cdf0 = rv_frozen.cdf(lower)
    cdf1 = rv_frozen.cdf(upper)
    cdf_loss = (cdf1 - cdf0) - mass
    return cdf_loss


def get_normal(lower, upper, mass):
    mu_init = (lower + upper) / 2
    sigma_init = ((upper - lower) / 4) / mass
    opt = least_squares(func, x0=(mu_init, sigma_init), args=(stats.norm, lower, upper, mass))
    return opt
