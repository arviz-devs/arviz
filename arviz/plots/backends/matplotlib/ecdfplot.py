"""Matplotlib ecdfplot."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, binom

from arviz.plots.plot_utils import _scale_fig_size
from arviz.plots.backends.matplotlib import backend_kwarg_defaults, create_axes_grid, backend_show


def plot_ecdf(
    values,
    values2,
    distribution,
    difference,
    pit,
    confidence_bands,
    granularity,
    num_trials,
    alpha,
    figsize,
    ecdf_fill,
    ax,
    show,
    backend_kwargs,
):
    """Matplotlib ecdfplot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    (figsize, _, _, _, _, _) = _scale_fig_size(figsize, None)
    backend_kwargs.setdefault("figsize", figsize)
    backend_kwargs["squeeze"] = True

    if ax is None:
        _, ax = create_axes_grid(1, backend_kwargs=backend_kwargs)

    n = len(values)

    if confidence_bands:
        if pit:
            x = np.linspace(1 / granularity, 1, granularity)
            z = x
            values = (
                distribution(values)
                if distribution
                else compute_ecdf(values2, values) / len(values2)
            )
        else:
            x = np.linspace(values[0], values[-1], granularity)
            z = distribution(x) if distribution else compute_ecdf(values2, x) / len(values2)

        gamma = get_gamma(n, z, granularity, num_trials, alpha)
        lower, higher = get_lims(gamma, n, z)

        x_coord, y_coord = [], []
        if not difference:
            for _, x_i in enumerate(x):
                f_x_i = compute_ecdf(values, x_i) / n
                x_coord.append(x_i)
                y_coord.append(f_x_i)

            plt.step(x_coord, y_coord, where="post")

            if ecdf_fill:
                plt.fill_between(x, lower / n, higher / n, color="C0", alpha=0.2)
            else:
                plt.plot(x, lower / n, x, higher / n, color="C0", alpha=0.2)
        else:
            for _, x_i in enumerate(x):
                f_x_i = compute_ecdf(values, x_i) / n - (
                    x_i
                    if pit
                    else distribution(x_i)
                    if distribution
                    else compute_ecdf(values2, x_i) / len(values2)
                )
                x_coord.append(x_i)
                y_coord.append(f_x_i)

            plt.step(x_coord, y_coord, where="post")
            if pit:
                lower = lower / n - x
                higher = higher / n - x
            else:
                lower = lower / n - (
                    distribution(x) if distribution else compute_ecdf(values2, x) / len(values2)
                )
                higher = higher / n - (
                    distribution(x) if distribution else compute_ecdf(values2, x) / len(values2)
                )

            if ecdf_fill:
                plt.fill_between(x, lower, higher, color="C0", alpha=0.2)
            else:
                plt.plot(x, lower, x, higher, color="C0", alpha=0.2)

    else:
        if pit:
            x = np.linspace(1 / granularity, 1, granularity)
            values = distribution(values)
        else:
            x = np.linspace(values[0], values[-1], granularity)

        x_coord, y_coord = [], []
        if not difference:
            for _, x_i in enumerate(x):
                f_x_i = compute_ecdf(values, x_i) / n
                x_coord.append(x_i)
                y_coord.append(f_x_i)

            plt.step(x_coord, y_coord, where="post")
        else:
            for _, x_i in enumerate(x):
                f_x_i = compute_ecdf(values, x_i) / n - (
                    x_i
                    if pit
                    else distribution(x_i)
                    if distribution
                    else compute_ecdf(values2, x_i) / len(values2)
                )
                x_coord.append(x_i)
                y_coord.append(f_x_i)

            plt.step(x_coord, y_coord, where="post")

    if backend_show(show):
        plt.show()

    return ax


def compute_ecdf(sample, z):
    """This function computes the ecdf value at the evaluation point or a
    sorted set of evaluation points
    """
    if not isinstance(z, np.ndarray):
        ## if z is just an instance then use Binary search
        left, right = 0, len(sample) - 1
        while left <= right:
            mid = int((left + right) / 2)
            if sample[mid] > z:
                right = mid - 1
            else:
                left = mid + 1
        return left
    else:
        ## if z is a list then follow this approach
        f_z = []
        u_idx = 0
        for _, z_i in enumerate(z):
            # print(type(z_i), z, type(sample))
            while u_idx < len(sample) and sample[u_idx] < z_i:
                u_idx += 1
            f_z.append(u_idx)

        return np.array(f_z)


def get_gamma(n, z, granularity=None, num_trials=1000, alpha=0.95):
    """This function simulates an adjusted value of gamma to account for multiplicity
    when forming an 1-alpha level confidence envelope for the ECDF of a sample
    """
    if granularity is None:
        granularity = n
    gamma = []
    alpha = 1 - alpha
    for _ in range(num_trials):
        unif_samples = uniform.rvs(0, 1, n)
        unif_samples = np.sort(unif_samples)
        gamma_m = 1000
        ## Can compute ecdf for all the z together or one at a time.
        f_z = compute_ecdf(unif_samples, z)
        for i in range(granularity):
            curr = min(binom.cdf(f_z[i], n, z[i]), 1 - binom.cdf(f_z[i] - 1, n, z[i]))
            gamma_m = min(2 * curr, gamma_m)
        gamma.append(gamma_m)
    return np.quantile(gamma, alpha)


def get_lims(gamma, n, z):
    """This function computes the simultaneous alpha level confidence bands"""
    lower = binom.ppf(gamma / 2, n, z)
    upper = binom.ppf(1 - gamma / 2, n, z)
    return lower, upper
