"""One-dimensional kernel density estimate plots."""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gaussian, convolve, convolve2d  # pylint: disable=no-name-in-module
from scipy.sparse import coo_matrix
import xarray as xr
from ..data.inference_data import InferenceData
from ..utils import conditional_jit
from .plot_utils import _scale_fig_size


def plot_kde(
    values,
    values2=None,
    cumulative=False,
    rug=False,
    label=None,
    bw=4.5,
    quantiles=None,
    rotated=False,
    contour=True,
    fill_last=True,
    textsize=None,
    plot_kwargs=None,
    fill_kwargs=None,
    rug_kwargs=None,
    contour_kwargs=None,
    contourf_kwargs=None,
    pcolormesh_kwargs=None,
    ax=None,
    legend=True,
):
    """1D or 2D KDE plot taking into account boundary conditions.

    Parameters
    ----------
    values : array-like
        Values to plot
    values2 : array-like, optional
        Values to plot. If present, a 2D KDE will be estimated
    cumulative : bool
        If true plot the estimated cumulative distribution function. Defaults to False.
        Ignored for 2D KDE
    rug : bool
        If True adds a rugplot. Defaults to False. Ignored for 2D KDE
    label : string
        Text to include as part of the legend
    bw : float
        Bandwidth scaling factor for 1D KDE. Should be larger than 0. The higher this number the
        smoother the KDE will be. Defaults to 4.5 which is essentially the same as the Scott's
        rule of thumb (the default rule used by SciPy).
    quantiles : list
        Quantiles in ascending order used to segment the KDE. Use [.25, .5, .75] for quartiles.
        Defaults to None.
    rotated : bool
        Whether to rotate the 1D KDE plot 90 degrees.
    contour : bool
        If True plot the 2D KDE using contours, otherwise plot a smooth 2D KDE. Defaults to True.
    fill_last : bool
        If True fill the last contour of the 2D KDE plot. Defaults to True.
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.
    plot_kwargs : dict
        Keywords passed to the pdf line of a 1D KDE.
    fill_kwargs : dict
        Keywords passed to the fill under the line (use fill_kwargs={'alpha': 0} to disable fill).
        Ignored for 2D KDE
    rug_kwargs : dict
        Keywords passed to the rug plot. Ignored if rug=False or for 2D KDE
        Use `space` keyword (float) to control the position of the rugplot. The larger this number
        the lower the rugplot.
    contour_kwargs : dict
        Keywords passed to ax.contour. Ignored for 1D KDE.
    contourf_kwargs : dict
        Keywords passed to ax.contourf. Ignored for 1D KDE.
    pcolormesh_kwargs : dict
        Keywords passed to ax.pcolormesh. Ignored for 1D KDE.
    ax : matplotlib axes
    legend : bool
        Add legend to the figure. By default True.

    Returns
    -------
    ax : matplotlib axes

    Examples
    --------
    Plot default KDE

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> non_centered = az.load_arviz_data('non_centered_eight')
        >>> mu_posterior = np.concatenate(non_centered.posterior["mu"].values)
        >>> az.plot_kde(mu_posterior)


    Plot KDE with rugplot

    .. plot::
        :context: close-figs

        >>> az.plot_kde(mu_posterior, rug=True)


    Plot a cumulative distribution

    .. plot::
        :context: close-figs

        >>> az.plot_kde(mu_posterior, cumulative=True)



    Rotate plot 90 degrees

    .. plot::
        :context: close-figs

        >>> az.plot_kde(mu_posterior, rotated=True)


    Plot 2d contour KDE

    .. plot::
        :context: close-figs

        >>> tau_posterior = np.concatenate(non_centered.posterior["tau"].values)
        >>> az.plot_kde(mu_posterior, values2=tau_posterior)

    Remove fill for last contour in 2d KDE

    .. plot::
        :context: close-figs

        >>> az.plot_kde(mu_posterior, values2=tau_posterior, fill_last=False)

    Plot 2d smooth KDE

    .. plot::
        :context: close-figs

        >>> az.plot_kde(mu_posterior, values2=tau_posterior, contour=False)

    """
    if ax is None:
        ax = plt.gca()

    figsize = ax.get_figure().get_size_inches()

    figsize, *_, xt_labelsize, linewidth, markersize = _scale_fig_size(figsize, textsize, 1, 1)

    if isinstance(values, xr.Dataset):
        raise ValueError(
            "Xarray dataset object detected.Use plot_posterior, plot_density, plot_joint"
            "or plot_pair instead of plot_kde"
        )
    if isinstance(values, InferenceData):
        raise ValueError(" Inference Data object detected. Use plot_posterior instead of plot_kde")

    if values2 is None:
        if plot_kwargs is None:
            plot_kwargs = {}
        plot_kwargs.setdefault("color", "C0")

        default_color = plot_kwargs.get("color")

        if fill_kwargs is None:
            fill_kwargs = {}

        fill_kwargs.setdefault("color", default_color)

        if rug_kwargs is None:
            rug_kwargs = {}
        rug_kwargs.setdefault("marker", "_" if rotated else "|")
        rug_kwargs.setdefault("linestyle", "None")
        rug_kwargs.setdefault("color", default_color)
        rug_kwargs.setdefault("space", 0.2)

        plot_kwargs.setdefault("linewidth", linewidth)
        rug_kwargs.setdefault("markersize", 2 * markersize)

        density, lower, upper = _fast_kde(values, cumulative, bw)

        rug_space = max(density) * rug_kwargs.pop("space")

        x = np.linspace(lower, upper, len(density))

        if cumulative:
            density_q = density
        else:
            density_q = density.cumsum() / density.sum()
        fill_func = ax.fill_between
        fill_x, fill_y = x, density
        if rotated:
            x, density = density, x
            fill_func = ax.fill_betweenx

        ax.tick_params(labelsize=xt_labelsize)

        if rotated:
            ax.set_xlim(0, auto=True)
            rug_x, rug_y = np.zeros_like(values) - rug_space, values
        else:
            ax.set_ylim(0, auto=True)
            rug_x, rug_y = values, np.zeros_like(values) - rug_space

        if rug:
            ax.plot(rug_x, rug_y, **rug_kwargs)

        if quantiles is not None:
            fill_kwargs.setdefault("alpha", 0.75)

            idx = [np.sum(density_q < quant) for quant in quantiles]

            fill_func(
                fill_x,
                fill_y,
                where=np.isin(fill_x, fill_x[idx], invert=True, assume_unique=True),
                **fill_kwargs
            )
        else:
            fill_kwargs.setdefault("alpha", 0)
            if fill_kwargs.get("alpha") == 0:
                ax.plot(x, density, label=label, **plot_kwargs)
                fill_func(fill_x, fill_y, **fill_kwargs)
            else:
                ax.plot(x, density, **plot_kwargs)
                fill_func(fill_x, fill_y, label=label, **fill_kwargs)
        if legend and label:
            ax.legend()
    else:
        if contour_kwargs is None:
            contour_kwargs = {}
        contour_kwargs.setdefault("colors", "0.5")
        if contourf_kwargs is None:
            contourf_kwargs = {}
        if pcolormesh_kwargs is None:
            pcolormesh_kwargs = {}

        gridsize = (128, 128) if contour else (256, 256)

        density, xmin, xmax, ymin, ymax = _fast_kde_2d(values, values2, gridsize=gridsize)
        g_s = complex(gridsize[0])
        x_x, y_y = np.mgrid[xmin:xmax:g_s, ymin:ymax:g_s]

        ax.grid(False)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        if contour:
            qcfs = ax.contourf(x_x, y_y, density, antialiased=True, **contourf_kwargs)
            qcs = ax.contour(x_x, y_y, density, **contour_kwargs)
            if not fill_last:
                qcfs.collections[0].set_alpha(0)
                qcs.collections[0].set_alpha(0)
        else:
            ax.pcolormesh(x_x, y_y, density, **pcolormesh_kwargs)

    return ax


def _fast_kde(x, cumulative=False, bw=4.5, xmin=None, xmax=None):
    """Fast Fourier transform-based Gaussian kernel density estimate (KDE).

    The code was adapted from https://github.com/mfouesneau/faststats

    Parameters
    ----------
    x : Numpy array or list
    cumulative : bool
        If true, estimate the cdf instead of the pdf
    bw : float
        Bandwidth scaling factor for the KDE. Should be larger than 0. The higher this number the
        smoother the KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule
        of thumb (the default rule used by SciPy).
    xmin : float
        Manually set lower limit.
    xmax : float
        Manually set upper limit.

    Returns
    -------
    density: A gridded 1D KDE of the input points (x)
    xmin: minimum value of x
    xmax: maximum value of x
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        warnings.warn("kde plot failed, you may want to check your data")
        return np.array([np.nan]), np.nan, np.nan

    len_x = len(x)
    n_points = 200 if (xmin or xmax) is None else 500

    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)

    assert np.min(x) >= xmin
    assert np.max(x) <= xmax

    log_len_x = np.log(len_x) * bw

    n_bins = min(int(len_x ** (1 / 3) * log_len_x * 2), n_points)
    if n_bins < 2:
        warnings.warn("kde plot failed, you may want to check your data")
        return np.array([np.nan]), np.nan, np.nan

    d_x = (xmax - xmin) / (n_bins - 1)
    grid = _histogram(x, n_bins, range_hist=(xmin, xmax))

    scotts_factor = len_x ** (-0.2)
    kern_nx = int(scotts_factor * 2 * np.pi * log_len_x)
    kernel = gaussian(kern_nx, scotts_factor * log_len_x)

    npad = min(n_bins, 2 * kern_nx)
    grid = np.concatenate([grid[npad:0:-1], grid, grid[n_bins : n_bins - npad : -1]])
    density = convolve(grid, kernel, mode="same", method="direct")[npad : npad + n_bins]
    norm_factor = len_x * d_x * (2 * np.pi * log_len_x ** 2 * scotts_factor ** 2) ** 0.5

    density /= norm_factor

    if cumulative:
        density = density.cumsum() / density.sum()

    return density, xmin, xmax


@conditional_jit
def _histogram(x, n_bins, range_hist=None):
    grid, _ = np.histogram(x, bins=n_bins, range=range_hist)
    return grid


def _fast_kde_2d(x, y, gridsize=(128, 128), circular=False):
    """
    2D fft-based Gaussian kernel density estimate (KDE).

    The code was adapted from https://github.com/mfouesneau/faststats

    Parameters
    ----------
    x : Numpy array or list
    y : Numpy array or list
    gridsize : tuple
        Number of points used to discretize data. Use powers of 2 for fft optimization
    circular: bool
        If True, use circular boundaries. Defaults to False
    Returns
    -------
    grid: A gridded 2D KDE of the input points (x, y)
    xmin: minimum value of x
    xmax: maximum value of x
    ymin: minimum value of y
    ymax: maximum value of y
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    len_x = len(x)
    weights = np.ones(len_x)
    n_x, n_y = gridsize

    d_x = (xmax - xmin) / (n_x - 1)
    d_y = (ymax - ymin) / (n_y - 1)

    xyi = np.vstack((x, y)).T
    xyi -= [xmin, ymin]
    xyi /= [d_x, d_y]
    xyi = np.floor(xyi, xyi).T

    scotts_factor = len_x ** (-1 / 6)
    cov = np.cov(xyi)
    std_devs = np.diag(cov ** 0.5)
    kern_nx, kern_ny = np.round(scotts_factor * 2 * np.pi * std_devs)

    inv_cov = np.linalg.inv(cov * scotts_factor ** 2)

    x_x = np.arange(kern_nx) - kern_nx / 2
    y_y = np.arange(kern_ny) - kern_ny / 2
    x_x, y_y = np.meshgrid(x_x, y_y)

    kernel = np.vstack((x_x.flatten(), y_y.flatten()))
    kernel = np.dot(inv_cov, kernel) * kernel
    kernel = np.exp(-kernel.sum(axis=0) / 2)
    kernel = kernel.reshape((int(kern_ny), int(kern_nx)))

    boundary = "wrap" if circular else "symm"

    grid = coo_matrix((weights, xyi), shape=(n_x, n_y)).toarray()
    grid = convolve2d(grid, kernel, mode="same", boundary=boundary)

    norm_factor = np.linalg.det(2 * np.pi * cov * scotts_factor ** 2)
    norm_factor = len_x * d_x * d_y * norm_factor ** 0.5

    grid /= norm_factor

    return grid, xmin, xmax, ymin, ymax
