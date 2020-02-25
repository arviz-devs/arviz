"""Utilities for plotting."""
import warnings
from typing import Dict, Any
from itertools import product, tee
import importlib
from scipy.signal import convolve, convolve2d
from scipy.sparse import coo_matrix
from scipy.signal.windows import gaussian
from scipy.stats import mode

import packaging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cbook as cbook
import xarray as xr


from ..utils import conditional_jit, _stack
from ..rcparams import rcParams
from ..stats.stats_utils import histogram as _histogram

KwargSpec = Dict[str, Any]


def make_2d(ary):
    """Convert any array into a 2d numpy array.

    In case the array is already more than 2 dimensional, will ravel the
    dimensions after the first.
    """
    dim_0, *_ = np.atleast_1d(ary).shape
    return ary.reshape(dim_0, -1, order="F")


def _scale_fig_size(figsize, textsize, rows=1, cols=1):
    """Scale figure properties according to rows and cols.

    Parameters
    ----------
    figsize : float or None
        Size of figure in inches
    textsize : float or None
        fontsize
    rows : int
        Number of rows
    cols : int
        Number of columns

    Returns
    -------
    figsize : float or None
        Size of figure in inches
    ax_labelsize : int
        fontsize for axes label
    titlesize : int
        fontsize for title
    xt_labelsize : int
        fontsize for axes ticks
    linewidth : int
        linewidth
    markersize : int
        markersize
    """
    params = mpl.rcParams
    rc_width, rc_height = tuple(params["figure.figsize"])
    rc_ax_labelsize = params["axes.labelsize"]
    rc_titlesize = params["axes.titlesize"]
    rc_xt_labelsize = params["xtick.labelsize"]
    rc_linewidth = params["lines.linewidth"]
    rc_markersize = params["lines.markersize"]
    if isinstance(rc_ax_labelsize, str):
        rc_ax_labelsize = 15
    if isinstance(rc_titlesize, str):
        rc_titlesize = 16
    if isinstance(rc_xt_labelsize, str):
        rc_xt_labelsize = 14

    if figsize is None:
        width, height = rc_width, rc_height
        sff = 1 if (rows == cols == 1) else 1.15
        width = width * cols * sff
        height = height * rows * sff
    else:
        width, height = figsize

    if textsize is not None:
        scale_factor = textsize / rc_xt_labelsize
    elif rows == cols == 1:
        scale_factor = ((width * height) / (rc_width * rc_height)) ** 0.5
    else:
        scale_factor = 1

    ax_labelsize = rc_ax_labelsize * scale_factor
    titlesize = rc_titlesize * scale_factor
    xt_labelsize = rc_xt_labelsize * scale_factor
    linewidth = rc_linewidth * scale_factor
    markersize = rc_markersize * scale_factor

    return (width, height), ax_labelsize, titlesize, xt_labelsize, linewidth, markersize


def get_bins(values):
    """
    Automatically compute the number of bins for discrete variables.

    Parameters
    ----------
    values = numpy array
        values

    Returns
    -------
    array with the bins

    Notes
    -----
    Computes the width of the bins by taking the maximun of the Sturges and the Freedman-Diaconis
    estimators. Acording to numpy `np.histogram` this provides good all around performance.

    The Sturges is a very simplistic estimator based on the assumption of normality of the data.
    This estimator has poor performance for non-normal data, which becomes especially obvious for
    large data sets. The estimate depends only on size of the data.

    The Freedman-Diaconis rule uses interquartile range (IQR) to estimate the binwidth.
    It is considered a robusts version of the Scott rule as the IQR is less affected by outliers
    than the standard deviation. However, the IQR depends on fewer points than the standard
    deviation, so it is less accurate, especially for long tailed distributions.
    """
    x_min = values.min().astype(int)
    x_max = values.max().astype(int)

    # Sturges histogram bin estimator
    bins_sturges = (x_max - x_min) / (np.log2(values.size) + 1)

    # The Freedman-Diaconis histogram bin estimator.
    iqr = np.subtract(*np.percentile(values, [75, 25]))  # pylint: disable=assignment-from-no-return
    bins_fd = 2 * iqr * values.size ** (-1 / 3)

    width = round(np.max([1, bins_sturges, bins_fd])).astype(int)

    return np.arange(x_min, x_max + width + 1, width)


def _sturges_formula(dataset, mult=1):
    """Use Sturges' formula to determine number of bins.

    See https://en.wikipedia.org/wiki/Histogram#Sturges'_formula
    or https://doi.org/10.1080%2F01621459.1926.10502161

    Parameters
    ----------
    dataset: xarray.DataSet
        Must have the `draw` dimension

    mult: float
        Used to scale the number of bins up or down. Default is 1 for Sturges' formula.

    Returns
    -------
    int
        Number of bins to use
    """
    return int(np.ceil(mult * np.log2(dataset.draw.size)) + 1)


def default_grid(n_items, max_cols=4, min_cols=3):  # noqa: D202
    """Make a grid for subplots.

    Tries to get as close to sqrt(n_items) x sqrt(n_items) as it can,
    but allows for custom logic

    Parameters
    ----------
    n_items : int
        Number of panels required
    max_cols : int
        Maximum number of columns, inclusive
    min_cols : int
        Minimum number of columns, inclusive

    Returns
    -------
    (int, int)
        Rows and columns, so that rows * columns >= n_items
    """

    def in_bounds(val):
        return np.clip(val, min_cols, max_cols)

    if n_items <= max_cols:
        return 1, n_items
    ideal = in_bounds(round(n_items ** 0.5))

    for offset in (0, 1, -1, 2, -2):
        cols = in_bounds(ideal + offset)
        rows, extra = divmod(n_items, cols)
        if extra == 0:
            return rows, cols
    return n_items // ideal + 1, ideal


def _create_axes_grid(length_plotters, rows, cols, backend=None, backend_kwargs=None, **kwargs):
    """Create figure and axes for grids with multiple plots.

    Parameters
    ----------
    n_items : int
        Number of panels required
    rows : int
        Number of rows
    cols : int
        Number of columns
    backend: str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    backend_kwargs: dict, optional
        kwargs for backend figure.

    Returns
    -------
    fig : matplotlib figure
    ax : matplotlib axes
    """
    if backend_kwargs is None:
        backend_kwargs = {}

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    if backend == "bokeh":
        from bokeh.plotting import figure
        from .backends.bokeh import backend_kwarg_defaults

        backend_kwargs = {
            **backend_kwarg_defaults(("dpi", "plot.bokeh.figure.dpi"),),
            **backend_kwargs,
        }
        dpi = backend_kwargs.pop("dpi")
        if "figsize" in kwargs:
            figsize = kwargs["figsize"]
            backend_kwargs.setdefault("width", int(figsize[0] * dpi / cols))
            backend_kwargs.setdefault("height", int(figsize[1] * dpi / rows))

        sharex = kwargs.get("sharex", False)
        sharey = kwargs.get("sharey", False)
        fig = None
        ax = []
        extra = (rows * cols) - length_plotters
        for row in range(rows):
            row_ax = []
            for col in range(cols):
                if (row == 0) and (col == 0) and (sharex or sharey):
                    bokeh_ax = figure(**backend_kwargs)
                    row_ax.append(bokeh_ax)
                    if sharex:
                        backend_kwargs["x_range"] = bokeh_ax.x_range
                    if sharey:
                        backend_kwargs["y_range"] = bokeh_ax.y_range
                else:
                    if row * cols + (col + 1) > length_plotters:
                        row_ax.append(None)
                    else:
                        row_ax.append(figure(**backend_kwargs))
            ax.append(row_ax)
        ax = np.array(ax)
    else:
        from .backends.matplotlib import backend_kwarg_defaults

        backend_kwargs = {**backend_kwarg_defaults(), **backend_kwargs, **kwargs}
        fig, ax = plt.subplots(rows, cols, **backend_kwargs)
        ax = np.ravel(ax)
        extra = (rows * cols) - length_plotters
        if extra:
            for i in range(1, extra + 1):
                ax[-i].set_axis_off()
            ax = ax[:-extra]
    return fig, ax


def selection_to_string(selection):
    """Convert dictionary of coordinates to a string for labels.

    Parameters
    ----------
    selection : dict[Any] -> Any

    Returns
    -------
    str
        key1: value1, key2: value2, ...
    """
    return ", ".join(["{}".format(v) for _, v in selection.items()])


def make_label(var_name, selection, position="below"):
    """Consistent labelling for plots.

    Parameters
    ----------
    var_name : str
       Name of the variable

    selection : dict[Any] -> Any
        Coordinates of the variable
    position : str
        Whether to position the coordinates' label "below" (default) or "beside"
        the name of the variable

    Returns
    -------
    label
        A text representation of the label
    """
    if selection:
        sel = selection_to_string(selection)
        if position == "below":
            sep = "\n"
        elif position == "beside":
            sep = " "
    else:
        sep = sel = ""
    return "{}{}{}".format(var_name, sep, sel)


def format_sig_figs(value, default=None):
    """Get a default number of significant figures.

    Gives the integer part or `default`, whichever is bigger.

    Examples
    --------
    0.1234 --> 0.12
    1.234  --> 1.2
    12.34  --> 12
    123.4  --> 123
    """
    if default is None:
        default = 2
    if value == 0:
        return 1
    return max(int(np.log10(np.abs(value))) + 1, default)


def round_num(n, round_to):
    """
    Return a string representing a number with `round_to` significant figures.

    Parameters
    ----------
    n : float
        number to round
    round_to : int
        number of significant figures
    """
    sig_figs = format_sig_figs(n, round_to)
    return "{n:.{sig_figs}g}".format(n=n, sig_figs=sig_figs)


def purge_duplicates(list_in):
    """Remove duplicates from list while preserving order.

    Parameters
    ----------
    list_in: Iterable

    Returns
    -------
    list
        List of first occurrences in order
    """
    # Algorithm taken from Stack Overflow,
    # https://stackoverflow.com/questions/480214. Content by Georgy
    # Skorobogatov (https://stackoverflow.com/users/7851470/georgy) and
    # Markus Jarderot
    # (https://stackoverflow.com/users/22364/markus-jarderot), licensed
    # under CC-BY-SA 4.0.
    # https://creativecommons.org/licenses/by-sa/4.0/.

    seen = set()
    seen_add = seen.add
    return [x for x in list_in if not (x in seen or seen_add(x))]


def _dims(data, var_name, skip_dims):
    return [dim for dim in data[var_name].dims if dim not in skip_dims]


def _zip_dims(new_dims, vals):
    return [{k: v for k, v in zip(new_dims, prod)} for prod in product(*vals)]


def xarray_sel_iter(data, var_names=None, combined=False, skip_dims=None, reverse_selections=False):
    """Convert xarray data to an iterator over variable names and selections.

    Iterates over each var_name and all of its coordinates, returning the variable
    names and selections that allow properly obtain the data from ``data`` as desired.

    Parameters
    ----------
    data : xarray.Dataset
        Posterior data in an xarray

    var_names : iterator of strings (optional)
        Should be a subset of data.data_vars. Defaults to all of them.

    combined : bool
        Whether to combine chains or leave them separate

    skip_dims : set
        dimensions to not iterate over

    reverse_selections : bool
        Whether to reverse selections before iterating.

    Returns
    -------
    Iterator of (var_name: str, selection: dict(str, any))
        The string is the variable name, the dictionary are coordinate names to values,.
        To get the values of the variable at these coordinates, do
        ``data[var_name].sel(**selection)``.
    """
    if skip_dims is None:
        skip_dims = set()

    if combined:
        skip_dims = skip_dims.union({"chain", "draw"})
    else:
        skip_dims.add("draw")

    if var_names is None:
        if isinstance(data, xr.Dataset):
            var_names = list(data.data_vars)
        elif isinstance(data, xr.DataArray):
            var_names = [data.name]
            data = {data.name: data}

    for var_name in var_names:
        if var_name in data:
            new_dims = _dims(data, var_name, skip_dims)
            vals = [purge_duplicates(data[var_name][dim].values) for dim in new_dims]
            dims = _zip_dims(new_dims, vals)
            if reverse_selections:
                dims = reversed(dims)

            for selection in dims:
                yield var_name, selection


def xarray_var_iter(data, var_names=None, combined=False, skip_dims=None, reverse_selections=False):
    """Convert xarray data to an iterator over vectors.

    Iterates over each var_name and all of its coordinates, returning the 1d
    data.

    Parameters
    ----------
    data : xarray.Dataset
        Posterior data in an xarray

    var_names : iterator of strings (optional)
        Should be a subset of data.data_vars. Defaults to all of them.

    combined : bool
        Whether to combine chains or leave them separate

    skip_dims : set
        dimensions to not iterate over

    reverse_selections : bool
        Whether to reverse selections before iterating.

    Returns
    -------
    Iterator of (str, dict(str, any), np.array)
        The string is the variable name, the dictionary are coordinate names to values,
        and the array are the values of the variable at those coordinates.
    """
    data_to_sel = data
    if var_names is None and isinstance(data, xr.DataArray):
        data_to_sel = {data.name: data}

    for var_name, selection in xarray_sel_iter(
        data,
        var_names=var_names,
        combined=combined,
        skip_dims=skip_dims,
        reverse_selections=reverse_selections,
    ):
        yield var_name, selection, data_to_sel[var_name].sel(**selection).values


def xarray_to_ndarray(data, *, var_names=None, combined=True):
    """Take xarray data and unpacks into variables and data into list and numpy array respectively.

    Assumes that chain and draw are in coordinates

    Parameters
    ----------
    data: xarray.DataSet
        Data in an xarray from an InferenceData object. Examples include posterior or sample_stats

    var_names: iter
        Should be a subset of data.data_vars not including chain and draws. Defaults to all of them

    combined: bool
        Whether to combine chain into one array

    Returns
    -------
    var_names: list
        List of variable names
    data: np.array
        Data values
    """
    data_to_sel = data
    if var_names is None and isinstance(data, xr.DataArray):
        data_to_sel = {data.name: data}

    iterator1, iterator2 = tee(xarray_sel_iter(data, var_names=var_names, combined=combined))
    vars_and_sel = list(iterator1)
    unpacked_var_names = [make_label(var_name, selection) for var_name, selection in vars_and_sel]

    # Merge chains and variables, check dtype to be compatible with divergences data
    data0 = data_to_sel[vars_and_sel[0][0]].sel(**vars_and_sel[0][1])
    unpacked_data = np.empty((len(unpacked_var_names), data0.size), dtype=data0.dtype)
    for idx, (var_name, selection) in enumerate(iterator2):
        unpacked_data[idx] = data_to_sel[var_name].sel(**selection).values.flatten()

    return unpacked_var_names, unpacked_data


def get_coords(data, coords):
    """Subselects xarray DataSet or DataArray object to provided coords. Raises exception if fails.

    Raises
    ------
    ValueError
        If coords name are not available in data

    KeyError
        If coords dims are not available in data

    Returns
    -------
    data: xarray
        xarray.DataSet or xarray.DataArray object, same type as input
    """
    if not isinstance(data, (list, tuple)):
        try:
            return data.sel(**coords)

        except ValueError:
            invalid_coords = set(coords.keys()) - set(data.coords.keys())
            raise ValueError("Coords {} are invalid coordinate keys".format(invalid_coords))

        except KeyError as err:
            raise KeyError(
                (
                    "Coords should follow mapping format {{coord_name:[dim1, dim2]}}. "
                    "Check that coords structure is correct and"
                    " dimensions are valid. {}"
                ).format(err)
            )
    if not isinstance(coords, (list, tuple)):
        coords = [coords] * len(data)
    data_subset = []
    for idx, (datum, coords_dict) in enumerate(zip(data, coords)):
        try:
            data_subset.append(get_coords(datum, coords_dict))
        except ValueError as err:
            raise ValueError("Error in data[{}]: {}".format(idx, err))
        except KeyError as err:
            raise KeyError("Error in data[{}]: {}".format(idx, err))
    return data_subset


def color_from_dim(dataarray, dim_name):
    """Return colors and color mapping of a DataArray using coord values as color code.

    Parameters
    ----------
    dataarray : xarray.DataArray
    dim_name : str
        dimension whose coordinates will be used as color code.

    Returns
    -------
    colors : array of floats
        Array of colors (as floats for use with a cmap) for each element in the dataarray.
    color_mapping : mapping coord_value -> float
        Mapping from coord values to corresponding color
    """
    present_dims = dataarray.dims
    coord_values = dataarray[dim_name].values
    unique_coords = set(coord_values)
    color_mapping = {coord: num / len(unique_coords) for num, coord in enumerate(unique_coords)}
    if len(present_dims) > 1:
        multi_coords = dataarray.coords.to_index()
        coord_idx = present_dims.index(dim_name)
        colors = [color_mapping[coord[coord_idx]] for coord in multi_coords]
    else:
        colors = [color_mapping[coord] for coord in coord_values]
    return colors, color_mapping


def format_coords_as_labels(dataarray, skip_dims=None):
    """Format 1d or multi-d dataarray coords as strings.

    Parameters
    ----------
    dataarray : xarray.DataArray
        DataArray whose coordinates will be converted to labels.
    skip_dims : str of list_like, optional
        Dimensions whose values should not be included in the labels
    """
    if skip_dims is None:
        coord_labels = dataarray.coords.to_index()
    else:
        coord_labels = dataarray.coords.to_index().droplevel(skip_dims).drop_duplicates()
    coord_labels = coord_labels.values
    if isinstance(coord_labels[0], tuple):
        fmt = ", ".join(["{}" for _ in coord_labels[0]])
        coord_labels[:] = [fmt.format(*x) for x in coord_labels]
    else:
        coord_labels[:] = ["{}".format(s) for s in coord_labels]
    return coord_labels


def set_xticklabels(ax, coord_labels):
    """Set xticklabels to label list using Matplotlib default formatter."""
    ax.xaxis.get_major_locator().set_params(nbins=9, steps=[1, 2, 5, 10])
    xticks = ax.get_xticks().astype(np.int64)
    xticks = xticks[(xticks >= 0) & (xticks < len(coord_labels))]
    if len(xticks) > len(coord_labels):
        ax.set_xticks(np.arange(len(coord_labels)))
        ax.set_xticklabels(coord_labels)
    else:
        ax.set_xticks(xticks)
        ax.set_xticklabels(coord_labels[xticks])


def filter_plotters_list(plotters, plot_kind):
    """Cut list of plotters so that it is at most of lenght "plot.max_subplots"."""
    max_plots = rcParams["plot.max_subplots"]
    max_plots = len(plotters) if max_plots is None else max_plots
    if len(plotters) > max_plots:
        warnings.warn(
            "rcParams['plot.max_subplots'] ({max_plots}) is smaller than the number "
            "of variables to plot ({len_plotters}) in {plot_kind}, generating only "
            "{max_plots} plots".format(
                max_plots=max_plots, len_plotters=len(plotters), plot_kind=plot_kind
            ),
            UserWarning,
        )
        return plotters[:max_plots]
    return plotters


def get_plotting_function(plot_name, plot_module, backend):
    """Return plotting function for correct backend."""
    _backend = {
        "mpl": "matplotlib",
        "bokeh": "bokeh",
        "matplotlib": "matplotlib",
    }

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    try:
        backend = _backend[backend]
    except KeyError:
        raise KeyError(
            "Backend {} is not implemented. Try backend in {}".format(
                backend, set(_backend.values())
            )
        )

    if backend == "bokeh":
        try:
            import bokeh

            assert packaging.version.parse(bokeh.__version__) >= packaging.version.parse("1.4.0")

        except (ImportError, AssertionError):
            raise ImportError(
                "'bokeh' backend needs Bokeh (1.4.0+) installed." " Please upgrade or install"
            )

    # Perform import of plotting method
    # TODO: Convert module import to top level for all plots
    module = importlib.import_module(
        "arviz.plots.backends.{backend}.{plot_module}".format(
            backend=backend, plot_module=plot_module
        )
    )

    plotting_method = getattr(module, plot_name)

    return plotting_method


def calculate_point_estimate(point_estimate, values, bw=4.5):
    """Validate and calculate the point estimate.

    Parameters
    ----------
    point_estimate : Optional[str]
        Plot point estimate per variable. Values should be 'mean', 'median', 'mode' or None.
        Defaults to 'auto' i.e. it falls back to default set in rcParams.
    values : 1-d array
    bw : float
        Bandwidth scaling factor. Should be larger than 0. The higher this number the smoother the
        KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule of thumb
        (the default used rule by SciPy).

    Returns
    -------
    point_value : float
        best estimate of data distribution
    """
    point_value = None
    if point_estimate == "auto":
        point_estimate = rcParams["plot.point_estimate"]
    elif point_estimate not in ("mean", "median", "mode", None):
        raise ValueError(
            "Point estimate should be 'mean', 'median', 'mode' or None, not {}".format(
                point_estimate
            )
        )
    if point_estimate == "mean":
        point_value = values.mean()
    elif point_estimate == "mode":
        if isinstance(values[0], float):
            density, lower, upper = _fast_kde(values, bw=bw)
            x = np.linspace(lower, upper, len(density))
            point_value = x[np.argmax(density)]
        else:
            point_value = mode(values)[0][0]
    elif point_estimate == "median":
        point_value = np.median(values)

    return point_value


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

    hist, bin_edges = _histogram(x, bins=n_bins, range=(xmin, xmax))
    grid = hist / (hist.sum() * np.diff(bin_edges))

    # _, grid, _ = histogram(x, n_bins, range_hist=(xmin, xmax))

    scotts_factor = len_x ** (-0.2)
    kern_nx = int(scotts_factor * 2 * np.pi * log_len_x)
    kernel = gaussian(kern_nx, scotts_factor * log_len_x)

    npad = min(n_bins, 2 * kern_nx)
    grid = np.concatenate([grid[npad:0:-1], grid, grid[n_bins : n_bins - npad : -1]])
    density = convolve(grid, kernel, mode="same", method="direct")[npad : npad + n_bins]
    norm_factor = (2 * np.pi * log_len_x ** 2 * scotts_factor ** 2) ** 0.5

    density /= norm_factor

    if cumulative:
        density = density.cumsum() / density.sum()

    return density, xmin, xmax


def _cov_1d(x):
    x = x - x.mean(axis=0)
    ddof = x.shape[0] - 1
    return np.dot(x.T, x.conj()) / ddof


def _cov(data):
    if data.ndim == 1:
        return _cov_1d(data)
    elif data.ndim == 2:
        x = data.astype(float)
        avg, _ = np.average(x, axis=1, weights=None, returned=True)
        ddof = x.shape[1] - 1
        if ddof <= 0:
            warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=2)
            ddof = 0.0
        x -= avg[:, None]
        prod = _dot(x, x.T.conj())
        prod *= np.true_divide(1, ddof)
        prod = prod.squeeze()
        prod += 1e-6 * np.eye(prod.shape[0])
        return prod
    else:
        raise ValueError("{} dimension arrays are not supported".format(data.ndim))


@conditional_jit(cache=True)
def _dot(x, y):
    return np.dot(x, y)


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

    xyi = _stack(x, y).T
    xyi -= [xmin, ymin]
    xyi /= [d_x, d_y]
    xyi = np.floor(xyi, xyi).T

    scotts_factor = len_x ** (-1 / 6)
    cov = _cov(xyi)
    std_devs = np.diag(cov ** 0.5)
    kern_nx, kern_ny = np.round(scotts_factor * 2 * np.pi * std_devs)

    inv_cov = np.linalg.inv(cov * scotts_factor ** 2)

    x_x = np.arange(kern_nx) - kern_nx / 2
    y_y = np.arange(kern_ny) - kern_ny / 2
    x_x, y_y = np.meshgrid(x_x, y_y)

    kernel = _stack(x_x.flatten(), y_y.flatten())
    kernel = _dot(inv_cov, kernel) * kernel
    kernel = np.exp(-kernel.sum(axis=0) / 2)
    kernel = kernel.reshape((int(kern_ny), int(kern_nx)))

    boundary = "wrap" if circular else "symm"

    grid = coo_matrix((weights, xyi), shape=(n_x, n_y)).toarray()
    grid = convolve2d(grid, kernel, mode="same", boundary=boundary)

    norm_factor = np.linalg.det(2 * np.pi * cov * scotts_factor ** 2)
    norm_factor = len_x * d_x * d_y * norm_factor ** 0.5

    grid /= norm_factor

    return grid, xmin, xmax, ymin, ymax


def matplotlib_kwarg_dealiaser(args, kind, backend="matplotlib"):
    """De-aliase the kwargs passed to plots."""
    if args is None:
        return {}
    matplotlib_kwarg_dealiaser_dict = {
        "scatter": mpl.collections.PathCollection,
        "plot": mpl.lines.Line2D,
        "hist": mpl.patches.Patch,
        "hexbin": mpl.collections.PolyCollection,
        "hlines": mpl.collections.LineCollection,
        "text": mpl.text.Text,
        "contour": mpl.contour.ContourSet,
        "pcolormesh": mpl.collections.QuadMesh,
    }
    if backend == "matplotlib":
        return cbook.normalize_kwargs(
            args, getattr(matplotlib_kwarg_dealiaser_dict[kind], "_alias_map", {})
        )
    return args
