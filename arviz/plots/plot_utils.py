"""Utilities for plotting."""
import importlib
import warnings
from itertools import product, tee
from typing import Any, Dict

import matplotlib as mpl
import numpy as np
import packaging
import xarray as xr
from matplotlib.colors import to_hex
from scipy.stats import mode, rankdata
from scipy.interpolate import CubicSpline


from ..rcparams import rcParams
from ..stats.density_utils import kde

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


def default_grid(n_items, grid=None, max_cols=4, min_cols=3):  # noqa: D202
    """Make a grid for subplots.

    Tries to get as close to sqrt(n_items) x sqrt(n_items) as it can,
    but allows for custom logic

    Parameters
    ----------
    n_items : int
        Number of panels required
    grid : tuple
        Number of rows and columns
    max_cols : int
        Maximum number of columns, inclusive
    min_cols : int
        Minimum number of columns, inclusive

    Returns
    -------
    (int, int)
        Rows and columns, so that rows * columns >= n_items
    """

    if grid is None:

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
    else:
        rows, cols = grid
        if rows * cols < n_items:
            raise ValueError("The number of rows times columns is less than the number of subplots")
        if (rows * cols) - n_items >= cols:
            warnings.warn("The number of rows times columns is larger than necessary")
        return rows, cols


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


def vectorized_to_hex(c_values, keep_alpha=False):
    """Convert a color (including vector of colors) to hex.

    Parameters
    ----------
    c: Matplotlib color

    keep_alpha: boolean
        to select if alpha values should be kept in the final hex values.

    Returns
    -------
    rgba_hex : vector of hex values
    """
    try:
        hex_color = to_hex(c_values, keep_alpha)

    except ValueError:
        hex_color = [to_hex(color, keep_alpha) for color in c_values]
    return hex_color


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
    except KeyError as err:
        raise KeyError(
            "Backend {} is not implemented. Try backend in {}".format(
                backend, set(_backend.values())
            )
        ) from err

    if backend == "bokeh":
        try:
            import bokeh

            assert packaging.version.parse(bokeh.__version__) >= packaging.version.parse("1.4.0")

        except (ImportError, AssertionError) as err:
            raise ImportError(
                "'bokeh' backend needs Bokeh (1.4.0+) installed." " Please upgrade or install"
            ) from err

    # Perform import of plotting method
    # TODO: Convert module import to top level for all plots
    module = importlib.import_module(
        "arviz.plots.backends.{backend}.{plot_module}".format(
            backend=backend, plot_module=plot_module
        )
    )

    plotting_method = getattr(module, plot_name)

    return plotting_method


def calculate_point_estimate(point_estimate, values, bw="default", circular=False, skipna=False):
    """Validate and calculate the point estimate.

    Parameters
    ----------
    point_estimate : Optional[str]
        Plot point estimate per variable. Values should be 'mean', 'median', 'mode' or None.
        Defaults to 'auto' i.e. it falls back to default set in rcParams.
    values : 1-d array
    bw: Optional[float or str]
        If numeric, indicates the bandwidth and must be positive.
        If str, indicates the method to estimate the bandwidth and must be
        one of "scott", "silverman", "isj" or "experimental" when `circular` is False
        and "taylor" (for now) when `circular` is True.
        Defaults to "default" which means "experimental" when variable is not circular
        and "taylor" when it is.
    circular: Optional[bool]
        If True, it interprets the values passed are from a circular variable measured in radians
        and a circular KDE is used. Only valid for 1D KDE. Defaults to False.
    skipna=True,
        If true ignores nan values when computing the hdi. Defaults to false.

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
        if skipna:
            point_value = np.nanmean(values)
        else:
            point_value = np.mean(values)
    elif point_estimate == "mode":
        if values.dtype.kind == "f":
            if bw == "default":
                if circular:
                    bw = "taylor"
                else:
                    bw = "experimental"
            x, density = kde(values, circular=circular, bw=bw)
            point_value = x[np.argmax(density)]
        else:
            point_value = mode(values)[0][0]
    elif point_estimate == "median":
        if skipna:
            point_value = np.nanmedian(values)
        else:
            point_value = np.median(values)

    return point_value


def is_valid_quantile(value):
    """Check if value is a number between 0 and 1."""
    try:
        value = float(value)
        return 0 < value < 1
    except ValueError:
        return False


def sample_reference_distribution(dist, shape):
    """Generate samples from a scipy distribution with a given shape."""
    x_ss = []
    densities = []
    dist_rvs = dist.rvs(size=shape)
    for idx in range(shape[1]):
        x_s, density = kde(dist_rvs[:, idx])
        x_ss.append(x_s)
        densities.append(density)
    return np.array(x_ss).T, np.array(densities).T


def set_bokeh_circular_ticks_labels(ax, hist, labels):
    """Place ticks and ticklabels on Bokeh's circular histogram."""
    ticks = np.linspace(-np.pi, np.pi, len(labels), endpoint=False)
    ax.annular_wedge(
        x=0,
        y=0,
        inner_radius=0,
        outer_radius=np.max(hist) * 1.1,
        start_angle=ticks,
        end_angle=ticks,
        line_color="grey",
    )

    radii_circles = np.linspace(0, np.max(hist) * 1.1, 4)
    ax.circle(0, 0, radius=radii_circles, fill_color=None, line_color="grey")

    offset = np.max(hist * 1.05) * 0.15
    ticks_labels_pos_1 = np.max(hist * 1.05)
    ticks_labels_pos_2 = ticks_labels_pos_1 * np.sqrt(2) / 2

    ax.text(
        [
            ticks_labels_pos_1 + offset,
            ticks_labels_pos_2 + offset,
            0,
            -ticks_labels_pos_2 - offset,
            -ticks_labels_pos_1 - offset,
            -ticks_labels_pos_2 - offset,
            0,
            ticks_labels_pos_2 + offset,
        ],
        [
            0,
            ticks_labels_pos_2 + offset / 2,
            ticks_labels_pos_1 + offset,
            ticks_labels_pos_2 + offset / 2,
            0,
            -ticks_labels_pos_2 - offset,
            -ticks_labels_pos_1 - offset,
            -ticks_labels_pos_2 - offset,
        ],
        text=labels,
        text_align="center",
    )

    return ax


def compute_ranks(ary):
    """Compute ranks for continuos and discrete variables."""
    if ary.dtype.kind == "i":
        ary_shape = ary.shape
        ary = ary.flatten()
        min_ary, max_ary = min(ary), max(ary)
        x = np.linspace(min_ary, max_ary, len(ary))
        csi = CubicSpline(x, ary)
        ary = csi(np.linspace(min_ary + 0.001, max_ary - 0.001, len(ary))).reshape(ary_shape)
    ranks = rankdata(ary, method="average").reshape(ary.shape)

    return ranks
