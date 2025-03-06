# pylint: disable=too-many-nested-blocks
"""General utilities."""
import functools
import importlib
import importlib.resources
import re
import warnings
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis

from .rcparams import rcParams


STATIC_FILES = ("static/html/icons-svg-inline.html", "static/css/style.css")


class BehaviourChangeWarning(Warning):
    """Custom warning to ease filtering it."""


def _check_tilde_start(x):
    return bool(isinstance(x, str) and x.startswith("~"))


def _var_names(var_names, data, filter_vars=None, errors="raise"):
    """Handle var_names input across arviz.

    Parameters
    ----------
    var_names: str, list, or None
    data : xarray.Dataset
        Posterior data in an xarray
    filter_vars: {None, "like", "regex"}, optional, default=None
        If `None` (default), interpret var_names as the real variables names. If "like",
         interpret var_names as substrings of the real variables names. If "regex",
         interpret var_names as regular expressions on the real variables names. A la
        `pandas.filter`.
    errors: {"raise", "ignore"}, optional, default="raise"
        Select either to raise or ignore the invalid names.

    Returns
    -------
    var_name: list or None
    """
    if filter_vars not in {None, "like", "regex"}:
        raise ValueError(
            f"'filter_vars' can only be None, 'like', or 'regex', got: '{filter_vars}'"
        )

    if errors not in {"raise", "ignore"}:
        raise ValueError(f"'errors' can only be 'raise', or 'ignore', got: '{errors}'")

    if var_names is not None:
        if isinstance(data, (list, tuple)):
            all_vars = []
            for dataset in data:
                dataset_vars = list(dataset.data_vars)
                for var in dataset_vars:
                    if var not in all_vars:
                        all_vars.append(var)
        else:
            all_vars = list(data.data_vars)

        all_vars_tilde = [var for var in all_vars if _check_tilde_start(var)]
        if all_vars_tilde:
            warnings.warn(
                """ArviZ treats '~' as a negation character for variable selection.
                   Your model has variables names starting with '~', {0}. Please double check
                   your results to ensure all variables are included""".format(
                    ", ".join(all_vars_tilde)
                )
            )

        try:
            var_names = _subset_list(
                var_names, all_vars, filter_items=filter_vars, warn=False, errors=errors
            )
        except KeyError as err:
            msg = " ".join(("var names:", f"{err}", "in dataset"))
            raise KeyError(msg) from err
    return var_names


def _subset_list(subset, whole_list, filter_items=None, warn=True, errors="raise"):
    """Handle list subsetting (var_names, groups...) across arviz.

    Parameters
    ----------
    subset : str, list, or None
    whole_list : list
        List from which to select a subset according to subset elements and
        filter_items value.
    filter_items : {None, "like", "regex"}, optional
        If `None` (default), interpret `subset` as the exact elements in `whole_list`
        names. If "like", interpret `subset` as substrings of the elements in
        `whole_list`. If "regex", interpret `subset` as regular expressions to match
        elements in `whole_list`. A la `pandas.filter`.
    errors: {"raise", "ignore"}, optional, default="raise"
        Select either to raise or ignore the invalid names.

    Returns
    -------
    list or None
        A subset of ``whole_list`` fulfilling the requests imposed by ``subset``
        and ``filter_items``.
    """
    if subset is not None:
        if isinstance(subset, str):
            subset = [subset]

        whole_list_tilde = [item for item in whole_list if _check_tilde_start(item)]
        if whole_list_tilde and warn:
            warnings.warn(
                "ArviZ treats '~' as a negation character for selection. There are "
                "elements in `whole_list` starting with '~', {0}. Please double check"
                "your results to ensure all elements are included".format(
                    ", ".join(whole_list_tilde)
                )
            )

        excluded_items = [
            item[1:] for item in subset if _check_tilde_start(item) and item not in whole_list
        ]
        filter_items = str(filter_items).lower()
        if excluded_items:
            not_found = []

            if filter_items in {"like", "regex"}:
                for pattern in excluded_items[:]:
                    excluded_items.remove(pattern)
                    if filter_items == "like":
                        real_items = [real_item for real_item in whole_list if pattern in real_item]
                    else:
                        # i.e filter_items == "regex"
                        real_items = [
                            real_item for real_item in whole_list if re.search(pattern, real_item)
                        ]
                    if not real_items:
                        not_found.append(pattern)
                    excluded_items.extend(real_items)
            not_found.extend([item for item in excluded_items if item not in whole_list])
            if not_found:
                warnings.warn(
                    f"Items starting with ~: {not_found} have not been found and will be ignored"
                )
            subset = [item for item in whole_list if item not in excluded_items]

        elif filter_items == "like":
            subset = [item for item in whole_list for name in subset if name in item]
        elif filter_items == "regex":
            subset = [item for item in whole_list for name in subset if re.search(name, item)]

        existing_items = np.isin(subset, whole_list)
        if not np.all(existing_items) and (errors == "raise"):
            raise KeyError(f"{np.array(subset)[~existing_items]} are not present")

    return subset


class lazy_property:  # pylint: disable=invalid-name
    """Used to load numba first time it is needed."""

    def __init__(self, fget):
        """Lazy load a property with `fget`."""
        self.fget = fget

        # copy the getter function's docstring and other attributes
        functools.update_wrapper(self, fget)

    def __get__(self, obj, cls):
        """Call the function, set the attribute."""
        if obj is None:
            return self

        value = self.fget(obj)
        setattr(obj, self.fget.__name__, value)
        return value


class maybe_numba_fn:  # pylint: disable=invalid-name
    """Wrap a function to (maybe) use a (lazy) jit-compiled version."""

    def __init__(self, function, **kwargs):
        """Wrap a function and save compilation keywords."""
        self.function = function
        kwargs.setdefault("nopython", True)
        self.kwargs = kwargs

    @lazy_property
    def numba_fn(self):
        """Memoized compiled function."""
        try:
            numba = importlib.import_module("numba")
            numba_fn = numba.jit(**self.kwargs)(self.function)
        except ImportError:
            numba_fn = self.function
        return numba_fn

    def __call__(self, *args, **kwargs):
        """Call the jitted function or normal, depending on flag."""
        if Numba.numba_flag:
            return self.numba_fn(*args, **kwargs)
        else:
            return self.function(*args, **kwargs)


class interactive_backend:  # pylint: disable=invalid-name
    """Context manager to change backend temporarily in ipython sesson.

    It uses ipython magic to change temporarily from the ipython inline backend to
    an interactive backend of choice. It cannot be used outside ipython sessions nor
    to change backends different than inline -> interactive.

    Notes
    -----
    The first time ``interactive_backend`` context manager is called, any of the available
    interactive backends can be chosen. The following times, this same backend must be used
    unless the kernel is restarted.

    Parameters
    ----------
    backend : str, optional
        Interactive backend to use. It will be passed to ``%matplotlib`` magic, refer to
        its docs to see available options.

    Examples
    --------
    Inside an ipython session (i.e. a jupyter notebook) with the inline backend set:

    .. code::

        >>> import arviz as az
        >>> idata = az.load_arviz_data("centered_eight")
        >>> az.plot_posterior(idata) # inline
        >>> with az.interactive_backend():
        ...     az.plot_density(idata) # interactive
        >>> az.plot_trace(idata) # inline

    """

    # based on matplotlib.rc_context
    def __init__(self, backend=""):
        """Initialize context manager."""
        try:
            from IPython import get_ipython
        except ImportError as err:
            raise ImportError(
                "The exception below was risen while importing Ipython, this "
                f"context manager can only be used inside ipython sessions:\n{err}"
            ) from err
        self.ipython = get_ipython()
        if self.ipython is None:
            raise EnvironmentError("This context manager can only be used inside ipython sessions")
        self.ipython.magic(f"matplotlib {backend}")

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Exit context manager."""
        plt.show(block=True)
        self.ipython.magic("matplotlib inline")


def conditional_jit(_func=None, **kwargs):
    """Use numba's jit decorator if numba is installed.

    Notes
    -----
        If called without arguments  then return wrapped function.

        @conditional_jit
        def my_func():
            return

        else called with arguments

        @conditional_jit(nopython=True)
        def my_func():
            return

    """
    if _func is None:
        return lambda fn: functools.wraps(fn)(maybe_numba_fn(fn, **kwargs))
    lazy_numba = maybe_numba_fn(_func, **kwargs)
    return functools.wraps(_func)(lazy_numba)


def conditional_vect(function=None, **kwargs):  # noqa: D202
    """Use numba's vectorize decorator if numba is installed.

    Notes
    -----
        If called without arguments  then return wrapped function.
        @conditional_vect
        def my_func():
            return
        else called with arguments
        @conditional_vect(nopython=True)
        def my_func():
            return

    """

    def wrapper(function):
        try:
            numba = importlib.import_module("numba")
            return numba.vectorize(**kwargs)(function)

        except ImportError:
            return function

    if function:
        return wrapper(function)
    else:
        return wrapper


def numba_check():
    """Check if numba is installed."""
    numba = importlib.util.find_spec("numba")
    return numba is not None


class Numba:
    """A class to toggle numba states."""

    numba_flag = numba_check()
    """bool: Indicates whether Numba optimizations are enabled. Defaults to False."""

    @classmethod
    def disable_numba(cls):
        """To disable numba."""
        cls.numba_flag = False

    @classmethod
    def enable_numba(cls):
        """To enable numba."""
        if numba_check():
            cls.numba_flag = True
        else:
            raise ValueError("Numba is not installed")


def _numba_var(numba_function, standard_numpy_func, data, axis=None, ddof=0):
    """Replace the numpy methods used to calculate variance.

    Parameters
    ----------
    numba_function : function()
        Custom numba function included in stats/stats_utils.py.

    standard_numpy_func: function()
        Standard function included in the numpy library.

    data : array.
    axis : axis along which the variance is calculated.
    ddof : degrees of freedom allowed while calculating variance.

    Returns
    -------
    array:
        variance values calculate by appropriate function for numba speedup
        if Numba is installed or enabled.

    """
    if Numba.numba_flag:
        return numba_function(data, axis=axis, ddof=ddof)
    else:
        return standard_numpy_func(data, axis=axis, ddof=ddof)


def _stack(x, y):
    assert x.shape[1:] == y.shape[1:]
    return np.vstack((x, y))


def arange(x):
    """Jitting numpy arange."""
    return np.arange(x)


def one_de(x):
    """Jitting numpy atleast_1d."""
    if not isinstance(x, np.ndarray):
        return np.atleast_1d(x)
    result = x.reshape(1) if x.ndim == 0 else x
    return result


def two_de(x):
    """Jitting numpy at_least_2d."""
    if not isinstance(x, np.ndarray):
        return np.atleast_2d(x)
    if x.ndim == 0:
        result = x.reshape(1, 1)
    elif x.ndim == 1:
        result = x[newaxis, :]
    else:
        result = x
    return result


def expand_dims(x):
    """Jitting numpy expand_dims."""
    if not isinstance(x, np.ndarray):
        return np.expand_dims(x, 0)
    shape = x.shape
    return x.reshape(shape[:0] + (1,) + shape[0:])


@conditional_jit(cache=True, nopython=True)
def _dot(x, y):
    return np.dot(x, y)


@conditional_jit(cache=True, nopython=True)
def _cov_1d(x):
    x = x - x.mean()
    ddof = x.shape[0] - 1
    return np.dot(x.T, x.conj()) / ddof


# @conditional_jit(cache=True)
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
        raise ValueError(f"{data.ndim} dimension arrays are not supported")


def flatten_inference_data_to_dict(
    data,
    var_names=None,
    groups=None,
    dimensions=None,
    group_info=False,
    var_name_format=None,
    index_origin=None,
):
    """Transform data to dictionary.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_inference_data for details
    var_names : str or list of str, optional
        Variables to be processed, if None all variables are processed.
    groups : str or list of str, optional
        Select groups for CDS. Default groups are
        {"posterior_groups", "prior_groups", "posterior_groups_warmup"}
            - posterior_groups: posterior, posterior_predictive, sample_stats
            - prior_groups: prior, prior_predictive, sample_stats_prior
            - posterior_groups_warmup: warmup_posterior, warmup_posterior_predictive,
                                       warmup_sample_stats
    ignore_groups : str or list of str, optional
        Ignore specific groups from CDS.
    dimension : str, or list of str, optional
        Select dimensions along to slice the data. By default uses ("chain", "draw").
    group_info : bool
        Add group info for `var_name_format`
    var_name_format : str or tuple of tuple of string, optional
        Select column name format for non-scalar input.
        Predefined options are {"brackets", "underscore", "cds"}
            "brackets":
                - add_group_info == False: theta[0,0]
                - add_group_info == True: theta_posterior[0,0]
            "underscore":
                - add_group_info == False: theta_0_0
                - add_group_info == True: theta_posterior_0_0_
            "cds":
                - add_group_info == False: theta_ARVIZ_CDS_SELECTION_0_0
                - add_group_info == True: theta_ARVIZ_GROUP_posterior__ARVIZ_CDS_SELECTION_0_0
            tuple:
                Structure:
                    tuple: (dim_info, group_info)
                        dim_info: (str: `.join` separator,
                                   str: dim_separator_start,
                                   str: dim_separator_end)
                        group_info: (str: group separator start, str: group separator end)
                Example: ((",", "[", "]"), ("_", ""))
                    - add_group_info == False: theta[0,0]
                    - add_group_info == True: theta_posterior[0,0]
    index_origin : int, optional
        Start parameter indices from `index_origin`. Either 0 or 1.

    Returns
    -------
    dict
    """
    from .data import convert_to_inference_data

    data = convert_to_inference_data(data)

    if groups is None:
        groups = ["posterior", "posterior_predictive", "sample_stats"]
    elif isinstance(groups, str):
        if groups.lower() == "posterior_groups":
            groups = ["posterior", "posterior_predictive", "sample_stats"]
        elif groups.lower() == "prior_groups":
            groups = ["prior", "prior_predictive", "sample_stats_prior"]
        elif groups.lower() == "posterior_groups_warmup":
            groups = ["warmup_posterior", "warmup_posterior_predictive", "warmup_sample_stats"]
        else:
            raise TypeError(
                (
                    "Valid predefined groups are "
                    "{posterior_groups, prior_groups, posterior_groups_warmup}"
                )
            )

    if dimensions is None:
        dimensions = "chain", "draw"
    elif isinstance(dimensions, str):
        dimensions = (dimensions,)

    if var_name_format is None:
        var_name_format = "brackets"

    if isinstance(var_name_format, str):
        var_name_format = var_name_format.lower()

    if var_name_format == "brackets":
        dim_join_separator, dim_separator_start, dim_separator_end = ",", "[", "]"
        group_separator_start, group_separator_end = "_", ""
    elif var_name_format == "underscore":
        dim_join_separator, dim_separator_start, dim_separator_end = "_", "_", ""
        group_separator_start, group_separator_end = "_", ""
    elif var_name_format == "cds":
        dim_join_separator, dim_separator_start, dim_separator_end = (
            "_",
            "_ARVIZ_CDS_SELECTION_",
            "",
        )
        group_separator_start, group_separator_end = "_ARVIZ_GROUP_", ""
    elif isinstance(var_name_format, str):
        msg = 'Invalid predefined format. Select one {"brackets", "underscore", "cds"}'
        raise TypeError(msg)
    else:
        (
            (dim_join_separator, dim_separator_start, dim_separator_end),
            (group_separator_start, group_separator_end),
        ) = var_name_format

    if index_origin is None:
        index_origin = rcParams["data.index_origin"]

    data_dict = {}
    for group in groups:
        if hasattr(data, group):
            group_data = getattr(data, group).stack(stack_dimension=dimensions)
            for var_name, var in group_data.data_vars.items():
                var_values = var.values
                if var_names is not None and var_name not in var_names:
                    continue
                for dim_name in dimensions:
                    if dim_name not in data_dict:
                        data_dict[dim_name] = var.coords.get(dim_name).values
                if len(var.shape) == 1:
                    if group_info:
                        var_name_dim = (
                            "{var_name}" "{group_separator_start}{group}{group_separator_end}"
                        ).format(
                            var_name=var_name,
                            group_separator_start=group_separator_start,
                            group=group,
                            group_separator_end=group_separator_end,
                        )
                    else:
                        var_name_dim = f"{var_name}"
                    data_dict[var_name_dim] = var.values
                else:
                    for loc in np.ndindex(var.shape[:-1]):
                        if group_info:
                            var_name_dim = (
                                "{var_name}"
                                "{group_separator_start}{group}{group_separator_end}"
                                "{dim_separator_start}{dim_join}{dim_separator_end}"
                            ).format(
                                var_name=var_name,
                                group_separator_start=group_separator_start,
                                group=group,
                                group_separator_end=group_separator_end,
                                dim_separator_start=dim_separator_start,
                                dim_join=dim_join_separator.join(
                                    (str(item + index_origin) for item in loc)
                                ),
                                dim_separator_end=dim_separator_end,
                            )
                        else:
                            var_name_dim = (
                                "{var_name}" "{dim_separator_start}{dim_join}{dim_separator_end}"
                            ).format(
                                var_name=var_name,
                                dim_separator_start=dim_separator_start,
                                dim_join=dim_join_separator.join(
                                    (str(item + index_origin) for item in loc)
                                ),
                                dim_separator_end=dim_separator_end,
                            )

                        data_dict[var_name_dim] = var_values[loc]
    return data_dict


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

        except ValueError as err:
            invalid_coords = set(coords.keys()) - set(data.coords.keys())
            raise ValueError(f"Coords {invalid_coords} are invalid coordinate keys") from err

        except KeyError as err:
            raise KeyError(
                (
                    "Coords should follow mapping format {{coord_name:[dim1, dim2]}}. "
                    "Check that coords structure is correct and"
                    " dimensions are valid. {}"
                ).format(err)
            ) from err
    if not isinstance(coords, (list, tuple)):
        coords = [coords] * len(data)
    data_subset = []
    for idx, (datum, coords_dict) in enumerate(zip(data, coords)):
        try:
            data_subset.append(get_coords(datum, coords_dict))
        except ValueError as err:
            raise ValueError(f"Error in data[{idx}]: {err}") from err
        except KeyError as err:
            raise KeyError(f"Error in data[{idx}]: {err}") from err
    return data_subset


@lru_cache(None)
def _load_static_files():
    """Lazily load the resource files into memory the first time they are needed.

    Clone from xarray.core.formatted_html_template.
    """
    return [
        importlib.resources.files("arviz").joinpath(fname).read_text(encoding="utf-8")
        for fname in STATIC_FILES
    ]


class HtmlTemplate:
    """Contain html templates for InferenceData repr."""

    html_template = """
            <div>
              <div class='xr-header'>
                <div class="xr-obj-type">arviz.InferenceData</div>
              </div>
              <ul class="xr-sections group-sections">
              {}
              </ul>
            </div>
            """
    element_template = """
            <li class = "xr-section-item">
                  <input id="idata_{group_id}" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_{group_id}" class = "xr-section-summary">{group}</label>
                  <div class="xr-section-inline-details"></div>
                  <div class="xr-section-details">
                      <ul id="xr-dataset-coord-list" class="xr-var-list">
                          <div style="padding-left:2rem;">{xr_data}<br></div>
                      </ul>
                  </div>
            </li>
            """
    _, css_style = _load_static_files()  # pylint: disable=protected-access
    specific_style = ".xr-wrap{width:700px!important;}"
    css_template = f"<style> {css_style}{specific_style} </style>"


def either_dict_or_kwargs(
    pos_kwargs,
    kw_kwargs,
    func_name,
):
    """Clone from xarray.core.utils."""
    if pos_kwargs is None:
        return kw_kwargs
    if not hasattr(pos_kwargs, "keys") and hasattr(pos_kwargs, "__getitem__"):
        raise ValueError(f"the first argument to .{func_name} must be a dictionary")
    if kw_kwargs:
        raise ValueError(f"cannot specify both keyword and positional arguments to .{func_name}")
    return pos_kwargs


class Dask:
    """Class to toggle Dask states.

    Warnings
    --------
    Dask integration is an experimental feature still in progress. It can already be used
    but it doesn't work with all stats nor diagnostics yet.
    """

    dask_flag = False
    """bool: Enables Dask parallelization when set to True. Defaults to False."""
    dask_kwargs = None
    """dict: Additional keyword arguments for Dask configuration.
    Defaults to an empty dictionary."""

    @classmethod
    def enable_dask(cls, dask_kwargs=None):
        """To enable Dask.

        Parameters
        ----------
        dask_kwargs : dict
            Dask related kwargs passed to :func:`~arviz.wrap_xarray_ufunc`.
        """
        cls.dask_flag = True
        cls.dask_kwargs = dask_kwargs

    @classmethod
    def disable_dask(cls):
        """To disable Dask."""
        cls.dask_flag = False
        cls.dask_kwargs = None


def conditional_dask(func):
    """Conditionally pass dask kwargs to `wrap_xarray_ufunc`."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not Dask.dask_flag:
            return func(*args, **kwargs)
        user_kwargs = kwargs.pop("dask_kwargs", None)
        if user_kwargs is None:
            user_kwargs = {}
        default_kwargs = Dask.dask_kwargs
        return func(dask_kwargs={**default_kwargs, **user_kwargs}, *args, **kwargs)

    return wrapper
