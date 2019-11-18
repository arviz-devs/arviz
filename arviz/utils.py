"""General utilities."""
import importlib
import functools
import warnings
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt


def _var_names(var_names, data):
    """Handle var_names input across arviz.

    Parameters
    ----------
    var_names: str, list, or None
    data : xarray.Dataset
        Posterior data in an xarray
    Returns
    -------
    var_name: list or None
    """
    if var_names is not None:

        if isinstance(var_names, str):
            var_names = [var_names]

        if isinstance(data, (list, tuple)):
            all_vars = []
            for dataset in data:
                dataset_vars = list(dataset.data_vars)
                for var in dataset_vars:
                    if var not in all_vars:
                        all_vars.append(var)
        else:
            all_vars = list(data.data_vars)

        excluded_vars = [i[1:] for i in var_names if i.startswith("~") and i not in all_vars]

        all_vars_tilde = [i for i in all_vars if i.startswith("~")]

        if all_vars_tilde:
            warnings.warn(
                """ArviZ treats '~' as a negation character for variable selection.
                   Your model has variables names starting with '~', {0}. Please double check
                   your results to ensure all variables are included""".format(
                    ", ".join(all_vars_tilde)
                )
            )

        if excluded_vars:
            var_names = [i for i in all_vars if i not in excluded_vars]

        existent_vars = np.isin(var_names, all_vars)
        if not np.all(existent_vars):
            raise KeyError(
                "{} var names are not present in dataset".format(
                    np.array(var_names)[~existent_vars]
                )
            )

    return var_names


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
                "context manager can only be used inside ipython sessions:\n{}".format(err)
            )
        self.ipython = get_ipython()
        if self.ipython is None:
            raise EnvironmentError("This context manager can only be used inside ipython sessions")
        self.ipython.magic("matplotlib {}".format(backend))

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
    else:
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
    if x.ndim == 0:
        result = x.reshape(1)
    else:
        result = x
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


@conditional_jit(parallel=True)
def full(shape, x, dtype=None):
    """Jitting numpy full."""
    return np.full(shape, x, dtype=dtype)
