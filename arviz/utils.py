"""General utilities."""
import importlib
import warnings
import numpy as np


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

    return var_names


def conditional_jit(function=None, **kwargs):  # noqa: D202
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

    def wrapper(function):
        try:
            numba = importlib.import_module("numba")
            return numba.jit(**kwargs)(function)

        except ImportError:
            return function

    if function:
        return wrapper(function)
    else:
        return wrapper


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
