"""General utilities."""
import importlib
import warnings
import timeit
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




def wrapper(function, *args, **kwargs):
    def wrapped():
        return function(*args, **kwargs
def numba_check(function, *args, **kwargs):
    """
    Compares the time of a numbified function to a non numbified one.

    """

    def wrapper(function, *args, **kwargs):
        def wrapped():
            return function(*args, **kwargs)

        return wrapped

    wrapped = wrapper(function, *args, **kwargs)
    wrapped_numba = wrapper(conditional_jit(function), *args, **kwargs)
    a = timeit.timeit(wrapped, number=1000)
    dummy = timeit.timeit(wrapped_numba, number=1000)
    b = timeit.timeit(wrapped_numba, number=1000)
    return a / b
    dummy = timeit.timeit(wrapped_numba, number=1)
    time = np.zeros(100)
    for i in range(0, 100):
        a = timeit.timeit(wrapped, number=1000)
        b = timeit.timeit(wrapped_numba, number=1000)
        time[i] = a / b
    return time.mean()

