"""General utilities."""
import importlib


def _var_names(var_names):
    """Handle var_names input across arviz.

    Parameters
    ----------
    var_names: str, list, or None

    Returns
    -------
    var_name: list or None
    """
    if var_names is None:
        return None

    elif isinstance(var_names, str):
        return [var_names]

    else:
        return var_names


def conditional_jit(function=None, **kwargs):
    """Use numba's jit decorator if numba is installed."""
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
