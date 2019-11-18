# pylint: disable=no-member
"""Bokeh Plotting Backend."""


def output_notebook(*args, **kwargs):
    """Wrap bokeh.plotting.output_notebook."""
    import bokeh.plotting as bkp

    return bkp.output_notebook(*args, **kwargs)


def output_file(*args, **kwargs):
    """Wrap bokeh.plotting.output_file."""
    import bokeh.plotting as bkp

    return bkp.output_file(*args, **kwargs)


def copy_docstring(lib, function):
    """Extract docstring from function."""
    import importlib

    try:
        module = importlib.import_module(lib)
        func = getattr(module, function)
        doc = func.__doc__
    except ImportError:
        doc = "Failed to import {}".format(lib)

    return doc


output_notebook.__doc__ += "\n\n" + copy_docstring("bokeh.plotting", "output_notebook")
output_file.__doc__ += "\n\n" + copy_docstring("bokeh.plotting", "output_file")
