# pylint: disable=no-member,invalid-name,redefined-outer-name
"""ArviZ plotting backends."""


def output_notebook(*args, **kwargs):
    """Wrap bokeh.plotting.output_notebook."""
    import bokeh.plotting as bkp

    return bkp.output_notebook(*args, **kwargs)


def output_file(*args, **kwargs):
    """Wrap bokeh.plotting.output_file."""
    import bokeh.plotting as bkp

    return bkp.output_file(*args, **kwargs)


def _copy_docstring(lib, function):
    """Extract docstring from function."""
    import importlib

    try:
        module = importlib.import_module(lib)
        func = getattr(module, function)
        doc = func.__doc__
    except ImportError:
        doc = "Failed to import function {} from {}".format(function, lib)

    return doc


def ColumnDataSource(*args, **kwargs):
    """Wrap bokeh.models.ColumnDataSource."""
    from bokeh.models import ColumnDataSource

    return ColumnDataSource(*args, **kwargs)


output_notebook.__doc__ += "\n\n" + _copy_docstring("bokeh.plotting", "output_notebook")
output_file.__doc__ += "\n\n" + _copy_docstring("bokeh.plotting", "output_file")
ColumnDataSource.__doc__ += "\n\n" + _copy_docstring("bokeh.models", "ColumnDataSource")
