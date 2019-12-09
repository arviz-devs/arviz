# pylint: disable=no-member,invalid-name,redefined-outer-name
"""Bokeh Plotting Backend."""
import packaging

# Set plot generic bokeh keyword arg defaults if none provided
KWARG_DEFAULTS = {"show": True}

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


def check_bokeh_version():
    """Check minimum bokeh version."""
    try:
        import bokeh

        assert packaging.version.parse(bokeh.__version__) >= packaging.version.parse("1.4.0")
    except (ImportError, AssertionError):
        raise ImportError("'bokeh' backend needs Bokeh (1.4.0+) installed.")


def ColumnDataSource(*args, **kwargs):
    """Wrap bokeh.models.ColumnDataSource."""
    from bokeh.models import ColumnDataSource

    return ColumnDataSource(*args, **kwargs)


output_notebook.__doc__ += "\n\n" + copy_docstring("bokeh.plotting", "output_notebook")
output_file.__doc__ += "\n\n" + copy_docstring("bokeh.plotting", "output_file")
ColumnDataSource.__doc__ += "\n\n" + copy_docstring("bokeh.models", "ColumnDataSource")
