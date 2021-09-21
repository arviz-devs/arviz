# pylint: disable=no-member,invalid-name,redefined-outer-name
"""ArviZ plotting backends."""
import re

import numpy as np
from pandas import DataFrame

from ...rcparams import rcParams

__all__ = [
    "to_cds",
    "output_notebook",
    "output_file",
    "ColumnDataSource",
    "create_layout",
    "show_layout",
]


def to_cds(
    data,
    var_names=None,
    groups=None,
    dimensions=None,
    group_info=True,
    var_name_format=None,
    index_origin=None,
):
    """Transform data to ColumnDataSource (CDS) compatible with Bokeh.

    Uses `_ARVIZ_GROUP_` and `_ARVIZ_CDS_SELECTION_` to separate var_name
    from group and dimensions in CDS columns.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_inference_data for details
    var_names : str or list of str, optional
        Variables to be processed, if None all variables are processed.
    groups : str or list of str, optional
        Select groups for CDS. Default groups are {"posterior_groups", "prior_groups",
        "posterior_groups_warmup"}

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
                - add_group_info == False: ``theta[0,0]``
                - add_group_info == True: ``theta_posterior[0,0]``
            "underscore":
                - add_group_info == False: ``theta_0_0``
                - add_group_info == True: ``theta_posterior_0_0_``
            "cds":
                - add_group_info == False: ``theta_ARVIZ_CDS_SELECTION_0_0``
                - add_group_info == True: ``theta_ARVIZ_GROUP_posterior__ARVIZ_CDS_SELECTION_0_0``
            tuple:
                Structure:

                    - tuple: (dim_info, group_info)

                        - dim_info: (str: `.join` separator,
                          str: dim_separator_start,
                          str: dim_separator_end)
                        - group_info: (str: group separator start, str: group separator end)

                Example: ((",", "[", "]"), ("_", ""))

                    - add_group_info == False: ``theta[0,0]``
                    - add_group_info == True: ``theta_posterior[0,0]``

    index_origin : int, optional
        Start parameter indices from `index_origin`. Either 0 or 1.

    Returns
    -------
    bokeh.models.ColumnDataSource object
    """
    from ...utils import flatten_inference_data_to_dict

    if var_name_format is None:
        var_name_format = "cds"

    cds_dict = flatten_inference_data_to_dict(
        data=data,
        var_names=var_names,
        groups=groups,
        dimensions=dimensions,
        group_info=group_info,
        index_origin=index_origin,
        var_name_format=var_name_format,
    )
    cds_data = ColumnDataSource(DataFrame.from_dict(cds_dict, orient="columns"))
    return cds_data


def output_notebook(*args, **kwargs):
    """Wrap bokeh.plotting.output_notebook."""
    import bokeh.plotting as bkp

    return bkp.output_notebook(*args, **kwargs)


def output_file(*args, **kwargs):
    """Wrap bokeh.plotting.output_file."""
    import bokeh.plotting as bkp

    return bkp.output_file(*args, **kwargs)


def ColumnDataSource(*args, **kwargs):
    """Wrap bokeh.models.ColumnDataSource."""
    from bokeh.models import ColumnDataSource

    return ColumnDataSource(*args, **kwargs)


def create_layout(ax, force_layout=False):
    """Transform bokeh array of figures to layout."""
    ax = np.atleast_2d(ax)
    subplot_order = rcParams["plot.bokeh.layout.order"]
    if force_layout:
        from bokeh.layouts import gridplot as layout

        ax = ax.tolist()
        layout_args = {
            "sizing_mode": rcParams["plot.bokeh.layout.sizing_mode"],
            "toolbar_location": rcParams["plot.bokeh.layout.toolbar_location"],
        }
    elif any(item in subplot_order for item in ("row", "column")):
        # check number of rows
        match = re.match(r"(\d*)(row|column)", subplot_order)
        n = int(match.group(1)) if match.group(1) is not None else 1
        subplot_order = match.group(2)
        # set up 1D list of axes
        ax = [item for item in ax.ravel().tolist() if item is not None]
        layout_args = {"sizing_mode": rcParams["plot.bokeh.layout.sizing_mode"]}
        if subplot_order == "row" and n == 1:
            from bokeh.layouts import row as layout
        elif subplot_order == "column" and n == 1:
            from bokeh.layouts import column as layout
        else:
            from bokeh.layouts import layout

        if n != 1:
            ax = np.array(ax + [None for _ in range(int(np.ceil(len(ax) / n)) - len(ax))])
            if subplot_order == "row":
                ax = ax.reshape(n, -1)
            else:
                ax = ax.reshape(-1, n)
            ax = ax.tolist()
    else:
        if subplot_order in ("square", "square_trimmed"):
            ax = [item for item in ax.ravel().tolist() if item is not None]
            n = int(np.ceil(len(ax) ** 0.5))
            ax = ax + [None for _ in range(n ** 2 - len(ax))]
            ax = np.array(ax).reshape(n, n)
        ax = ax.tolist()
        if (subplot_order == "square_trimmed") and any(
            all(item is None for item in row) for row in ax
        ):
            from bokeh.layouts import layout

            ax = [row for row in ax if not all(item is None for item in row)]
            layout_args = {"sizing_mode": rcParams["plot.bokeh.layout.sizing_mode"]}
        else:
            from bokeh.layouts import gridplot as layout

            layout_args = {
                "sizing_mode": rcParams["plot.bokeh.layout.sizing_mode"],
                "toolbar_location": rcParams["plot.bokeh.layout.toolbar_location"],
            }
    # ignore "fixed" sizing_mode without explicit width and height
    if layout_args.get("sizing_mode", "") == "fixed":
        layout_args.pop("sizing_mode")
    return layout(ax, **layout_args)


def show_layout(ax, show=True, force_layout=False):
    """Create a layout and call bokeh show."""
    if show is None:
        show = rcParams["plot.bokeh.show"]
    if show:
        import bokeh.plotting as bkp

        layout = create_layout(ax, force_layout=force_layout)
        bkp.show(layout)


def _copy_docstring(lib, function):
    """Extract docstring from function."""
    import importlib

    try:
        module = importlib.import_module(lib)
        func = getattr(module, function)
        doc = func.__doc__
    except ImportError:
        doc = f"Failed to import function {function} from {lib}"

    if not isinstance(doc, str):
        doc = ""
    return doc


output_notebook.__doc__ += "\n\n" + _copy_docstring("bokeh.plotting", "output_notebook")
output_file.__doc__ += "\n\n" + _copy_docstring("bokeh.plotting", "output_file")
ColumnDataSource.__doc__ += "\n\n" + _copy_docstring("bokeh.models", "ColumnDataSource")
