# pylint: disable=no-member,invalid-name,redefined-outer-name
"""ArviZ plotting backends."""
from pandas import DataFrame


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

    Uses `_ARVIZ_GROUP_` and `_ARVIZ_CDS_SELECTION_`to separate var_name
    from group and dimensions in CDS columns.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_inference_data for details
    var_names : str or list of str, optional
        Variables to be processed, if None all variables are processed.
    groups : str or list of str, optional
        Select groups for CDS. Default groups are {"posterior_groups", "prior_groups"}
            - posterior_groups: posterior, posterior_predictive, sample_stats
            - prior_groups: prior, posterior_predictive, sample_stats_prior
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
    bokeh.models.ColumnDataSource object
    """
    from ...utils import flat_inference_data_to_dict

    if var_name_format is None:
        var_name_format = "cds"

    cds_dict = flat_inference_data_to_dict(
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


output_notebook.__doc__ += "\n\n" + _copy_docstring("bokeh.plotting", "output_notebook")
output_file.__doc__ += "\n\n" + _copy_docstring("bokeh.plotting", "output_file")
ColumnDataSource.__doc__ += "\n\n" + _copy_docstring("bokeh.models", "ColumnDataSource")
