# pylint: disable=no-member,invalid-name,redefined-outer-name
"""ArviZ plotting backends."""
import numpy as np

from ...data import convert_to_inference_data
from ...rcparams import rcParams


def to_cds(data, var_names=None, groups=None, ignore_groups=None, index_origin=None):
    """Transform data to ColumnDataSource (CDS) compatible with Bokeh.

    Uses `_ARVIZ_CDS_SELECTION_` to separate var_name from dimensions.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_inference_data for details
    var_names : list of variable names, optional
        Variables to be processed, if None all variables are processed.
    groups : str or list of str, optional
        Select groups for CDS. Default groups are {"posterior_groups", "prior_groups"}
            posterior_groups: posterior, posterior_predictive, sample_stats
            prior_groups: prior, posterior_predictive, sample_stats_prior
    ignore_groups : str or list of str, optional
        Ignore specific groups from CDS.
    index_origin : int, optional
        Start parameter indeces from `index_origin`. Either 0 or 1.

    Returns
    -------
    bokeh.models.ColumnDataSource object
    """
    from bokeh.models import ColumnDataSource

    data = convert_to_inference_data(data)

    if groups is None:
        groups = ["posterior", "posterior_predictive", "sample_stats"]
    elif isinstance(groups, str):
        if groups.lower() == "posterior_groups":
            groups = ["posterior", "posterior_predictive", "sample_stats"]
        elif groups.lower() == "prior_groups":
            groups = ["prior", "prior_predictive", "sample_stats_prior"]
        else:
            raise TypeError("Valid predefined groups are {posterior_groups, prior_groups}")

    if ignore_groups is None:
        ignore_groups = []
    elif isinstance(ignore_groups, str):
        ignore_groups = [ignore_groups]

    if index_origin is None:
        index_origin = rcParams["data.index_origin"]

    cds_dict = {}
    for group in groups:
        if group in ignore_groups:
            continue
        if hasattr(data, group):
            group_data = getattr(data, group).stack(samples=("chain", "draw"))
            for var_name, var in group_data.data_vars.items():
                if var_names is not None and var_name not in var_names:
                    continue
                if "chain" not in cds_dict:
                    cds_dict["chain"] = var.coords.get("chain").values
                if "draw" not in cds_dict:
                    cds_dict["draw"] = var.coords.get("draw").values
                if len(var.shape) == 1:
                    cds_dict["{}_ARVIZ_GROUP_{}".format(var_name, group)] = var.values
                else:
                    for loc in np.ndindex(var.shape[:-1]):
                        var_name_dim = "{}_ARVIZ_GROUP_{}_ARVIZ_CDS_SELECTION_{}".format(
                            var_name, group, "_".join((str(item + index_origin) for item in loc))
                        )
                        cds_dict[var_name_dim] = var[loc].values
    cds_data = ColumnDataSource(cds_dict)
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
