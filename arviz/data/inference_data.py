"""Data structure for using netcdf groups with xarray."""
from collections import OrderedDict
from collections.abc import Sequence
from copy import copy as ccopy, deepcopy
from datetime import datetime
from html import escape
import warnings
import uuid

import netCDF4 as nc
import numpy as np
import xarray as xr
from xarray.core.options import OPTIONS

from ..utils import _subset_list, HtmlTemplate
from ..rcparams import rcParams

SUPPORTED_GROUPS = [
    "posterior",
    "posterior_predictive",
    "predictions",
    "log_likelihood",
    "sample_stats",
    "prior",
    "prior_predictive",
    "sample_stats_prior",
    "observed_data",
    "constant_data",
    "predictions_constant_data",
]

WARMUP_TAG = "warmup_"

SUPPORTED_GROUPS_WARMUP = [
    "{}posterior".format(WARMUP_TAG),
    "{}posterior_predictive".format(WARMUP_TAG),
    "{}predictions".format(WARMUP_TAG),
    "{}sample_stats".format(WARMUP_TAG),
    "{}log_likelihood".format(WARMUP_TAG),
]

SUPPORTED_GROUPS_ALL = SUPPORTED_GROUPS + SUPPORTED_GROUPS_WARMUP


class InferenceData:
    """Container for inference data storage using xarray.

    For a detailed introduction to ``InferenceData`` objects and their usage, see
    :doc:`/notebooks/XarrayforArviZ`. This page provides help and documentation
    on ``InferenceData`` methods and their low level implementation.
    """

    def __init__(self, **kwargs):
        """Initialize InferenceData object from keyword xarray datasets.

        Parameters
        ----------
        kwargs :
            Keyword arguments of xarray datasets

        Examples
        --------
        Initiate an InferenceData object from scratch, not recommended. InferenceData
        objects should be initialized using ``from_xyz`` methods, see :ref:`data_api` for more
        details.

        .. ipython::

            In [1]: import arviz as az
               ...: import numpy as np
               ...: import xarray as xr
               ...: dataset = xr.Dataset(
               ...:     {
               ...:         "a": (["chain", "draw", "a_dim"], np.random.normal(size=(4, 100, 3))),
               ...:         "b": (["chain", "draw"], np.random.normal(size=(4, 100))),
               ...:     },
               ...:     coords={
               ...:         "chain": (["chain"], np.arange(4)),
               ...:         "draw": (["draw"], np.arange(100)),
               ...:         "a_dim": (["a_dim"], ["x", "y", "z"]),
               ...:     }
               ...: )
               ...: idata = az.InferenceData(posterior=dataset, prior=dataset)
               ...: idata

        We have created an ``InferenceData`` object with two groups. Now we can check its
        contents:

        .. ipython::

            In [1]: idata.posterior

        """
        self._groups = []
        self._groups_warmup = []
        save_warmup = kwargs.pop("save_warmup", False)
        key_list = [key for key in SUPPORTED_GROUPS_ALL if key in kwargs]
        for key in kwargs:
            if key not in SUPPORTED_GROUPS_ALL:
                key_list.append(key)
                warnings.warn(
                    "{} group is not defined in the InferenceData scheme".format(key), UserWarning
                )
        for key in key_list:
            dataset = kwargs[key]
            dataset_warmup = None
            if dataset is None:
                continue
            elif isinstance(dataset, (list, tuple)):
                dataset, dataset_warmup = kwargs[key]
            elif not isinstance(dataset, xr.Dataset):
                raise ValueError(
                    "Arguments to InferenceData must be xarray Datasets "
                    "(argument '{}' was type '{}')".format(key, type(dataset))
                )
            if not key.startswith(WARMUP_TAG):
                if dataset:
                    setattr(self, key, dataset)
                    self._groups.append(key)
            elif key.startswith(WARMUP_TAG):
                if dataset:
                    setattr(self, key, dataset)
                    self._groups_warmup.append(key)
            if save_warmup and dataset_warmup is not None:
                if dataset_warmup:
                    key = "{}{}".format(WARMUP_TAG, key)
                    setattr(self, key, dataset_warmup)
                    self._groups_warmup.append(key)

    def __repr__(self):
        """Make string representation of InferenceData object."""
        msg = "Inference data with groups:\n\t> {options}".format(
            options="\n\t> ".join(self._groups)
        )
        if self._groups_warmup:
            msg += "\n\nWarmup iterations saved ({}*).".format(WARMUP_TAG)
        return msg

    def _repr_html_(self):
        """Make html representation of InferenceData object."""
        display_style = OPTIONS["display_style"]
        if display_style == "text":
            html_repr = f"<pre>{escape(repr(self))}</pre>"
        else:
            elements = "".join(
                [
                    HtmlTemplate.element_template.format(
                        group_id=group + str(uuid.uuid4()),
                        group=group,
                        xr_data=getattr(  # pylint: disable=protected-access
                            self, group
                        )._repr_html_(),
                    )
                    for group in self._groups_all
                ]
            )
            formatted_html_template = HtmlTemplate.html_template.format(  # pylint: disable=possibly-unused-variable
                elements
            )
            css_template = HtmlTemplate.css_template  # pylint: disable=possibly-unused-variable
            html_repr = "%(formatted_html_template)s%(css_template)s" % locals()
        return html_repr

    def __delattr__(self, group):
        """Delete a group from the InferenceData object."""
        if group in self._groups:
            self._groups.remove(group)
        elif group in self._groups_warmup:
            self._groups_warmup.remove(group)
        object.__delattr__(self, group)

    @property
    def _groups_all(self):
        return self._groups + self._groups_warmup

    @staticmethod
    def from_netcdf(filename):
        """Initialize object from a netcdf file.

        Expects that the file will have groups, each of which can be loaded by xarray.
        By default, the datasets of the InferenceData object will be lazily loaded instead
        of being loaded into memory. This
        behaviour is regulated by the value of ``az.rcParams["data.load"]``.

        Parameters
        ----------
        filename : str
            location of netcdf file

        Returns
        -------
        InferenceData object
        """
        groups = {}
        with nc.Dataset(filename, mode="r") as data:
            data_groups = list(data.groups)

        for group in data_groups:
            with xr.open_dataset(filename, group=group) as data:
                if rcParams["data.load"] == "eager":
                    groups[group] = data.load()
                else:
                    groups[group] = data
        return InferenceData(**groups)

    def to_netcdf(self, filename, compress=True, groups=None):
        """Write InferenceData to file using netcdf4.

        Parameters
        ----------
        filename : str
            Location to write to
        compress : bool, optional
            Whether to compress result. Note this saves disk space, but may make
            saving and loading somewhat slower (default: True).
        groups : list, optional
            Write only these groups to netcdf file.

        Returns
        -------
        str
            Location of netcdf file
        """
        mode = "w"  # overwrite first, then append
        if self._groups_all:  # check's whether a group is present or not.
            if groups is None:
                groups = self._groups_all
            else:
                groups = [group for group in self._groups_all if group in groups]

            for group in groups:
                data = getattr(self, group)
                kwargs = {}
                if compress:
                    kwargs["encoding"] = {var_name: {"zlib": True} for var_name in data.variables}
                data.to_netcdf(filename, mode=mode, group=group, **kwargs)
                data.close()
                mode = "a"
        else:  # creates a netcdf file for an empty InferenceData object.
            empty_netcdf_file = nc.Dataset(filename, mode="w", format="NETCDF4")
            empty_netcdf_file.close()
        return filename

    def __add__(self, other):
        """Concatenate two InferenceData objects."""
        return concat(self, other, copy=True, inplace=False)

    def sel(
        self, groups=None, filter_groups=None, inplace=False, chain_prior=None, **kwargs,
    ):
        """Perform an xarray selection on all groups.

        Loops groups to perform Dataset.sel(key=item)
        for every kwarg if key is a dimension of the dataset.
        One example could be performing a burn in cut on the InferenceData object
        or discarding a chain. The selection is performed on all relevant groups (like
        posterior, prior, sample stats) while non relevant groups like observed data are
        omitted. See :meth:`xarray.Dataset.sel <xarray:xarray.Dataset.sel>`

        Parameters
        ----------
        groups: str or list of str, optional
            Groups where the selection is to be applied. Can either be group names
            or metagroup names.
        inplace: bool, optional
            If ``True``, modify the InferenceData object inplace,
            otherwise, return the modified copy.
        chain_prior: bool, optional, deprecated
            If ``False``, do not select prior related groups using ``chain`` dim.
            Otherwise, use selection on ``chain`` if present. Default=False
        **kwargs: mapping
            It must be accepted by Dataset.sel().

        Returns
        -------
        InferenceData
            A new InferenceData object by default.
            When `inplace==True` perform selection in place and return `None`

        Examples
        --------
        Use ``sel`` to discard one chain of the InferenceData object. We first check the
        dimensions of the original object:

        .. ipython::

            In [1]: import arviz as az
               ...: idata = az.load_arviz_data("centered_eight")
               ...: del idata.prior  # prior group only has 1 chain currently
               ...: print(idata.posterior.coords)
               ...: print(idata.posterior_predictive.coords)
               ...: print(idata.observed_data.coords)

        In order to remove the third chain:

        .. ipython::

            In [1]: idata_subset = idata.sel(chain=[0, 1, 3])
               ...: print(idata_subset.posterior.coords)
               ...: print(idata_subset.posterior_predictive.coords)
               ...: print(idata_subset.observed_data.coords)

        """
        if chain_prior is not None:
            warnings.warn(
                "chain_prior has been deprecated. Use groups argument and "
                "rcParams['data.metagroups'] instead.",
                DeprecationWarning,
            )
        else:
            chain_prior = False
        groups = self._group_names(groups, filter_groups)

        out = self if inplace else deepcopy(self)
        for group in groups:
            dataset = getattr(self, group)
            valid_keys = set(kwargs.keys()).intersection(dataset.dims)
            if not chain_prior and "prior" in group:
                valid_keys -= {"chain"}
            dataset = dataset.sel(**{key: kwargs[key] for key in valid_keys})
            setattr(out, group, dataset)
        if inplace:
            return None
        else:
            return out

    def _group_names(self, groups, filter_groups=None):
        """Handle expansion of group names input across arviz.

        Parameters
        ----------
        groups: str, list of str or None
            group or metagroup names.
        idata: xarray.Dataset
            Posterior data in an xarray
        filter_groups: {None, "like", "regex"}, optional, default=None
            If `None` (default), interpret groups as the real group or metagroup names.
            If "like", interpret groups as substrings of the real group or metagroup names.
            If "regex", interpret groups as regular expressions on the real group or
            metagroup names. A la `pandas.filter`.

        Returns
        -------
        groups: list
        """
        all_groups = self._groups_all
        if groups is None:
            return all_groups
        if isinstance(groups, str):
            groups = [groups]
        sel_groups = []
        metagroups = rcParams["data.metagroups"]
        for group in groups:
            if group[0] == "~":
                sel_groups.extend(
                    [f"~{item}" for item in metagroups[group[1:]] if item in all_groups]
                    if group[1:] in metagroups
                    else [group]
                )
            else:
                sel_groups.extend(
                    [item for item in metagroups[group] if item in all_groups]
                    if group in metagroups
                    else [group]
                )

        try:
            group_names = _subset_list(sel_groups, all_groups, filter_items=filter_groups)
        except KeyError as err:
            msg = " ".join(("groups:", f"{err}", "in InferenceData"))
            raise KeyError(msg)
        return group_names

    def map(self, fun, groups=None, filter_groups=None, inplace=False, args=None, **kwargs):
        """Apply a function to multiple groups.

        Applies ``fun`` groupwise to the selected ``InferenceData`` groups and overwrites the
        group with the result of the function.

        Parameters
        ----------
        fun : callable
            Function to be applied to each group. Assumes the function is called as
            ``fun(dataset, *args, **kwargs)``.
        groups : str or list of str, optional
            Groups where the selection is to be applied. Can either be group names
            or metagroup names.
        filter_groups : {None, "like", "regex"}, optional
            If `None` (default), interpret var_names as the real variables names. If "like",
            interpret var_names as substrings of the real variables names. If "regex",
            interpret var_names as regular expressions on the real variables names. A la
            `pandas.filter`.
        inplace : bool, optional
            If ``True``, modify the InferenceData object inplace,
            otherwise, return the modified copy.
        args : array_like, optional
            Positional arguments passed to ``fun``.
        **kwargs : mapping, optional
            Keyword arguments passed to ``fun``.

        Returns
        -------
        InferenceData
            A new InferenceData object by default.
            When `inplace==True` perform selection in place and return `None`

        Examples
        --------
        Shift observed_data, prior_predictive and posterior_predictive.

        .. ipython::

            In [1]: import arviz as az
               ...: idata = az.load_arviz_data("non_centered_eight")
               ...: idata_shifted_obs = idata.map(lambda x: x + 3, groups="observed_vars")
               ...: print(idata_shifted_obs.observed_data)
               ...: print(idata_shifted_obs.posterior_predictive)

        Rename and update the coordinate values in both posterior and prior groups.

        .. ipython::

            In [1]: idata = az.load_arviz_data("radon")
               ...: idata = idata.map(
               ...:     lambda ds: ds.rename({"gamma_dim_0": "uranium_coefs"}).assign(
               ...:         uranium_coefs=["intercept", "u_slope", "xbar_slope"]
               ...:     ),
               ...:     groups=["posterior", "prior"]
               ...: )
               ...: idata.posterior

        Add extra coordinates to all groups containing observed variables

        .. ipython::

            In [1]: idata = az.load_arviz_data("rugby")
               ...: home_team, away_team = np.array([
               ...:     m.split() for m in idata.observed_data.match.values
               ...: ]).T
               ...: idata = idata.map(
               ...:     lambda ds, **kwargs: ds.assign_coords(**kwargs),
               ...:     groups="observed_vars",
               ...:     home_team=("match", home_team),
               ...:     away_team=("match", away_team),
               ...: )
               ...: print(idata.posterior_predictive)
               ...: print(idata.observed_data)

        """
        if args is None:
            args = []
        groups = self._group_names(groups, filter_groups)

        out = self if inplace else deepcopy(self)
        for group in groups:
            dataset = getattr(self, group)
            dataset = fun(dataset, *args, **kwargs)
            setattr(out, group, dataset)
        if inplace:
            return None
        else:
            return out

    def _wrap_xarray_method(
        self, method, groups=None, filter_groups=None, inplace=False, args=None, **kwargs
    ):
        """Extend and xarray.Dataset method to InferenceData object.

        Parameters
        ----------
        method: str
            Method to be extended. Must be a ``xarray.Dataset`` method.
        groups: str or list of str, optional
            Groups where the selection is to be applied. Can either be group names
            or metagroup names.
        inplace: bool, optional
            If ``True``, modify the InferenceData object inplace,
            otherwise, return the modified copy.
        **kwargs: mapping, optional
            Keyword arguments passed to the xarray Dataset method.

        Returns
        -------
        InferenceData
            A new InferenceData object by default.
            When `inplace==True` perform selection in place and return `None`

        Examples
        --------
        Compute the mean of `posterior_groups`:

        .. ipython::

            In [1]: import arviz as az
               ...: idata = az.load_arviz_data("non_centered_eight")
               ...: idata_means = idata._wrap_xarray_method("mean", groups="latent_vars")
               ...: print(idata_means.posterior)
               ...: print(idata_means.observed_data)

        .. ipython::

            In [1]: idata_stack = idata._wrap_xarray_method(
               ...:     "stack",
               ...:     groups=["posterior_groups", "prior_groups"],
               ...:     sample=["chain", "draw"]
               ...: )
               ...: print(idata_stack.posterior)
               ...: print(idata_stack.prior)
               ...: print(idata_stack.observed_data)

        """
        if args is None:
            args = []
        groups = self._group_names(groups, filter_groups)

        method = getattr(xr.Dataset, method)

        out = self if inplace else deepcopy(self)
        for group in groups:
            dataset = getattr(self, group)
            dataset = method(dataset, *args, **kwargs)
            setattr(out, group, dataset)
        if inplace:
            return None
        else:
            return out


# pylint: disable=protected-access, inconsistent-return-statements
def concat(*args, dim=None, copy=True, inplace=False, reset_dim=True):
    """Concatenate InferenceData objects.

    Concatenates over `group`, `chain` or `draw`.
    By default concatenates over unique groups.
    To concatenate over `chain` or `draw` function
    needs identical groups and variables.

    The `variables` in the `data` -group are merged if `dim` are not found.


    Parameters
    ----------
    *args : InferenceData
        Variable length InferenceData list or
        Sequence of InferenceData.
    dim : str, optional
        If defined, concatenated over the defined dimension.
        Dimension which is concatenated. If None, concatenates over
        unique groups.
    copy : bool
        If True, groups are copied to the new InferenceData object.
        Used only if `dim` is None.
    inplace : bool
        If True, merge args to first object.
    reset_dim : bool
        Valid only if dim is not None.

    Returns
    -------
    InferenceData
        A new InferenceData object by default.
        When `inplace==True` merge args to first arg and return `None`

    Examples
    --------
    Use ``concat`` method to concatenate InferenceData objects. This will concatenates over
    unique groups by default. We first create an ``InferenceData`` object:

    .. ipython::

        In [1]: import arviz as az
           ...: import numpy as np
           ...: data = {
           ...:     "a": np.random.normal(size=(4, 100, 3)),
           ...:     "b": np.random.normal(size=(4, 100)),
           ...: }
           ...: coords = {"a_dim": ["x", "y", "z"]}
           ...: dataA = az.from_dict(data, coords=coords, dims={"a": ["a_dim"]})
           ...: dataA

    We have created an ``InferenceData`` object with default group 'posterior'. Now, we will
    create another ``InferenceData`` object:

    .. ipython::

        In [1]: dataB = az.from_dict(prior=data, coords=coords, dims={"a": ["a_dim"]})
           ...: dataB

    We have created another ``InferenceData`` object with group 'prior'. Now, we will concatenate
    these two ``InferenceData`` objects:

    .. ipython::

        In [1]: az.concat(dataA, dataB)

    Now, we will concatenate over chain (or draw). It requires identical groups and variables.
    Here we are concatenating two identical ``InferenceData`` objects over dimension chain:

    .. ipython::

        In [1]: az.concat(dataA, dataA, dim="chain")

    It will create an ``InferenceData`` with the original group 'posterior'. In similar way,
    we can also concatenate over draws.

    """
    # pylint: disable=undefined-loop-variable, too-many-nested-blocks
    if len(args) == 0:
        if inplace:
            return
        return InferenceData()

    if len(args) == 1 and isinstance(args[0], Sequence):
        args = args[0]

    # assert that all args are InferenceData
    for i, arg in enumerate(args):
        if not isinstance(arg, InferenceData):
            raise TypeError(
                "Concatenating is supported only"
                "between InferenceData objects. Input arg {} is {}".format(i, type(arg))
            )

    if dim is not None and dim.lower() not in {"group", "chain", "draw"}:
        msg = "Invalid `dim`: {}. Valid `dim` are {}".format(dim, '{"group", "chain", "draw"}')
        raise TypeError(msg)
    dim = dim.lower() if dim is not None else dim

    if len(args) == 1 and isinstance(args[0], InferenceData):
        if inplace:
            return None
        else:
            if copy:
                return deepcopy(args[0])
            else:
                return args[0]

    current_time = str(datetime.now())

    if not inplace:
        # Keep order for python 3.5
        inference_data_dict = OrderedDict()

    if dim is None:
        arg0 = args[0]
        arg0_groups = ccopy(arg0._groups_all)
        args_groups = dict()
        # check if groups are independent
        # Concat over unique groups
        for arg in args[1:]:
            for group in arg._groups_all:
                if group in args_groups or group in arg0_groups:
                    msg = (
                        "Concatenating overlapping groups is not supported unless `dim` is defined."
                    )
                    msg += " Valid dimensions are `chain` and `draw`."
                    raise TypeError(msg)
                group_data = getattr(arg, group)
                args_groups[group] = deepcopy(group_data) if copy else group_data
        # add arg0 to args_groups if inplace is False
        # otherwise it will merge args_groups to arg0
        # inference data object
        if not inplace:
            for group in arg0_groups:
                group_data = getattr(arg0, group)
                args_groups[group] = deepcopy(group_data) if copy else group_data

        other_groups = [group for group in args_groups if group not in SUPPORTED_GROUPS_ALL]

        for group in SUPPORTED_GROUPS_ALL + other_groups:
            if group not in args_groups:
                continue
            if inplace:
                if group.startswith(WARMUP_TAG):
                    arg0._groups_warmup.append(group)
                else:
                    arg0._groups.append(group)
                setattr(arg0, group, args_groups[group])
            else:
                inference_data_dict[group] = args_groups[group]
        if inplace:
            other_groups = [
                group for group in arg0_groups if group not in SUPPORTED_GROUPS_ALL
            ] + other_groups
            sorted_groups = [
                group for group in SUPPORTED_GROUPS + other_groups if group in arg0._groups
            ]
            setattr(arg0, "_groups", sorted_groups)
            sorted_groups_warmup = [
                group
                for group in SUPPORTED_GROUPS_WARMUP + other_groups
                if group in arg0._groups_warmup
            ]
            setattr(arg0, "_groups_warmup", sorted_groups_warmup)
    else:
        arg0 = args[0]
        arg0_groups = arg0._groups_all
        for arg in args[1:]:
            for group0 in arg0_groups:
                if group0 not in arg._groups_all:
                    if group0 == "observed_data":
                        continue
                    msg = "Mismatch between the groups."
                    raise TypeError(msg)
            for group in arg._groups_all:
                # handle data groups seperately
                if group not in ["observed_data", "constant_data", "predictions_constant_data"]:
                    # assert that groups are equal
                    if group not in arg0_groups:
                        msg = "Mismatch between the groups."
                        raise TypeError(msg)

                    # assert that variables are equal
                    group_data = getattr(arg, group)
                    group_vars = group_data.data_vars

                    if not inplace and group in inference_data_dict:
                        group0_data = inference_data_dict[group]
                    else:
                        group0_data = getattr(arg0, group)
                    group0_vars = group0_data.data_vars

                    for var in group0_vars:
                        if var not in group_vars:
                            msg = "Mismatch between the variables."
                            raise TypeError(msg)

                    for var in group_vars:
                        if var not in group0_vars:
                            msg = "Mismatch between the variables."
                            raise TypeError(msg)
                        var_dims = getattr(group_data, var).dims
                        var0_dims = getattr(group0_data, var).dims
                        if var_dims != var0_dims:
                            msg = "Mismatch between the dimensions."
                            raise TypeError(msg)

                        if dim not in var_dims or dim not in var0_dims:
                            msg = "Dimension {} missing.".format(dim)
                            raise TypeError(msg)

                    # xr.concat
                    concatenated_group = xr.concat((group_data, group0_data), dim=dim)
                    if reset_dim:
                        concatenated_group[dim] = range(concatenated_group[dim].size)

                    # handle attrs
                    if hasattr(group0_data, "attrs"):
                        group0_attrs = deepcopy(getattr(group0_data, "attrs"))
                    else:
                        group0_attrs = OrderedDict()

                    if hasattr(group_data, "attrs"):
                        group_attrs = getattr(group_data, "attrs")
                    else:
                        group_attrs = dict()

                    # gather attrs results to group0_attrs
                    for attr_key, attr_values in group_attrs.items():
                        group0_attr_values = group0_attrs.get(attr_key, None)
                        equality = attr_values == group0_attr_values
                        if hasattr(equality, "__iter__"):
                            equality = np.all(equality)
                        if equality:
                            continue
                        # handle special cases:
                        if attr_key in ("created_at", "previous_created_at"):
                            # check the defaults
                            if not hasattr(group0_attrs, "previous_created_at"):
                                group0_attrs["previous_created_at"] = []
                                if group0_attr_values is not None:
                                    group0_attrs["previous_created_at"].append(group0_attr_values)
                            # check previous values
                            if attr_key == "previous_created_at":
                                if not isinstance(attr_values, list):
                                    attr_values = [attr_values]
                                group0_attrs["previous_created_at"].extend(attr_values)
                                continue
                            # update "created_at"
                            if group0_attr_values != current_time:
                                group0_attrs[attr_key] = current_time
                            group0_attrs["previous_created_at"].append(attr_values)

                        elif attr_key in group0_attrs:
                            combined_key = "combined_{}".format(attr_key)
                            if combined_key not in group0_attrs:
                                group0_attrs[combined_key] = [group0_attr_values]
                            group0_attrs[combined_key].append(attr_values)
                        else:
                            group0_attrs[attr_key] = attr_values
                    # update attrs
                    setattr(concatenated_group, "attrs", group0_attrs)

                    if inplace:
                        setattr(arg0, group, concatenated_group)
                    else:
                        inference_data_dict[group] = concatenated_group
                else:
                    # observed_data, "constant_data", "predictions_constant_data",
                    if group not in arg0_groups:
                        setattr(arg0, group, deepcopy(group_data) if copy else group_data)
                        arg0._groups.append(group)
                        continue

                    # assert that variables are equal
                    group_data = getattr(arg, group)
                    group_vars = group_data.data_vars

                    group0_data = getattr(arg0, group)
                    if not inplace:
                        group0_data = deepcopy(group0_data)
                    group0_vars = group0_data.data_vars

                    for var in group_vars:
                        if var not in group0_vars:
                            var_data = getattr(group_data, var)
                            getattr(arg0, group)[var] = var_data
                        else:
                            var_data = getattr(group_data, var)
                            var0_data = getattr(group0_data, var)
                            if dim in var_data.dims and dim in var0_data.dims:
                                concatenated_var = xr.concat((group_data, group0_data), dim=dim)
                                group0_data[var] = concatenated_var

                    # handle attrs
                    if hasattr(group0_data, "attrs"):
                        group0_attrs = getattr(group0_data, "attrs")
                    else:
                        group0_attrs = OrderedDict()

                    if hasattr(group_data, "attrs"):
                        group_attrs = getattr(group_data, "attrs")
                    else:
                        group_attrs = dict()

                    # gather attrs results to group0_attrs
                    for attr_key, attr_values in group_attrs.items():
                        group0_attr_values = group0_attrs.get(attr_key, None)
                        equality = attr_values == group0_attr_values
                        if hasattr(equality, "__iter__"):
                            equality = np.all(equality)
                        if equality:
                            continue
                        # handle special cases:
                        if attr_key in ("created_at", "previous_created_at"):
                            # check the defaults
                            if not hasattr(group0_attrs, "previous_created_at"):
                                group0_attrs["previous_created_at"] = []
                                if group0_attr_values is not None:
                                    group0_attrs["previous_created_at"].append(group0_attr_values)
                            # check previous values
                            if attr_key == "previous_created_at":
                                if not isinstance(attr_values, list):
                                    attr_values = [attr_values]
                                group0_attrs["previous_created_at"].extend(attr_values)
                                continue
                            # update "created_at"
                            if group0_attr_values != current_time:
                                group0_attrs[attr_key] = current_time
                            group0_attrs["previous_created_at"].append(attr_values)

                        elif attr_key in group0_attrs:
                            combined_key = "combined_{}".format(attr_key)
                            if combined_key not in group0_attrs:
                                group0_attrs[combined_key] = [group0_attr_values]
                            group0_attrs[combined_key].append(attr_values)

                        else:
                            group0_attrs[attr_key] = attr_values
                    # update attrs
                    setattr(group0_data, "attrs", group0_attrs)

                    if inplace:
                        setattr(arg0, group, group0_data)
                    else:
                        inference_data_dict[group] = group0_data

    return None if inplace else InferenceData(**inference_data_dict)
