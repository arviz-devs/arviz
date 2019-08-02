"""Data structure for using netcdf groups with xarray."""
from collections import OrderedDict
from collections.abc import Sequence
from copy import copy as ccopy, deepcopy
from datetime import datetime
import netCDF4 as nc
import numpy as np
import xarray as xr

from ..rcparams import rcParams


class InferenceData:
    """Container for accessing netCDF files using xarray."""

    def __init__(self, **kwargs):
        """Initialize InferenceData object from keyword xarray datasets.

        Examples
        --------
        InferenceData(posterior=posterior, prior=prior)

        Parameters
        ----------
        kwargs :
            Keyword arguments of xarray datasets
        """
        self._groups = []
        for key, dataset in kwargs.items():
            if dataset is None:
                continue
            elif not isinstance(dataset, xr.Dataset):
                raise ValueError(
                    "Arguments to InferenceData must be xarray Datasets "
                    '(argument "{}" was type "{}")'.format(key, type(dataset))
                )
            setattr(self, key, dataset)
            self._groups.append(key)

    def __repr__(self):
        """Make string representation of object."""
        return "Inference data with groups:\n\t> {options}".format(
            options="\n\t> ".join(self._groups)
        )

    def __delattr__(self, group):
        """Delete a group from the InferenceData object."""
        self._groups.remove(group)
        object.__delattr__(self, group)

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
        if self._groups:  # check's whether a group is present or not.
            if groups is None:
                groups = self._groups
            else:
                groups = [group for group in self._groups if group in groups]
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

    def sel(self, inplace=True, **kwargs):
        """Perform an xarray selection on all groups.

        Loops over all groups to perform Dataset.sel(key=item)
        for every kwarg if key is a dimension of the dataset.
        The selection is performed inplace.

        Parameters
        ----------
        inplace : bool
            If True, modify the InferenceData object inplace, otherwise, return the modified copy.
        **kwargs : mapping
            It must be accepted by Dataset.sel()
        """
        out = self if inplace else deepcopy(self)
        for group in self._groups:
            dataset = getattr(self, group)
            valid_keys = set(kwargs.keys()).intersection(dataset.dims)
            dataset = dataset.sel(**{key: kwargs[key] for key in valid_keys})
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
        arg0_groups = ccopy(arg0._groups)
        args_groups = dict()
        # check if groups are independent
        # Concat over unique groups
        for arg in args[1:]:
            for group in arg._groups:
                if group in args_groups or group in arg0_groups:
                    msg = (
                        "Concatenating overlapping groups is not supported unless `dim` is defined."
                    )
                    msg += " Valid dimensions are `chain` and `draw`."
                    raise TypeError(msg)
            group_data = getattr(arg, group)
            args_groups[group] = deepcopy(group_data) if copy else group_data
        # add arg0 to args_groups if inplace is False
        if not inplace:
            for group in arg0_groups:
                group_data = getattr(arg0, group)
                args_groups[group] = deepcopy(group_data) if copy else group_data

        basic_order = [
            "posterior",
            "posterior_predictive",
            "sample_stats",
            "prior",
            "prior_predictive",
            "sample_stats_prior",
            "observed_data",
        ]
        other_groups = [group for group in args_groups if group not in basic_order]

        for group in basic_order + other_groups:
            if group not in args_groups:
                continue
            if inplace:
                arg0._groups.append(group)
                setattr(arg0, group, args_groups[group])
            else:
                inference_data_dict[group] = args_groups[group]
        if inplace:
            other_groups = [
                group for group in arg0_groups if group not in basic_order
            ] + other_groups
            sorted_groups = [group for group in basic_order + other_groups if group in arg0._groups]
            setattr(arg0, "_groups", sorted_groups)
    else:
        arg0 = args[0]
        arg0_groups = arg0._groups
        for arg in args[1:]:
            for group0 in arg0_groups:
                if group0 not in arg._groups:
                    if group0 == "observed_data":
                        continue
                    msg = "Mismatch between the groups."
                    raise TypeError(msg)
            for group in arg._groups:
                if group != "observed_data":
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
                    # observed_data
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
                            arg0.observed_data[var] = var_data
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
