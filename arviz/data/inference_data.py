"""Data structure for using netcdf groups with xarray."""
from collections import OrderedDict
from collections.abc import Sequence
from copy import copy as ccopy, deepcopy
import netCDF4 as nc
import xarray as xr


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

    @staticmethod
    def from_netcdf(filename):
        """Initialize object from a netcdf file.

        Expects that the file will have groups, each of which can be loaded by xarray.

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
                groups[group] = data
        return InferenceData(**groups)

    def to_netcdf(self, filename, compress=True):
        """Write InferenceData to file using netcdf4.

        Parameters
        ----------
        filename : str
            Location to write to
        compress : bool
            Whether to compress result. Note this saves disk space, but may make
            saving and loading somewhat slower (default: True).

        Returns
        -------
        str
            Location of netcdf file
        """
        mode = "w"  # overwrite first, then append
        if self._groups:  # check's whether a group is present or not.
            for group in self._groups:
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


# pylint: disable=protected-access
def concat(*args, dim=None, copy=True, inplace=False):
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
    inplace : bool
        If True, merge args to first object.

    Returns
    -------
    InferenceData
        A new InferenceData object by default.
        When `inplace==True` merge args to first arg and return `None`
    """
    if len(args) == 0:
        if inplace:
            return
        return InferenceData()

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

    if len(args) == 1 and isinstance(args[0], Sequence):
        args = args[0]
    elif len(args) == 1:
        if isinstance(args[0], InferenceData):
            if inplace:
                return
            else:
                if copy:
                    return deepcopy(args[0])
                else:
                    return args[0]


    arg0 = args[0]
    arg0_groups = ccopy(arg0._groups)
    args_groups = dict()

    if not inplace:
        # Keep order for python 3.5
        inference_data_dict = OrderedDict()

    if dim is None:
        # check if groups are independent
        # Concat over unique groups
        for arg in args[1:]:
            for group in arg._groups:
                if group in args_groups or group in arg0_groups:
                    msg = "Concatenating overlapping groups is not supported unless `dim` is defined."
                    msg += " Valid dimensions are `chain` and `draw`."
                    raise TypeError(msg)
            group_data = getattr(arg, group)
            args_groups[group] = deepcopy(group_data) if copy else group_data
        # add arg0 to args_groups if inplace is False
        if not inplace:
            for group in arg0:
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
            sorted_groups = [
                group for group in basic_order + other_groups if group in arg0._groups
            ]
            setattr(arg0, "_groups", sorted_groups)
    elif dim in ("chain", "draw"):
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

                    group0_data = getattr(arg0, group)
                    group0_vars = group0_data.data_vars

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
                    if inplace:
                        setattr(arg0, group, concatenated_group)
                    else:
                        inference_data_dict[group] = concatenated_group
                else:
                    # observed_data
                    if (group not in arg0_groups) and inplace:
                        msg = "observed_data missing from the groups."
                        raise TypeError(msg)

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
                    if inplace:
                        setattr(arg0, group, group0_data)
                    else:
                        inference_data_dict[group] = group0_data

    return None if inplace else InferenceData(**inference_data_dict)
