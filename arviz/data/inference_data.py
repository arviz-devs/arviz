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
def concat(*args, copy=True, inplace=False):
    """Concatenate InferenceData objects on a group level.

    Supports only concatenating with independent unique groups.

    Parameters
    ----------
    *args : InferenceData
        Variable length InferenceData list or
        Sequence of InferenceData.
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
        return InferenceData()
    if len(args) == 1 and isinstance(args[0], Sequence):
        args = args[0]
    elif len(args) == 1:
        if isinstance(args[0], InferenceData):
            if inplace:
                return None
            else:
                if copy:
                    return deepcopy(args[0])
                else:
                    return args[0]

    # assert that all args are InferenceData
    for i, arg in enumerate(args):
        if not isinstance(arg, InferenceData):
            raise TypeError(
                "Concatenating is supported only"
                "between InferenceData objects. Input arg {} is {}".format(i, type(arg))
            )
    # assert that groups are independent
    first_arg = args[0]
    first_arg_groups = ccopy(first_arg._groups)
    args_groups = dict()
    for arg in args[1:]:
        for group in arg._groups:
            if group in args_groups or group in first_arg_groups:
                raise NotImplementedError("Concatenating with overlapping groups is not supported.")
            group_data = getattr(arg, group)
            args_groups[group] = deepcopy(group_data) if copy else group_data

    # add first_arg to args_groups if inplace is False
    if not inplace:
        for group in first_arg_groups:
            group_data = getattr(first_arg, group)
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

    if not inplace:
        # Keep order for python 3.5
        inference_data_dict = OrderedDict()
    for group in basic_order + other_groups:
        if group not in args_groups:
            continue
        if inplace:
            first_arg._groups.append(group)
            setattr(first_arg, group, args_groups[group])
        else:
            inference_data_dict[group] = args_groups[group]
    if inplace:
        other_groups = [
            group for group in first_arg_groups if group not in basic_order
        ] + other_groups
        sorted_groups = [
            group for group in basic_order + other_groups if group in first_arg._groups
        ]
        setattr(first_arg, "_groups", sorted_groups)
        return None
    return InferenceData(**inference_data_dict)
