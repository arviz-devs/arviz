"""Data structure for using netcdf groups with xarray."""
from collections import OrderedDict, defaultdict
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


# pylint: disable=protected-access
def concat(*args, level="chain", copy=True, inplace=False):
    """Concatenate InferenceData objects on a group or chain level.

    Parameters
    ----------
    *args : InferenceData
        Variable length InferenceData list or
        Sequence of InferenceData.
    level : str
        Defines subset for which the combination is valid.
        - "group" Concatenate only unique groups
        - "chain" Concatenate against chain dimension (default)
        - "draw" Concatenate against draw dimension
        - user defined dimension
    copy : bool
        If True, groups are copied to the new InferenceData object.
    inplace : bool
        If True and level=="group", merge args to first object.

    Returns
    -------
    InferenceData
        A new InferenceData object by default.
        When `inplace==True` merge args to first arg and return `None`
    """
    if not isinstance(level, str):
        raise TypeError("level must be a string object")
    level = level.lower()
    if inplace and level != "group":
        raise TypeError('inplace supported only when level=="group"')
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

    # extract first argument
    # and argument groups
    farg = args[0]
    farg_groups = ccopy(farg._groups)
    args_groups = defaultdict(list)
    # add first arg to argument groups if inplace is False
    if not inplace:
        for group in farg_groups:
            group_data = getattr(farg, group)
            args_groups[group].append(deepcopy(group_data) if copy else group_data)

    # handle the rest of the arguments
    for arg in args[1:]:
        for group in arg._groups:
            if level == "group" and (group in args_groups or group in farg_groups):
                raise TypeError('Only unique groups supported when level=="group"')
            group_data = getattr(arg, group)
            args_groups[group].append(deepcopy(group_data) if copy else group_data)

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
            if group not in farg._groups:
                farg._groups.append(group)
                setattr(farg, group, args_groups[group])
            else:
                # violates inplace
                index = farg._groups.index(group)
                objs = [getattr(farg, group)] + args_groups[group]
                farg._groups[index] = xr.concat(objs, dim=level)
        else:
            objs = args_groups[group]
            inference_data_dict[group] = xr.concat(objs, dim=level)
    if inplace:
        other_groups = [group for group in farg_groups if group not in basic_order] + other_groups
        sorted_groups = [group for group in basic_order + other_groups if group in farg._groups]
        setattr(farg, "_groups", sorted_groups)
        return None
    return InferenceData(**inference_data_dict)
