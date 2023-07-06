# pylint: disable=too-many-lines,too-many-public-methods
"""Data structure for using netcdf groups with xarray."""
import re
import sys
import uuid
import warnings
from collections import OrderedDict, defaultdict
from collections.abc import MutableMapping, Sequence
from copy import copy as ccopy
from copy import deepcopy
from datetime import datetime
from html import escape
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)
import os

import numpy as np
import xarray as xr
from packaging import version

from ..rcparams import rcParams
from ..utils import HtmlTemplate, _subset_list, either_dict_or_kwargs
from .base import _extend_xr_method, _make_json_serializable, dict_to_dataset

if sys.version_info[:2] >= (3, 9):
    # As of 3.9, collections.abc types support generic parameters themselves.
    from collections.abc import ItemsView, ValuesView
else:
    # These typing imports are deprecated in 3.9, and moved to collections.abc instead.
    from typing import ItemsView, ValuesView

if TYPE_CHECKING:
    from typing_extensions import Literal

try:
    import ujson as json
except ImportError:
    # mypy struggles with conditional imports expressed as catching ImportError:
    # https://github.com/python/mypy/issues/1153
    import json  # type: ignore


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
    f"{WARMUP_TAG}posterior",
    f"{WARMUP_TAG}posterior_predictive",
    f"{WARMUP_TAG}predictions",
    f"{WARMUP_TAG}sample_stats",
    f"{WARMUP_TAG}log_likelihood",
]

SUPPORTED_GROUPS_ALL = SUPPORTED_GROUPS + SUPPORTED_GROUPS_WARMUP

InferenceDataT = TypeVar("InferenceDataT", bound="InferenceData")


def _compressible_dtype(dtype):
    """Check basic dtypes for automatic compression."""
    if dtype.kind == "V":
        return all(_compressible_dtype(item) for item, _ in dtype.fields.values())
    return dtype.kind in {"b", "i", "u", "f", "c", "S"}


class InferenceData(Mapping[str, xr.Dataset]):
    """Container for inference data storage using xarray.

    For a detailed introduction to ``InferenceData`` objects and their usage, see
    :ref:`xarray_for_arviz`. This page provides help and documentation
    on ``InferenceData`` methods and their low level implementation.
    """

    def __init__(
        self,
        attrs: Union[None, Mapping[Any, Any]] = None,
        **kwargs: Union[xr.Dataset, List[xr.Dataset], Tuple[xr.Dataset, xr.Dataset]],
    ) -> None:
        """Initialize InferenceData object from keyword xarray datasets.

        Parameters
        ----------
        attrs : dict
            sets global attribute for InferenceData object.
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
        self._groups: List[str] = []
        self._groups_warmup: List[str] = []
        self._attrs: Union[None, dict] = dict(attrs) if attrs is not None else None
        save_warmup = kwargs.pop("save_warmup", False)
        key_list = [key for key in SUPPORTED_GROUPS_ALL if key in kwargs]
        for key in kwargs:
            if key not in SUPPORTED_GROUPS_ALL:
                key_list.append(key)
                warnings.warn(
                    f"{key} group is not defined in the InferenceData scheme", UserWarning
                )
        for key in key_list:
            dataset = kwargs[key]
            dataset_warmup = None
            if dataset is None:
                continue
            elif isinstance(dataset, (list, tuple)):
                dataset, dataset_warmup = dataset
            elif not isinstance(dataset, xr.Dataset):
                raise ValueError(
                    "Arguments to InferenceData must be xarray Datasets "
                    f"(argument '{key}' was type '{type(dataset)}')"
                )
            if not key.startswith(WARMUP_TAG):
                if dataset:
                    setattr(self, key, dataset)
                    self._groups.append(key)
            elif key.startswith(WARMUP_TAG):
                if dataset:
                    setattr(self, key, dataset)
                    self._groups_warmup.append(key)
            if save_warmup and dataset_warmup is not None and dataset_warmup:
                key = f"{WARMUP_TAG}{key}"
                setattr(self, key, dataset_warmup)
                self._groups_warmup.append(key)

    @property
    def attrs(self) -> dict:
        """Attributes of InferenceData object."""
        if self._attrs is None:
            self._attrs = {}
        return self._attrs

    @attrs.setter
    def attrs(self, value) -> None:
        self._attrs = dict(value)

    def __repr__(self) -> str:
        """Make string representation of InferenceData object."""
        msg = "Inference data with groups:\n\t> {options}".format(
            options="\n\t> ".join(self._groups)
        )
        if self._groups_warmup:
            msg += f"\n\nWarmup iterations saved ({WARMUP_TAG}*)."
        return msg

    def _repr_html_(self) -> str:
        """Make html representation of InferenceData object."""
        try:
            from xarray.core.options import OPTIONS

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
                formatted_html_template = (  # pylint: disable=possibly-unused-variable
                    HtmlTemplate.html_template.format(elements)
                )
                css_template = HtmlTemplate.css_template  # pylint: disable=possibly-unused-variable
                html_repr = f"{locals()['formatted_html_template']}{locals()['css_template']}"
        except:  # pylint: disable=bare-except
            html_repr = f"<pre>{escape(repr(self))}</pre>"
        return html_repr

    def __delattr__(self, group: str) -> None:
        """Delete a group from the InferenceData object."""
        if group in self._groups:
            self._groups.remove(group)
        elif group in self._groups_warmup:
            self._groups_warmup.remove(group)
        object.__delattr__(self, group)

    @property
    def _groups_all(self) -> List[str]:
        return self._groups + self._groups_warmup

    def __len__(self) -> int:
        """Return the number of groups in this InferenceData object."""
        return len(self._groups_all)

    def __iter__(self) -> Iterator[str]:
        """Iterate over groups in InferenceData object."""
        for group in self._groups_all:
            yield group

    def __contains__(self, key: object) -> bool:
        """Return True if the named item is present, and False otherwise."""
        return key in self._groups_all

    def __getitem__(self, key: str) -> xr.Dataset:
        """Get item by key."""
        if key not in self._groups_all:
            raise KeyError(key)
        return getattr(self, key)

    def groups(self) -> List[str]:
        """Return all groups present in InferenceData object."""
        return self._groups_all

    class InferenceDataValuesView(ValuesView[xr.Dataset]):
        """ValuesView implementation for InferenceData, to allow it to implement Mapping."""

        def __init__(  # pylint: disable=super-init-not-called
            self, parent: "InferenceData"
        ) -> None:
            """Create a new InferenceDataValuesView from an InferenceData object."""
            self.parent = parent

        def __len__(self) -> int:
            """Return the number of groups in the parent InferenceData."""
            return len(self.parent._groups_all)

        def __iter__(self) -> Iterator[xr.Dataset]:
            """Iterate through the Xarray datasets present in the InferenceData object."""
            parent = self.parent
            for group in parent._groups_all:
                yield getattr(parent, group)

        def __contains__(self, key: object) -> bool:
            """Return True if the given Xarray dataset is one of the values, and False otherwise."""
            if not isinstance(key, xr.Dataset):
                return False

            for dataset in self:
                if dataset.equals(key):
                    return True

            return False

    def values(self) -> "InferenceData.InferenceDataValuesView":
        """Return a view over the Xarray Datasets present in the InferenceData object."""
        return InferenceData.InferenceDataValuesView(self)

    class InferenceDataItemsView(ItemsView[str, xr.Dataset]):
        """ItemsView implementation for InferenceData, to allow it to implement Mapping."""

        def __init__(  # pylint: disable=super-init-not-called
            self, parent: "InferenceData"
        ) -> None:
            """Create a new InferenceDataItemsView from an InferenceData object."""
            self.parent = parent

        def __len__(self) -> int:
            """Return the number of groups in the parent InferenceData."""
            return len(self.parent._groups_all)

        def __iter__(self) -> Iterator[Tuple[str, xr.Dataset]]:
            """Iterate through the groups and corresponding Xarray datasets in the InferenceData."""
            parent = self.parent
            for group in parent._groups_all:
                yield group, getattr(parent, group)

        def __contains__(self, key: object) -> bool:
            """Return True if the (group, dataset) tuple is present, and False otherwise."""
            parent = self.parent
            if not isinstance(key, tuple) or len(key) != 2:
                return False

            group, dataset = key
            if group not in parent._groups_all:
                return False

            if not isinstance(dataset, xr.Dataset):
                return False

            existing_dataset = getattr(parent, group)
            return existing_dataset.equals(dataset)

    def items(self) -> "InferenceData.InferenceDataItemsView":
        """Return a view over the groups and datasets present in the InferenceData object."""
        return InferenceData.InferenceDataItemsView(self)

    @staticmethod
    def from_netcdf(
        filename,
        *,
        engine="h5netcdf",
        group_kwargs=None,
        regex=False,
        base_group: str = "/",
    ) -> "InferenceData":
        """Initialize object from a netcdf file.

        Expects that the file will have groups, each of which can be loaded by xarray.
        By default, the datasets of the InferenceData object will be lazily loaded instead
        of being loaded into memory. This
        behaviour is regulated by the value of ``az.rcParams["data.load"]``.

        Parameters
        ----------
        filename : str
            location of netcdf file
        engine : {"h5netcdf", "netcdf4"}, default "h5netcdf"
            Library used to read the netcdf file.
        group_kwargs : dict of {str: dict}, optional
            Keyword arguments to be passed into each call of :func:`xarray.open_dataset`.
            The keys of the higher level should be group names or regex matching group
            names, the inner dicts re passed to ``open_dataset``
            This feature is currently experimental.
        regex : bool, default False
            Specifies where regex search should be used to extend the keyword arguments.
            This feature is currently experimental.
        base_group : str, default "/"
            The group in the netCDF file where the InferenceData is stored. By default,
            assumes that the file only contains an InferenceData object.

        Returns
        -------
        InferenceData
        """
        groups = {}
        attrs = {}

        if engine == "h5netcdf":
            import h5netcdf
        elif engine == "netcdf4":
            import netCDF4 as nc
        else:
            raise ValueError(
                f"Invalid value for engine: {engine}. Valid options are: h5netcdf or netcdf4"
            )

        try:
            with h5netcdf.File(filename, mode="r") if engine == "h5netcdf" else nc.Dataset(
                filename, mode="r"
            ) as file_handle:
                if base_group == "/":
                    data = file_handle
                else:
                    data = file_handle[base_group]

                data_groups = list(data.groups)

            for group in data_groups:
                group_kws = {}

                group_kws = {}
                if group_kwargs is not None and regex is False:
                    group_kws = group_kwargs.get(group, {})
                if group_kwargs is not None and regex is True:
                    for key, kws in group_kwargs.items():
                        if re.search(key, group):
                            group_kws = kws
                group_kws.setdefault("engine", engine)
                with xr.open_dataset(filename, group=f"{base_group}/{group}", **group_kws) as data:
                    if rcParams["data.load"] == "eager":
                        groups[group] = data.load()
                    else:
                        groups[group] = data

            with xr.open_dataset(filename, engine=engine, group=base_group) as data:
                attrs.update(data.load().attrs)

            return InferenceData(attrs=attrs, **groups)
        except OSError as err:
            if err.errno == -101:
                raise type(err)(
                    str(err)
                    + (
                        " while reading a NetCDF file. This is probably an error in HDF5, "
                        "which happens because your OS does not support HDF5 file locking.  See "
                        "https://stackoverflow.com/questions/49317927/"
                        "errno-101-netcdf-hdf-error-when-opening-netcdf-file#49317928"
                        " for a possible solution."
                    )
                ) from err
            raise err

    def to_netcdf(
        self,
        filename: str,
        compress: bool = True,
        groups: Optional[List[str]] = None,
        engine: str = "h5netcdf",
        base_group: str = "/",
        overwrite_existing: bool = True,
    ) -> str:
        """Write InferenceData to netcdf4 file.

        Parameters
        ----------
        filename : str
            Location to write to
        compress : bool, optional
            Whether to compress result. Note this saves disk space, but may make
            saving and loading somewhat slower (default: True).
        groups : list, optional
            Write only these groups to netcdf file.
        engine : {"h5netcdf", "netcdf4"}, default "h5netcdf"
            Library used to read the netcdf file.
        base_group : str, default "/"
            The group in the netCDF file where the InferenceData is will be stored.
            By default, will write to the root of the netCDF file
        overwrite_existing : bool, default True
            Whether to overwrite the existing file or append to it.

        Returns
        -------
        str
            Location of netcdf file
        """
        if base_group is None:
            base_group = "/"

        if os.path.exists(filename) and not overwrite_existing:
            mode = "a"
        else:
            mode = "w"  # overwrite first, then append

        if self._attrs:
            xr.Dataset(attrs=self._attrs).to_netcdf(
                filename, mode=mode, engine=engine, group=base_group
            )
            mode = "a"

        if self._groups_all:  # check's whether a group is present or not.
            if groups is None:
                groups = self._groups_all
            else:
                groups = [group for group in self._groups_all if group in groups]

            for group in groups:
                data = getattr(self, group)
                kwargs = {"engine": engine}
                if compress:
                    kwargs["encoding"] = {
                        var_name: {"zlib": True}
                        for var_name, values in data.variables.items()
                        if _compressible_dtype(values.dtype)
                    }
                data.to_netcdf(filename, mode=mode, group=f"{base_group}/{group}", **kwargs)
                data.close()
                mode = "a"
        elif not self._attrs:  # creates a netcdf file for an empty InferenceData object.
            if engine == "h5netcdf":
                import h5netcdf

                empty_netcdf_file = h5netcdf.File(filename, mode="w")
            elif engine == "netcdf4":
                import netCDF4 as nc

                empty_netcdf_file = nc.Dataset(filename, mode="w", format="NETCDF4")
            empty_netcdf_file.close()
        return filename

    def to_datatree(self):
        """Convert InferenceData object to a :class:`~datatree.DataTree`."""
        try:
            from datatree import DataTree
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                "datatree must be installed in order to use InferenceData.to_datatree"
            ) from err
        return DataTree.from_dict({group: ds for group, ds in self.items()})

    @staticmethod
    def from_datatree(datatree):
        """Create an InferenceData object from a :class:`~datatree.DataTree`.

        Parameters
        ----------
        datatree : DataTree
        """
        return InferenceData(**{group: sub_dt.to_dataset() for group, sub_dt in datatree.items()})

    def to_dict(self, groups=None, filter_groups=None):
        """Convert InferenceData to a dictionary following xarray naming conventions.

        Parameters
        ----------
        groups : list, optional
            Groups where the transformation is to be applied. Can either be group names
            or metagroup names.
        filter_groups: {None, "like", "regex"}, optional, default=None
            If `None` (default), interpret groups as the real group or metagroup names.
            If "like", interpret groups as substrings of the real group or metagroup names.
            If "regex", interpret groups as regular expressions on the real group or
            metagroup names. A la `pandas.filter`.

        Returns
        -------
        dict
            A dictionary containing all groups of InferenceData object.
            When `data=False` return just the schema.
        """
        ret = defaultdict(dict)
        if self._groups_all:  # check's whether a group is present or not.
            if groups is None:
                groups = self._group_names(groups, filter_groups)
            else:
                groups = [group for group in self._groups_all if group in groups]

            for group in groups:
                dataset = getattr(self, group)
                data = {}
                for var_name, dataarray in dataset.items():
                    data[var_name] = dataarray.values
                    dims = []
                    for coord_name, coord_values in dataarray.coords.items():
                        if coord_name not in ("chain", "draw") and not coord_name.startswith(
                            f"{var_name}_dim_"
                        ):
                            dims.append(coord_name)
                            ret["coords"][coord_name] = coord_values.values

                    if group in (
                        "predictions",
                        "predictions_constant_data",
                    ):
                        dims_key = "pred_dims"
                    else:
                        dims_key = "dims"
                    if len(dims) > 0:
                        ret[dims_key][var_name] = dims
                    ret[group] = data
                ret[f"{group}_attrs"] = dataset.attrs

        ret["attrs"] = self.attrs
        return ret

    def to_json(self, filename, groups=None, filter_groups=None, **kwargs):
        """Write InferenceData to a json file.

        Parameters
        ----------
        filename : str
            Location to write to
        groups : list, optional
            Groups where the transformation is to be applied. Can either be group names
            or metagroup names.
        filter_groups: {None, "like", "regex"}, optional, default=None
            If `None` (default), interpret groups as the real group or metagroup names.
            If "like", interpret groups as substrings of the real group or metagroup names.
            If "regex", interpret groups as regular expressions on the real group or
            metagroup names. A la `pandas.filter`.
        kwargs : dict
            kwargs passed to json.dump()

        Returns
        -------
        str
            Location of json file
        """
        idata_dict = _make_json_serializable(
            self.to_dict(groups=groups, filter_groups=filter_groups)
        )

        with open(filename, "w", encoding="utf8") as file:
            json.dump(idata_dict, file, **kwargs)

        return filename

    def to_dataframe(
        self,
        groups=None,
        filter_groups=None,
        include_coords=True,
        include_index=True,
        index_origin=None,
    ):
        """Convert InferenceData to a :class:`pandas.DataFrame` following xarray naming conventions.

        This returns dataframe in a "wide" -format, where each item in ndimensional array is
        unpacked. To access "tidy" -format, use xarray functionality found for each dataset.

        In case of a multiple groups, function adds a group identification to the var name.

        Data groups ("observed_data", "constant_data", "predictions_constant_data") are
        skipped implicitly.

        Raises TypeError if no valid groups are found.

        Parameters
        ----------
        groups: str or list of str, optional
            Groups where the transformation is to be applied. Can either be group names
            or metagroup names.
        filter_groups: {None, "like", "regex"}, optional, default=None
            If `None` (default), interpret groups as the real group or metagroup names.
            If "like", interpret groups as substrings of the real group or metagroup names.
            If "regex", interpret groups as regular expressions on the real group or
            metagroup names. A la `pandas.filter`.
        include_coords: bool
            Add coordinate values to column name (tuple).
        include_index: bool
            Add index information for multidimensional arrays.
        index_origin: {0, 1}, optional
            Starting index  for multidimensional objects. 0- or 1-based.
            Defaults to rcParams["data.index_origin"].

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame containing all selected groups of InferenceData object.
        """
        # pylint: disable=too-many-nested-blocks
        if not include_coords and not include_index:
            raise TypeError("Both include_coords and include_index can not be False.")
        if index_origin is None:
            index_origin = rcParams["data.index_origin"]
        if index_origin not in [0, 1]:
            raise TypeError(f"index_origin must be 0 or 1, saw {index_origin}")

        group_names = list(
            filter(lambda x: "data" not in x, self._group_names(groups, filter_groups))
        )

        if not group_names:
            raise TypeError(f"No valid groups found: {groups}")

        dfs = {}
        for group in group_names:
            dataset = self[group]
            df = None
            coords_to_idx = {
                name: dict(map(reversed, enumerate(dataset.coords[name].values, index_origin)))
                for name in list(filter(lambda x: x not in ("chain", "draw"), dataset.coords))
            }
            for data_array in dataset.values():
                dataframe = data_array.to_dataframe()
                if list(filter(lambda x: x not in ("chain", "draw"), data_array.dims)):
                    levels = [
                        idx
                        for idx, dim in enumerate(data_array.dims)
                        if dim not in ("chain", "draw")
                    ]
                    dataframe = dataframe.unstack(level=levels)
                    tuple_columns = []
                    for name, *coords in dataframe.columns:
                        if include_index:
                            idxs = []
                            for coordname, coorditem in zip(dataframe.columns.names[1:], coords):
                                idxs.append(coords_to_idx[coordname][coorditem])
                            if include_coords:
                                tuple_columns.append(
                                    (f"{name}[{','.join(map(str, idxs))}]", *coords)
                                )
                            else:
                                tuple_columns.append(f"{name}[{','.join(map(str, idxs))}]")
                        else:
                            tuple_columns.append((name, *coords))

                    dataframe.columns = tuple_columns
                    dataframe.sort_index(axis=1, inplace=True)
                if df is None:
                    df = dataframe
                    continue
                df = df.join(dataframe, how="outer")
            df = df.reset_index()
            dfs[group] = df
        if len(dfs) > 1:
            for group, df in dfs.items():
                df.columns = [
                    col
                    if col in ("draw", "chain")
                    else (group, *col)
                    if isinstance(col, tuple)
                    else (group, col)
                    for col in df.columns
                ]
            dfs, *dfs_tail = list(dfs.values())
            for df in dfs_tail:
                dfs = dfs.merge(df, how="outer", copy=False)
        else:
            (dfs,) = dfs.values()  # pylint: disable=unbalanced-dict-unpacking
        return dfs

    def to_zarr(self, store=None):
        """Convert InferenceData to a :class:`zarr.hierarchy.Group`.

        The zarr storage is using the same group names as the InferenceData.

        Raises
        ------
        TypeError
            If no valid store is found.

        Parameters
        ----------
        store: zarr.storage i.e MutableMapping or str, optional
            Zarr storage class or path to desired DirectoryStore.

        Returns
        -------
        zarr.hierarchy.group
            A zarr hierarchy group containing the InferenceData.

        References
        ----------
        https://zarr.readthedocs.io/
        """
        try:  # Check zarr
            import zarr

            assert version.parse(zarr.__version__) >= version.parse("2.5.0")
        except (ImportError, AssertionError) as err:
            raise ImportError("'to_zarr' method needs Zarr (2.5.0+) installed.") from err

        # Check store type and create store if necessary
        if store is None:
            store = zarr.storage.TempStore(suffix="arviz")
        elif isinstance(store, str):
            store = zarr.storage.DirectoryStore(path=store)
        elif not isinstance(store, MutableMapping):
            raise TypeError(f"No valid store found: {store}")

        groups = self.groups()

        if not groups:
            raise TypeError("No valid groups found!")

        # order matters here, saving attrs after the groups will erase the groups.
        if self.attrs:
            xr.Dataset(attrs=self.attrs).to_zarr(store=store, mode="w")

        for group in groups:
            # Create zarr group in store with same group name
            getattr(self, group).to_zarr(store=store, group=group, mode="w")

        return zarr.open(store)  # Open store to get overarching group

    @staticmethod
    def from_zarr(store) -> "InferenceData":
        """Initialize object from a zarr store or path.

        Expects that the zarr store will have groups, each of which can be loaded by xarray.
        By default, the datasets of the InferenceData object will be lazily loaded instead
        of being loaded into memory. This
        behaviour is regulated by the value of ``az.rcParams["data.load"]``.

        Parameters
        ----------
        store: MutableMapping or zarr.hierarchy.Group or str.
            Zarr storage class or path to desired Store.

        Returns
        -------
        InferenceData object

        References
        ----------
        https://zarr.readthedocs.io/
        """
        try:
            import zarr

            assert version.parse(zarr.__version__) >= version.parse("2.5.0")
        except (ImportError, AssertionError) as err:
            raise ImportError("'to_zarr' method needs Zarr (2.5.0+) installed.") from err

        # Check store type and create store if necessary
        if isinstance(store, str):
            store = zarr.storage.DirectoryStore(path=store)
        elif isinstance(store, zarr.hierarchy.Group):
            store = store.store
        elif not isinstance(store, MutableMapping):
            raise TypeError(f"No valid store found: {store}")

        groups = {}
        zarr_handle = zarr.open(store, mode="r")

        # Open each group via xarray method
        for key_group, _ in zarr_handle.groups():
            with xr.open_zarr(store=store, group=key_group) as data:
                groups[key_group] = data.load() if rcParams["data.load"] == "eager" else data

        with xr.open_zarr(store=store) as root:
            attrs = root.attrs

        return InferenceData(attrs=attrs, **groups)

    def __add__(self, other: "InferenceData") -> "InferenceData":
        """Concatenate two InferenceData objects."""
        return concat(self, other, copy=True, inplace=False)

    def sel(
        self: InferenceDataT,
        groups: Optional[Union[str, List[str]]] = None,
        filter_groups: Optional["Literal['like', 'regex']"] = None,
        inplace: bool = False,
        chain_prior: Optional[bool] = None,
        **kwargs: Any,
    ) -> Optional[InferenceDataT]:
        """Perform an xarray selection on all groups.

        Loops groups to perform Dataset.sel(key=item)
        for every kwarg if key is a dimension of the dataset.
        One example could be performing a burn in cut on the InferenceData object
        or discarding a chain. The selection is performed on all relevant groups (like
        posterior, prior, sample stats) while non relevant groups like observed data are
        omitted. See :meth:`xarray.Dataset.sel <xarray:xarray.Dataset.sel>`

        Parameters
        ----------
        groups : str or list of str, optional
            Groups where the selection is to be applied. Can either be group names
            or metagroup names.
        filter_groups : {None, "like", "regex"}, optional, default=None
            If `None` (default), interpret groups as the real group or metagroup names.
            If "like", interpret groups as substrings of the real group or metagroup names.
            If "regex", interpret groups as regular expressions on the real group or
            metagroup names. A la `pandas.filter`.
        inplace : bool, optional
            If ``True``, modify the InferenceData object inplace,
            otherwise, return the modified copy.
        chain_prior : bool, optional, deprecated
            If ``False``, do not select prior related groups using ``chain`` dim.
            Otherwise, use selection on ``chain`` if present. Default=False
        kwargs : dict, optional
            It must be accepted by Dataset.sel().

        Returns
        -------
        InferenceData
            A new InferenceData object by default.
            When `inplace==True` perform selection in-place and return `None`

        Examples
        --------
        Use ``sel`` to discard one chain of the InferenceData object. We first check the
        dimensions of the original object:

        .. jupyter-execute::

            import arviz as az
            idata = az.load_arviz_data("centered_eight")
            idata

        In order to remove the third chain:

        .. jupyter-execute::

            idata_subset = idata.sel(chain=[0, 1, 3], groups="posterior_groups")
            idata_subset

        See Also
        --------
        xarray.Dataset.sel :
            Returns a new dataset with each array indexed by tick labels along the specified
            dimension(s).
        isel : Returns a new dataset with each array indexed along the specified dimension(s).
        """
        if chain_prior is not None:
            warnings.warn(
                "chain_prior has been deprecated. Use groups argument and "
                "rcParams['data.metagroups'] instead.",
                DeprecationWarning,
            )
        else:
            chain_prior = False
        group_names = self._group_names(groups, filter_groups)

        out = self if inplace else deepcopy(self)
        for group in group_names:
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

    def isel(
        self: InferenceDataT,
        groups: Optional[Union[str, List[str]]] = None,
        filter_groups: Optional["Literal['like', 'regex']"] = None,
        inplace: bool = False,
        **kwargs: Any,
    ) -> Optional[InferenceDataT]:
        """Perform an xarray selection on all groups.

        Loops groups to perform Dataset.isel(key=item)
        for every kwarg if key is a dimension of the dataset.
        One example could be performing a burn in cut on the InferenceData object
        or discarding a chain. The selection is performed on all relevant groups (like
        posterior, prior, sample stats) while non relevant groups like observed data are
        omitted. See :meth:`xarray:xarray.Dataset.isel`

        Parameters
        ----------
        groups : str or list of str, optional
            Groups where the selection is to be applied. Can either be group names
            or metagroup names.
        filter_groups : {None, "like", "regex"}, optional
            If `None` (default), interpret groups as the real group or metagroup names.
            If "like", interpret groups as substrings of the real group or metagroup names.
            If "regex", interpret groups as regular expressions on the real group or
            metagroup names. A la `pandas.filter`.
        inplace : bool, optional
            If ``True``, modify the InferenceData object inplace,
            otherwise, return the modified copy.
        kwargs : dict, optional
            It must be accepted by :meth:`xarray:xarray.Dataset.isel`.

        Returns
        -------
        InferenceData
            A new InferenceData object by default.
            When `inplace==True` perform selection in-place and return `None`

        Examples
        --------
        Use ``isel`` to discard one chain of the InferenceData object. We first check the
        dimensions of the original object:

        .. jupyter-execute::

            import arviz as az
            idata = az.load_arviz_data("centered_eight")
            idata

        In order to remove the third chain:

        .. jupyter-execute::

            idata_subset = idata.isel(chain=[0, 1, 3], groups="posterior_groups")
            idata_subset

        You can expand the groups and coords in each group to see how now only the chains 0, 1 and
        3 are present.

        See Also
        --------
        xarray.Dataset.isel :
            Returns a new dataset with each array indexed along the specified dimension(s).
        sel :
            Returns a new dataset with each array indexed by tick labels along the specified
            dimension(s).
        """
        group_names = self._group_names(groups, filter_groups)

        out = self if inplace else deepcopy(self)
        for group in group_names:
            dataset = getattr(self, group)
            valid_keys = set(kwargs.keys()).intersection(dataset.dims)
            dataset = dataset.isel(**{key: kwargs[key] for key in valid_keys})
            setattr(out, group, dataset)
        if inplace:
            return None
        else:
            return out

    def stack(
        self,
        dimensions=None,
        groups=None,
        filter_groups=None,
        inplace=False,
        **kwargs,
    ):
        """Perform an xarray stacking on all groups.

        Stack any number of existing dimensions into a single new dimension.
        Loops groups to perform Dataset.stack(key=value)
        for every kwarg if value is a dimension of the dataset.
        The selection is performed on all relevant groups (like
        posterior, prior, sample stats) while non relevant groups like observed data are
        omitted. See :meth:`xarray:xarray.Dataset.stack`

        Parameters
        ----------
        dimensions : dict, optional
            Names of new dimensions, and the existing dimensions that they replace.
        groups: str or list of str, optional
            Groups where the selection is to be applied. Can either be group names
            or metagroup names.
        filter_groups : {None, "like", "regex"}, optional
            If `None` (default), interpret groups as the real group or metagroup names.
            If "like", interpret groups as substrings of the real group or metagroup names.
            If "regex", interpret groups as regular expressions on the real group or
            metagroup names. A la `pandas.filter`.
        inplace : bool, optional
            If ``True``, modify the InferenceData object inplace,
            otherwise, return the modified copy.
        kwargs : dict, optional
            It must be accepted by :meth:`xarray:xarray.Dataset.stack`.

        Returns
        -------
        InferenceData
            A new InferenceData object by default.
            When `inplace==True` perform selection in-place and return `None`

        Examples
        --------
        Use ``stack`` to stack any number of existing dimensions into a single new dimension.
        We first check the original object:

        .. jupyter-execute::

            import arviz as az
            idata = az.load_arviz_data("rugby")
            idata

        In order to stack two dimensions ``chain`` and ``draw`` to ``sample``, we can use:

        .. jupyter-execute::

            idata.stack(sample=["chain", "draw"], inplace=True)
            idata

        We can also take the example of custom InferenceData object and perform stacking. We first
        check the original object:

        .. jupyter-execute::

            import numpy as np
            datadict = {
                "a": np.random.randn(100),
                "b": np.random.randn(1, 100, 10),
                "c": np.random.randn(1, 100, 3, 4),
            }
            coords = {
                "c1": np.arange(3),
                "c99": np.arange(4),
                "b1": np.arange(10),
            }
            dims = {"c": ["c1", "c99"], "b": ["b1"]}
            idata = az.from_dict(
                posterior=datadict, posterior_predictive=datadict, coords=coords, dims=dims
            )
            idata

        In order to stack two dimensions ``c1`` and ``c99`` to ``z``, we can use:

        .. jupyter-execute::

            idata.stack(z=["c1", "c99"], inplace=True)
            idata

        See Also
        --------
        xarray.Dataset.stack : Stack any number of existing dimensions into a single new dimension.
        unstack : Perform an xarray unstacking on all groups of InferenceData object.
        """
        groups = self._group_names(groups, filter_groups)

        dimensions = {} if dimensions is None else dimensions
        dimensions.update(kwargs)
        out = self if inplace else deepcopy(self)
        for group in groups:
            dataset = getattr(self, group)
            kwarg_dict = {}
            for key, value in dimensions.items():
                try:
                    if not set(value).difference(dataset.dims):
                        kwarg_dict[key] = value
                except TypeError:
                    kwarg_dict[key] = value
            dataset = dataset.stack(**kwarg_dict)
            setattr(out, group, dataset)
        if inplace:
            return None
        else:
            return out

    def unstack(self, dim=None, groups=None, filter_groups=None, inplace=False):
        """Perform an xarray unstacking on all groups.

        Unstack existing dimensions corresponding to MultiIndexes into multiple new dimensions.
        Loops groups to perform Dataset.unstack(key=value).
        The selection is performed on all relevant groups (like posterior, prior,
        sample stats) while non relevant groups like observed data are omitted.
        See :meth:`xarray:xarray.Dataset.unstack`

        Parameters
        ----------
        dim : Hashable or iterable of Hashable, optional
            Dimension(s) over which to unstack. By default unstacks all MultiIndexes.
        groups : str or list of str, optional
            Groups where the selection is to be applied. Can either be group names
            or metagroup names.
        filter_groups : {None, "like", "regex"}, optional
            If `None` (default), interpret groups as the real group or metagroup names.
            If "like", interpret groups as substrings of the real group or metagroup names.
            If "regex", interpret groups as regular expressions on the real group or
            metagroup names. A la `pandas.filter`.
        inplace : bool, optional
            If ``True``, modify the InferenceData object inplace,
            otherwise, return the modified copy.

        Returns
        -------
        InferenceData
            A new InferenceData object by default.
            When `inplace==True` perform selection in place and return `None`

        Examples
        --------
        Use ``unstack`` to unstack existing dimensions corresponding to MultiIndexes into
        multiple new dimensions. We first stack two dimensions ``c1`` and ``c99`` to ``z``:

        .. jupyter-execute::

            import arviz as az
            import numpy as np
            datadict = {
                "a": np.random.randn(100),
                "b": np.random.randn(1, 100, 10),
                "c": np.random.randn(1, 100, 3, 4),
            }
            coords = {
                "c1": np.arange(3),
                "c99": np.arange(4),
                "b1": np.arange(10),
            }
            dims = {"c": ["c1", "c99"], "b": ["b1"]}
            idata = az.from_dict(
                posterior=datadict, posterior_predictive=datadict, coords=coords, dims=dims
            )
            idata.stack(z=["c1", "c99"], inplace=True)
            idata

        In order to unstack the dimension ``z``, we use:

        .. jupyter-execute::

            idata.unstack(inplace=True)
            idata

        See Also
        --------
        xarray.Dataset.unstack :
            Unstack existing dimensions corresponding to MultiIndexes into multiple new dimensions.
        stack : Perform an xarray stacking on all groups of InferenceData object.
        """
        groups = self._group_names(groups, filter_groups)
        if isinstance(dim, str):
            dim = [dim]

        out = self if inplace else deepcopy(self)
        for group in groups:
            dataset = getattr(self, group)
            valid_dims = set(dim).intersection(dataset.dims) if dim is not None else dim
            dataset = dataset.unstack(dim=valid_dims)
            setattr(out, group, dataset)
        if inplace:
            return None
        else:
            return out

    def rename(self, name_dict=None, groups=None, filter_groups=None, inplace=False):
        """Perform xarray renaming of variable and dimensions on all groups.

        Loops groups to perform Dataset.rename(name_dict)
        for every key in name_dict if key is a dimension/data_vars of the dataset.
        The renaming is performed on all relevant groups (like
        posterior, prior, sample stats) while non relevant groups like observed data are
        omitted. See :meth:`xarray:xarray.Dataset.rename`

        Parameters
        ----------
        name_dict : dict
            Dictionary whose keys are current variable or dimension names
            and whose values are the desired names.
        groups : str or list of str, optional
            Groups where the selection is to be applied. Can either be group names
            or metagroup names.
        filter_groups : {None, "like", "regex"}, optional
            If `None` (default), interpret groups as the real group or metagroup names.
            If "like", interpret groups as substrings of the real group or metagroup names.
            If "regex", interpret groups as regular expressions on the real group or
            metagroup names. A la `pandas.filter`.
        inplace : bool, optional
            If ``True``, modify the InferenceData object inplace,
            otherwise, return the modified copy.

        Returns
        -------
        InferenceData
            A new InferenceData object by default.
            When `inplace==True` perform renaming in-place and return `None`

        Examples
        --------
        Use ``rename`` to renaming of variable and dimensions on all groups of the InferenceData
        object. We first check the original object:

        .. jupyter-execute::

            import arviz as az
            idata = az.load_arviz_data("rugby")
            idata

        In order to rename the dimensions and variable, we use:

        .. jupyter-execute::

            idata.rename({"team": "team_new", "match":"match_new"}, inplace=True)
            idata

        See Also
        --------
        xarray.Dataset.rename : Returns a new object with renamed variables and dimensions.
        rename_vars :
            Perform xarray renaming of variable or coordinate names on all groups of an
            InferenceData object.
        rename_dims : Perform xarray renaming of dimensions on all groups of InferenceData object.
        """
        groups = self._group_names(groups, filter_groups)
        if "chain" in name_dict.keys() or "draw" in name_dict.keys():
            raise KeyError("'chain' or 'draw' dimensions can't be renamed")
        out = self if inplace else deepcopy(self)

        for group in groups:
            dataset = getattr(self, group)
            expected_keys = list(dataset.data_vars) + list(dataset.dims)
            valid_keys = set(name_dict.keys()).intersection(expected_keys)
            dataset = dataset.rename({key: name_dict[key] for key in valid_keys})
            setattr(out, group, dataset)
        if inplace:
            return None
        else:
            return out

    def rename_vars(self, name_dict=None, groups=None, filter_groups=None, inplace=False):
        """Perform xarray renaming of variable or coordinate names on all groups.

        Loops groups to perform Dataset.rename_vars(name_dict)
        for every key in name_dict if key is a variable or coordinate names of the dataset.
        The renaming is performed on all relevant groups (like
        posterior, prior, sample stats) while non relevant groups like observed data are
        omitted. See :meth:`xarray:xarray.Dataset.rename_vars`

        Parameters
        ----------
        name_dict : dict
            Dictionary whose keys are current variable or coordinate names
            and whose values are the desired names.
        groups : str or list of str, optional
            Groups where the selection is to be applied. Can either be group names
            or metagroup names.
        filter_groups : {None, "like", "regex"}, optional
            If `None` (default), interpret groups as the real group or metagroup names.
            If "like", interpret groups as substrings of the real group or metagroup names.
            If "regex", interpret groups as regular expressions on the real group or
            metagroup names. A la `pandas.filter`.
        inplace : bool, optional
            If ``True``, modify the InferenceData object inplace,
            otherwise, return the modified copy.


        Returns
        -------
        InferenceData
            A new InferenceData object with renamed variables including coordinates by default.
            When `inplace==True` perform renaming in-place and return `None`

        Examples
        --------
        Use ``rename_vars`` to renaming of variable and coordinates on all groups of the
        InferenceData object. We first check the data variables of original object:

        .. jupyter-execute::

            import arviz as az
            idata = az.load_arviz_data("rugby")
            idata

        In order to rename the data variables, we use:

        .. jupyter-execute::

            idata.rename_vars({"home": "home_new"}, inplace=True)
            idata

        See Also
        --------
        xarray.Dataset.rename_vars :
            Returns a new object with renamed variables including coordinates.
        rename :
            Perform xarray renaming of variable and dimensions on all groups of an InferenceData
            object.
        rename_dims : Perform xarray renaming of dimensions on all groups of InferenceData object.
        """
        groups = self._group_names(groups, filter_groups)

        out = self if inplace else deepcopy(self)
        for group in groups:
            dataset = getattr(self, group)
            valid_keys = set(name_dict.keys()).intersection(dataset.data_vars)
            dataset = dataset.rename_vars({key: name_dict[key] for key in valid_keys})
            setattr(out, group, dataset)
        if inplace:
            return None
        else:
            return out

    def rename_dims(self, name_dict=None, groups=None, filter_groups=None, inplace=False):
        """Perform xarray renaming of dimensions on all groups.

        Loops groups to perform Dataset.rename_dims(name_dict)
        for every key in name_dict if key is a dimension of the dataset.
        The renaming is performed on all relevant groups (like
        posterior, prior, sample stats) while non relevant groups like observed data are
        omitted. See :meth:`xarray:xarray.Dataset.rename_dims`

        Parameters
        ----------
        name_dict : dict
            Dictionary whose keys are current dimension names and whose values are the desired
            names.
        groups : str or list of str, optional
            Groups where the selection is to be applied. Can either be group names
            or metagroup names.
        filter_groups : {None, "like", "regex"}, optional
            If `None` (default), interpret groups as the real group or metagroup names.
            If "like", interpret groups as substrings of the real group or metagroup names.
            If "regex", interpret groups as regular expressions on the real group or
            metagroup names. A la `pandas.filter`.
        inplace : bool, optional
            If ``True``, modify the InferenceData object inplace,
            otherwise, return the modified copy.

        Returns
        -------
        InferenceData
            A new InferenceData object with renamed dimension by default.
            When `inplace==True` perform renaming in-place and return `None`

        Examples
        --------
        Use ``rename_dims`` to renaming of dimensions on all groups of the InferenceData
        object. We first check the dimensions of original object:

        .. jupyter-execute::

            import arviz as az
            idata = az.load_arviz_data("rugby")
            idata

        In order to rename the dimensions, we use:

        .. jupyter-execute::

            idata.rename_dims({"team": "team_new"}, inplace=True)
            idata

        See Also
        --------
        xarray.Dataset.rename_dims : Returns a new object with renamed dimensions only.
        rename :
            Perform xarray renaming of variable and dimensions on all groups of an InferenceData
            object.
        rename_vars :
            Perform xarray renaming of variable or coordinate names on all groups of an
            InferenceData object.
        """
        groups = self._group_names(groups, filter_groups)
        if "chain" in name_dict.keys() or "draw" in name_dict.keys():
            raise KeyError("'chain' or 'draw' dimensions can't be renamed")

        out = self if inplace else deepcopy(self)
        for group in groups:
            dataset = getattr(self, group)
            valid_keys = set(name_dict.keys()).intersection(dataset.dims)
            dataset = dataset.rename_dims({key: name_dict[key] for key in valid_keys})
            setattr(out, group, dataset)
        if inplace:
            return None
        else:
            return out

    def add_groups(self, group_dict=None, coords=None, dims=None, **kwargs):
        """Add new groups to InferenceData object.

        Parameters
        ----------
        group_dict : dict of {str : dict or xarray.Dataset}, optional
            Groups to be added
        coords : dict of {str : array_like}, optional
            Coordinates for the dataset
        dims : dict of {str : list of str}, optional
            Dimensions of each variable. The keys are variable names, values are lists of
            coordinates.
        kwargs : dict, optional
            The keyword arguments form of group_dict. One of group_dict or kwargs must be provided.

        Examples
        --------
        Add a ``log_likelihood`` group to the "rugby" example InferenceData after loading.
        It originally doesn't have the ``log_likelihood`` group:

        .. jupyter-execute::

            import arviz as az
            idata = az.load_arviz_data("rugby")
            idata2 = idata.copy()
            post = idata.posterior
            obs = idata.observed_data
            idata

        Knowing the model, we can compute it manually. In this case however,
        we will generate random samples with the right shape.

        .. jupyter-execute::

            import numpy as np
            rng = np.random.default_rng(73)
            ary = rng.normal(size=(post.dims["chain"], post.dims["draw"], obs.dims["match"]))
            idata.add_groups(
                log_likelihood={"home_points": ary},
                dims={"home_points": ["match"]},
            )
            idata

        This is fine if we have raw data, but a bit inconvenient if we start with labeled
        data already. Why provide dims and coords manually again?
        Let's generate a fake log likelihood (doesn't match the model but it serves just
        the same for illustration purposes here) working from the posterior and
        observed_data groups manually:

        .. jupyter-execute::

            import xarray as xr
            from xarray_einstats.stats import XrDiscreteRV
            from scipy.stats import poisson
            dist = XrDiscreteRV(poisson)
            log_lik = xr.Dataset()
            log_lik["home_points"] = dist.logpmf(obs["home_points"], np.exp(post["atts"]))
            idata2.add_groups({"log_likelihood": log_lik})
            idata2

        Note that in the first example we have used the ``kwargs`` argument
        and in the second we have used the ``group_dict`` one.

        See Also
        --------
        extend : Extend InferenceData with groups from another InferenceData.
        concat : Concatenate InferenceData objects.
        """
        group_dict = either_dict_or_kwargs(group_dict, kwargs, "add_groups")
        if not group_dict:
            raise ValueError("One of group_dict or kwargs must be provided.")
        repeated_groups = [group for group in group_dict.keys() if group in self._groups]
        if repeated_groups:
            raise ValueError(f"{repeated_groups} group(s) already exists.")
        for group, dataset in group_dict.items():
            if group not in SUPPORTED_GROUPS_ALL:
                warnings.warn(
                    f"The group {group} is not defined in the InferenceData scheme",
                    UserWarning,
                )
            if dataset is None:
                continue
            elif isinstance(dataset, dict):
                if (
                    group in ("observed_data", "constant_data", "predictions_constant_data")
                    or group not in SUPPORTED_GROUPS_ALL
                ):
                    warnings.warn(
                        "the default dims 'chain' and 'draw' will be added automatically",
                        UserWarning,
                    )
                dataset = dict_to_dataset(dataset, coords=coords, dims=dims)
            elif isinstance(dataset, xr.DataArray):
                if dataset.name is None:
                    dataset.name = "x"
                dataset = dataset.to_dataset()
            elif not isinstance(dataset, xr.Dataset):
                raise ValueError(
                    "Arguments to add_groups() must be xr.Dataset, xr.Dataarray or dicts\
                    (argument '{}' was type '{}')".format(
                        group, type(dataset)
                    )
                )
            if dataset:
                setattr(self, group, dataset)
                if group.startswith(WARMUP_TAG):
                    supported_order = [
                        key for key in SUPPORTED_GROUPS_ALL if key in self._groups_warmup
                    ]
                    if (supported_order == self._groups_warmup) and (group in SUPPORTED_GROUPS_ALL):
                        group_order = [
                            key
                            for key in SUPPORTED_GROUPS_ALL
                            if key in self._groups_warmup + [group]
                        ]
                        group_idx = group_order.index(group)
                        self._groups_warmup.insert(group_idx, group)
                    else:
                        self._groups_warmup.append(group)
                else:
                    supported_order = [key for key in SUPPORTED_GROUPS_ALL if key in self._groups]
                    if (supported_order == self._groups) and (group in SUPPORTED_GROUPS_ALL):
                        group_order = [
                            key for key in SUPPORTED_GROUPS_ALL if key in self._groups + [group]
                        ]
                        group_idx = group_order.index(group)
                        self._groups.insert(group_idx, group)
                    else:
                        self._groups.append(group)

    def extend(self, other, join="left"):
        """Extend InferenceData with groups from another InferenceData.

        Parameters
        ----------
        other : InferenceData
            InferenceData to be added
        join : {'left', 'right'}, default 'left'
            Defines how the two decide which group to keep when the same group is
            present in both objects. 'left' will discard the group in ``other`` whereas 'right'
            will keep the group in ``other`` and discard the one in ``self``.

        Examples
        --------
        Take two InferenceData objects, and extend the first with the groups it doesn't have
        but are present in the 2nd InferenceData object.

        First InferenceData:

        .. jupyter-execute::

            import arviz as az
            idata = az.load_arviz_data("rugby")

        Second InferenceData:

        .. jupyter-execute::

            other_idata = az.load_arviz_data("radon")

        Call the ``extend`` method:

        .. jupyter-execute::

            idata.extend(other_idata)
            idata

        See how now the first InferenceData has more groups, with the data from the
        second one, but the groups it originally had have not been modified,
        even if also present in the second InferenceData.

        See Also
        --------
        add_groups : Add new groups to InferenceData object.
        concat : Concatenate InferenceData objects.

        """
        if not isinstance(other, InferenceData):
            raise ValueError("Extending is possible between two InferenceData objects only.")
        if join not in ("left", "right"):
            raise ValueError(f"join must be either 'left' or 'right', found {join}")
        for group in other._groups_all:  # pylint: disable=protected-access
            if hasattr(self, group) and join == "left":
                continue
            if group not in SUPPORTED_GROUPS_ALL:
                warnings.warn(
                    f"{group} group is not defined in the InferenceData scheme", UserWarning
                )
            dataset = getattr(other, group)
            setattr(self, group, dataset)
            if group.startswith(WARMUP_TAG):
                if group not in self._groups_warmup:
                    supported_order = [
                        key for key in SUPPORTED_GROUPS_ALL if key in self._groups_warmup
                    ]
                    if (supported_order == self._groups_warmup) and (group in SUPPORTED_GROUPS_ALL):
                        group_order = [
                            key
                            for key in SUPPORTED_GROUPS_ALL
                            if key in self._groups_warmup + [group]
                        ]
                        group_idx = group_order.index(group)
                        self._groups_warmup.insert(group_idx, group)
                    else:
                        self._groups_warmup.append(group)
            elif group not in self._groups:
                supported_order = [key for key in SUPPORTED_GROUPS_ALL if key in self._groups]
                if (supported_order == self._groups) and (group in SUPPORTED_GROUPS_ALL):
                    group_order = [
                        key for key in SUPPORTED_GROUPS_ALL if key in self._groups + [group]
                    ]
                    group_idx = group_order.index(group)
                    self._groups.insert(group_idx, group)
                else:
                    self._groups.append(group)

    set_index = _extend_xr_method(xr.Dataset.set_index, see_also="reset_index")
    get_index = _extend_xr_method(xr.Dataset.get_index)
    reset_index = _extend_xr_method(xr.Dataset.reset_index, see_also="set_index")
    set_coords = _extend_xr_method(xr.Dataset.set_coords, see_also="reset_coords")
    reset_coords = _extend_xr_method(xr.Dataset.reset_coords, see_also="set_coords")
    assign = _extend_xr_method(xr.Dataset.assign)
    assign_coords = _extend_xr_method(xr.Dataset.assign_coords)
    sortby = _extend_xr_method(xr.Dataset.sortby)
    chunk = _extend_xr_method(xr.Dataset.chunk)
    unify_chunks = _extend_xr_method(xr.Dataset.unify_chunks)
    load = _extend_xr_method(xr.Dataset.load)
    compute = _extend_xr_method(xr.Dataset.compute)
    persist = _extend_xr_method(xr.Dataset.persist)
    quantile = _extend_xr_method(xr.Dataset.quantile)

    # The following lines use methods on xr.Dataset that are dynamically defined and attached.
    # As a result mypy cannot see them, so we have to suppress the resulting mypy errors.
    mean = _extend_xr_method(xr.Dataset.mean, see_also="median")  # type: ignore[attr-defined]
    median = _extend_xr_method(xr.Dataset.median, see_also="mean")  # type: ignore[attr-defined]
    min = _extend_xr_method(xr.Dataset.min, see_also=["max", "sum"])  # type: ignore[attr-defined]
    max = _extend_xr_method(xr.Dataset.max, see_also=["min", "sum"])  # type: ignore[attr-defined]
    cumsum = _extend_xr_method(xr.Dataset.cumsum, see_also="sum")  # type: ignore[attr-defined]
    sum = _extend_xr_method(xr.Dataset.sum, see_also="cumsum")  # type: ignore[attr-defined]

    def _group_names(
        self,
        groups: Optional[Union[str, List[str]]],
        filter_groups: Optional["Literal['like', 'regex']"] = None,
    ) -> List[str]:
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
        if filter_groups not in {None, "like", "regex"}:
            raise ValueError(
                f"'filter_groups' can only be None, 'like', or 'regex', got: '{filter_groups}'"
            )

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
            raise KeyError(msg) from err
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

        .. jupyter-execute::

            import arviz as az
            import numpy as np
            idata = az.load_arviz_data("non_centered_eight")
            idata_shifted_obs = idata.map(lambda x: x + 3, groups="observed_vars")
            idata_shifted_obs

        Rename and update the coordinate values in both posterior and prior groups.

        .. jupyter-execute::

            idata = az.load_arviz_data("radon")
            idata = idata.map(
                lambda ds: ds.rename({"g_coef": "uranium_coefs"}).assign(
                    uranium_coefs=["intercept", "u_slope"]
                ),
                groups=["posterior", "prior"]
            )
            idata

        Add extra coordinates to all groups containing observed variables

        .. jupyter-execute::

            idata = az.load_arviz_data("rugby")
            home_team, away_team = np.array([
                m.split() for m in idata.observed_data.match.values
            ]).T
            idata = idata.map(
                lambda ds, **kwargs: ds.assign_coords(**kwargs),
                groups="observed_vars",
                home_team=("match", home_team),
                away_team=("match", away_team),
            )
            idata

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

    def copy(self) -> "InferenceData":
        """Return a fresh copy of the ``InferenceData`` object."""
        return deepcopy(self)


@overload
def concat(
    *args,
    dim: Optional[str] = None,
    copy: bool = True,
    inplace: "Literal[True]",
    reset_dim: bool = True,
) -> None:
    ...


@overload
def concat(
    *args,
    dim: Optional[str] = None,
    copy: bool = True,
    inplace: "Literal[False]",
    reset_dim: bool = True,
) -> InferenceData:
    ...


@overload
def concat(
    ids: Iterable[InferenceData],
    dim: Optional[str] = None,
    *,
    copy: bool = True,
    inplace: "Literal[False]",
    reset_dim: bool = True,
) -> InferenceData:
    ...


@overload
def concat(
    ids: Iterable[InferenceData],
    dim: Optional[str] = None,
    *,
    copy: bool = True,
    inplace: "Literal[True]",
    reset_dim: bool = True,
) -> None:
    ...


@overload
def concat(
    ids: Iterable[InferenceData],
    dim: Optional[str] = None,
    *,
    copy: bool = True,
    inplace: bool = False,
    reset_dim: bool = True,
) -> Optional[InferenceData]:
    ...


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

    See Also
    --------
    add_groups : Add new groups to InferenceData object.
    extend : Extend InferenceData with groups from another InferenceData.

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
        msg = f'Invalid `dim`: {dim}. Valid `dim` are {{"group", "chain", "draw"}}'
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
    combined_attr = defaultdict(list)
    for idata in args:
        for key, val in idata.attrs.items():
            combined_attr[key].append(val)

    for key, val in combined_attr.items():
        all_same = True
        for indx in range(len(val) - 1):
            if val[indx] != val[indx + 1]:
                all_same = False
                break
        if all_same:
            combined_attr[key] = val[0]
    if inplace:
        setattr(args[0], "_attrs", dict(combined_attr))

    if not inplace:
        # Keep order for python 3.5
        inference_data_dict = OrderedDict()

    if dim is None:
        arg0 = args[0]
        arg0_groups = ccopy(arg0._groups_all)
        args_groups = {}
        # check if groups are independent
        # Concat over unique groups
        for arg in args[1:]:
            for group in arg._groups_all:
                if group in args_groups or group in arg0_groups:
                    msg = (
                        "Concatenating overlapping groups is not supported unless `dim` is defined."
                        " Valid dimensions are `chain` and `draw`. Alternatively, use extend to"
                        " combine InferenceData with overlapping groups"
                    )
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
                # handle data groups separately
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
                        var_dims = group_data[var].dims
                        var0_dims = group0_data[var].dims
                        if var_dims != var0_dims:
                            msg = "Mismatch between the dimensions."
                            raise TypeError(msg)

                        if dim not in var_dims or dim not in var0_dims:
                            msg = f"Dimension {dim} missing."
                            raise TypeError(msg)

                    # xr.concat
                    concatenated_group = xr.concat((group0_data, group_data), dim=dim)
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
                        group_attrs = {}

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
                            combined_key = f"combined_{attr_key}"
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
                            var_data = group_data[var]
                            getattr(arg0, group)[var] = var_data
                        else:
                            var_data = group_data[var]
                            var0_data = group0_data[var]
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
                        group_attrs = {}

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
                            combined_key = f"combined_{attr_key}"
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

    if not inplace:
        inference_data_dict["attrs"] = combined_attr

    return None if inplace else InferenceData(**inference_data_dict)
