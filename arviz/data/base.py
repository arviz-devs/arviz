"""Low level converters usually used by other functions."""
import datetime
import functools
import warnings
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
import pkg_resources
import xarray as xr

try:
    import ujson as json
except ImportError:
    # mypy struggles with conditional imports expressed as catching ImportError:
    # https://github.com/python/mypy/issues/1153
    import json  # type: ignore

from .. import __version__, utils

CoordSpec = Dict[str, List[Any]]
DimSpec = Dict[str, List[str]]


class requires:  # pylint: disable=invalid-name
    """Decorator to return None if an object does not have the required attribute.

    If the decorator is called various times on the same function with different
    attributes, it will return None if one of them is missing. If instead a list
    of attributes is passed, it will return None if all attributes in the list are
    missing. Both functionalities can be combined as desired.
    """

    def __init__(self, *props):
        self.props = props

    def __call__(self, func):  # noqa: D202
        """Wrap the decorated function."""

        def wrapped(cls, *args, **kwargs):
            """Return None if not all props are available."""
            for prop in self.props:
                prop = [prop] if isinstance(prop, str) else prop
                if all([getattr(cls, prop_i) is None for prop_i in prop]):
                    return None
            return func(cls, *args, **kwargs)

        return wrapped


def generate_dims_coords(
    shape, var_name, dims=None, coords=None, default_dims=None, skip_event_dims=None
):
    """Generate default dimensions and coordinates for a variable.

    Parameters
    ----------
    shape : tuple[int]
        Shape of the variable
    var_name : str
        Name of the variable. If no dimension name(s) is provided, ArviZ
        will generate a default dimension name using ``var_name``, e.g.,
        ``"foo_dim_0"`` for the first dimension if ``var_name`` is ``"foo"``.
    dims : list
        List of dimensions for the variable
    coords : dict[str] -> list[str]
        Map of dimensions to coordinates
    default_dims : list[str]
        Dimension names that are not part of the variable's shape. For example,
        when manipulating Monte Carlo traces, the ``default_dims`` would be
        ``["chain" , "draw"]`` which ArviZ uses as its own names for dimensions
        of MCMC traces.
    skip_event_dims : bool, default False

    Returns
    -------
    list[str]
        Default dims
    dict[str] -> list[str]
        Default coords
    """
    if default_dims is None:
        default_dims = []
    if dims is None:
        dims = []
    if skip_event_dims is None:
        skip_event_dims = False

    if coords is None:
        coords = {}

    coords = deepcopy(coords)
    dims = deepcopy(dims)

    ndims = len([dim for dim in dims if dim not in default_dims])
    if ndims > len(shape):
        if skip_event_dims:
            dims = dims[: len(shape)]
        else:
            warnings.warn(
                (
                    "In variable {var_name}, there are "
                    + "more dims ({dims_len}) given than exist ({shape_len}). "
                    + "Passed array should have shape ({defaults}*shape)"
                ).format(
                    var_name=var_name,
                    dims_len=len(dims),
                    shape_len=len(shape),
                    defaults=",".join(default_dims) + ", " if default_dims is not None else "",
                ),
                UserWarning,
            )
    if skip_event_dims:
        # this is needed in case the reduction keeps the dimension with size 1
        for i, (dim, dim_size) in enumerate(zip(dims, shape)):
            print(f"{i}, dim: {dim}, {dim_size} =? {len(coords.get(dim, []))}")
            if (dim in coords) and (dim_size != len(coords[dim])):
                dims = dims[:i]
                break

    for idx, dim_len in enumerate(shape):
        if (len(dims) < idx + 1) or (dims[idx] is None):
            dim_name = "{var_name}_dim_{idx}".format(var_name=var_name, idx=idx)
            if len(dims) < idx + 1:
                dims.append(dim_name)
            else:
                dims[idx] = dim_name
        dim_name = dims[idx]
        if dim_name not in coords:
            coords[dim_name] = utils.arange(dim_len)
    coords = {key: coord for key, coord in coords.items() if any(key == dim for dim in dims)}
    return dims, coords


def numpy_to_data_array(ary, *, var_name="data", coords=None, dims=None, skip_event_dims=None):
    """Convert a numpy array to an xarray.DataArray.

    The first two dimensions will be (chain, draw), and any remaining
    dimensions will be "shape".
    If the numpy array is 1d, this dimension is interpreted as draw
    If the numpy array is 2d, it is interpreted as (chain, draw)
    If the numpy array is 3 or more dimensions, the last dimensions are kept as shapes.

    Parameters
    ----------
    ary : np.ndarray
        A numpy array. If it has 2 or more dimensions, the first dimension should be
        independent chains from a simulation. Use `np.expand_dims(ary, 0)` to add a
        single dimension to the front if there is only 1 chain.
    var_name : str
        If there are no dims passed, this string is used to name dimensions
    coords : dict[str, iterable]
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : List(str)
        A list of coordinate names for the variable
    skip_event_dims : bool

    Returns
    -------
    xr.DataArray
        Will have the same data as passed, but with coordinates and dimensions
    """
    # manage and transform copies
    default_dims = ["chain", "draw"]
    ary = utils.two_de(ary)
    n_chains, n_samples, *shape = ary.shape
    if n_chains > n_samples:
        warnings.warn(
            "More chains ({n_chains}) than draws ({n_samples}). "
            "Passed array should have shape (chains, draws, *shape)".format(
                n_chains=n_chains, n_samples=n_samples
            ),
            UserWarning,
        )

    dims, coords = generate_dims_coords(
        shape,
        var_name,
        dims=dims,
        coords=coords,
        default_dims=default_dims,
        skip_event_dims=skip_event_dims,
    )

    # reversed order for default dims: 'chain', 'draw'
    if "draw" not in dims:
        dims = ["draw"] + dims
    if "chain" not in dims:
        dims = ["chain"] + dims

    if "chain" not in coords:
        coords["chain"] = utils.arange(n_chains)
    if "draw" not in coords:
        coords["draw"] = utils.arange(n_samples)

    # filter coords based on the dims
    coords = {key: xr.IndexVariable((key,), data=coords[key]) for key in dims}
    return xr.DataArray(ary, coords=coords, dims=dims)


def dict_to_dataset(
    data, *, attrs=None, library=None, coords=None, dims=None, skip_event_dims=None
):
    """Convert a dictionary of numpy arrays to an xarray.Dataset.

    Parameters
    ----------
    data : dict[str] -> ndarray
        Data to convert. Keys are variable names.
    attrs : dict
        Json serializable metadata to attach to the dataset, in addition to defaults.
    library : module
        Library used for performing inference. Will be attached to the attrs metadata.
    coords : dict[str] -> ndarray
        Coordinates for the dataset
    dims : dict[str] -> list[str]
        Dimensions of each variable. The keys are variable names, values are lists of
        coordinates.
    skip_event_dims : bool
        If True, cut extra dims whenever present to match the shape of the data.
        Necessary for PPLs which have the same name in both observed data and log
        likelihood groups, to account for their different shapes when observations are
        multivariate.

    Returns
    -------
    xr.Dataset

    Examples
    --------
    dict_to_dataset({'x': np.random.randn(4, 100), 'y': np.random.rand(4, 100)})

    """
    if dims is None:
        dims = {}

    data_vars = {}
    for key, values in data.items():
        data_vars[key] = numpy_to_data_array(
            values, var_name=key, coords=coords, dims=dims.get(key), skip_event_dims=skip_event_dims
        )
    return xr.Dataset(data_vars=data_vars, attrs=make_attrs(attrs=attrs, library=library))


def make_attrs(attrs=None, library=None):
    """Make standard attributes to attach to xarray datasets.

    Parameters
    ----------
    attrs : dict (optional)
        Additional attributes to add or overwrite

    Returns
    -------
    dict
        attrs
    """
    default_attrs = {
        "created_at": datetime.datetime.utcnow().isoformat(),
        "arviz_version": __version__,
    }
    if library is not None:
        library_name = library.__name__
        default_attrs["inference_library"] = library_name
        try:
            version = pkg_resources.get_distribution(library_name).version
            default_attrs["inference_library_version"] = version
        except pkg_resources.DistributionNotFound:
            if hasattr(library, "__version__"):
                version = library.__version__
                default_attrs["inference_library_version"] = version

    if attrs is not None:
        default_attrs.update(attrs)
    return default_attrs


def _extend_xr_method(func, doc="", description="", examples="", see_also=""):
    """Make wrapper to extend methods from xr.Dataset to InferenceData Class.

    Parameters
    ----------
    func : callable
        An xr.Dataset function
    doc : str
        docstring for the func
    description : str
        the description of the func to be added in docstring
    examples : str
        the examples of the func to be added in docstring
    see_also : str, list
        the similar methods of func to be included in See Also section of docstring

    """
    # pydocstyle requires a non empty line

    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        _filter = kwargs.pop("filter_groups", None)
        _groups = kwargs.pop("groups", None)
        _inplace = kwargs.pop("inplace", False)

        out = self if _inplace else deepcopy(self)

        groups = self._group_names(_groups, _filter)  # pylint: disable=protected-access
        for group in groups:
            xr_data = getattr(out, group)
            xr_data = func(xr_data, *args, **kwargs)  # pylint: disable=not-callable
            setattr(out, group, xr_data)

        return None if _inplace else out

    description_default = """{method_name} method is extended from xarray.Dataset methods.
    
    {description}For more info see :meth:`xarray:xarray.Dataset.{method_name}`
    """.format(
        description=description, method_name=func.__name__  # pylint: disable=no-member
    )
    params = """
    Parameters
    ----------
    groups: str or list of str, optional
        Groups where the selection is to be applied. Can either be group names
        or metagroup names.
    filter_groups: {None, "like", "regex"}, optional, default=None
        If `None` (default), interpret groups as the real group or metagroup names.
        If "like", interpret groups as substrings of the real group or metagroup names.
        If "regex", interpret groups as regular expressions on the real group or
        metagroup names. A la `pandas.filter`.
    inplace: bool, optional
        If ``True``, modify the InferenceData object inplace,
        otherwise, return the modified copy.
    """

    if not isinstance(see_also, str):
        see_also = "\n".join(see_also)
    see_also_basic = """
    See Also
    --------
    xarray.Dataset.{method_name}
    {custom_see_also}
    """.format(
        method_name=func.__name__, custom_see_also=see_also  # pylint: disable=no-member
    )
    wrapped.__doc__ = (
        description_default + params + examples + see_also_basic if doc is None else doc
    )

    return wrapped


def _make_json_serializable(data: dict) -> dict:
    """Convert `data` with numpy.ndarray-like values to JSON-serializable form."""
    ret = dict()
    for key, value in data.items():
        try:
            json.dumps(value)
        except (TypeError, OverflowError):
            pass
        else:
            ret[key] = value
            continue
        if isinstance(value, dict):
            ret[key] = _make_json_serializable(value)
        elif isinstance(value, np.ndarray):
            ret[key] = np.asarray(value).tolist()
        else:
            raise TypeError(
                f"Value associated with variable `{type(value)}` is not JSON serializable."
            )
    return ret
