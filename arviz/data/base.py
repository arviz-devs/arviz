"""Low level converters usually used by other functions."""

import datetime
import functools
import importlib
import re
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import xarray as xr

try:
    import tree
except ImportError:
    tree = None

try:
    import ujson as json
except ImportError:
    # mypy struggles with conditional imports expressed as catching ImportError:
    # https://github.com/python/mypy/issues/1153
    import json  # type: ignore

from .. import __version__, utils
from ..rcparams import rcParams

CoordSpec = Dict[str, List[Any]]
DimSpec = Dict[str, List[str]]
RequiresArgTypeT = TypeVar("RequiresArgTypeT")
RequiresReturnTypeT = TypeVar("RequiresReturnTypeT")


class requires:  # pylint: disable=invalid-name
    """Decorator to return None if an object does not have the required attribute.

    If the decorator is called various times on the same function with different
    attributes, it will return None if one of them is missing. If instead a list
    of attributes is passed, it will return None if all attributes in the list are
    missing. Both functionalities can be combined as desired.
    """

    def __init__(self, *props: Union[str, List[str]]) -> None:
        self.props: Tuple[Union[str, List[str]], ...] = props

    # Until typing.ParamSpec (https://www.python.org/dev/peps/pep-0612/) is available
    # in all our supported Python versions, there is no way to simultaneously express
    # the following two properties:
    # - the input function may take arbitrary args/kwargs, and
    # - the output function takes those same arbitrary args/kwargs, but has a different return type.
    # We either have to limit the input function to e.g. only allowing a "self" argument,
    # or we have to adopt the current approach of annotating the returned function as if
    # it was defined as "def f(*args: Any, **kwargs: Any) -> Optional[RequiresReturnTypeT]".
    #
    # Since all functions decorated with @requires currently only accept a single argument,
    # we choose to limit application of @requires to only functions of one argument.
    # When typing.ParamSpec is available, this definition can be updated to use it.
    # See https://github.com/arviz-devs/arviz/pull/1504 for more discussion.
    def __call__(
        self, func: Callable[[RequiresArgTypeT], RequiresReturnTypeT]
    ) -> Callable[[RequiresArgTypeT], Optional[RequiresReturnTypeT]]:  # noqa: D202
        """Wrap the decorated function."""

        def wrapped(cls: RequiresArgTypeT) -> Optional[RequiresReturnTypeT]:
            """Return None if not all props are available."""
            for prop in self.props:
                prop = [prop] if isinstance(prop, str) else prop
                if all((getattr(cls, prop_i) is None for prop_i in prop)):
                    return None
            return func(cls)

        return wrapped


def _yield_flat_up_to(shallow_tree, input_tree, path=()):
    """Yields (path, value) pairs of input_tree flattened up to shallow_tree.

    Adapted from dm-tree (https://github.com/google-deepmind/tree) to allow
    lists as leaves.

    Args:
        shallow_tree: Nested structure. Traverse no further than its leaf nodes.
        input_tree: Nested structure. Return the paths and values from this tree.
            Must have the same upper structure as shallow_tree.
        path: Tuple. Optional argument, only used when recursing. The path from the
            root of the original shallow_tree, down to the root of the shallow_tree
            arg of this recursive call.

    Yields:
        Pairs of (path, value), where path the tuple path of a leaf node in
        shallow_tree, and value is the value of the corresponding node in
        input_tree.
    """
    # pylint: disable=protected-access
    if tree is None:
        raise ImportError("Missing optional dependency 'dm-tree'. Use pip or conda to install it")

    if isinstance(shallow_tree, tree._TEXT_OR_BYTES) or not (
        isinstance(shallow_tree, tree.collections_abc.Mapping)
        or tree._is_namedtuple(shallow_tree)
        or tree._is_attrs(shallow_tree)
    ):
        yield (path, input_tree)
    else:
        input_tree = dict(tree._yield_sorted_items(input_tree))
        for shallow_key, shallow_subtree in tree._yield_sorted_items(shallow_tree):
            subpath = path + (shallow_key,)
            input_subtree = input_tree[shallow_key]
            for leaf_path, leaf_value in _yield_flat_up_to(
                shallow_subtree, input_subtree, path=subpath
            ):
                yield (leaf_path, leaf_value)
    # pylint: enable=protected-access


def _flatten_with_path(structure):
    return list(_yield_flat_up_to(structure, structure))


def generate_dims_coords(
    shape,
    var_name,
    dims=None,
    coords=None,
    default_dims=None,
    index_origin=None,
    skip_event_dims=None,
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
    index_origin : int, optional
        Starting value of integer coordinate values. Defaults to the value in rcParam
        ``data.index_origin``.
    skip_event_dims : bool, default False

    Returns
    -------
    list[str]
        Default dims
    dict[str] -> list[str]
        Default coords
    """
    if index_origin is None:
        index_origin = rcParams["data.index_origin"]
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
            if (dim in coords) and (dim_size != len(coords[dim])):
                dims = dims[:i]
                break

    for i, dim_len in enumerate(shape):
        idx = i + len([dim for dim in default_dims if dim in dims])
        if len(dims) < idx + 1:
            dim_name = f"{var_name}_dim_{i}"
            dims.append(dim_name)
        elif dims[idx] is None:
            dim_name = f"{var_name}_dim_{i}"
            dims[idx] = dim_name
        dim_name = dims[idx]
        if dim_name not in coords:
            coords[dim_name] = np.arange(index_origin, dim_len + index_origin)
    coords = {
        key: coord
        for key, coord in coords.items()
        if any(key == dim for dim in dims + default_dims)
    }
    return dims, coords


def numpy_to_data_array(
    ary,
    *,
    var_name="data",
    coords=None,
    dims=None,
    default_dims=None,
    index_origin=None,
    skip_event_dims=None,
):
    """Convert a numpy array to an xarray.DataArray.

    By default, the first two dimensions will be (chain, draw), and any remaining
    dimensions will be "shape".
    * If the numpy array is 1d, this dimension is interpreted as draw
    * If the numpy array is 2d, it is interpreted as (chain, draw)
    * If the numpy array is 3 or more dimensions, the last dimensions are kept as shapes.

    To modify this behaviour, use ``default_dims``.

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
    default_dims : list of str, optional
        Passed to :py:func:`generate_dims_coords`. Defaults to ``["chain", "draw"]``, and
        an empty list is accepted
    index_origin : int, optional
        Passed to :py:func:`generate_dims_coords`
    skip_event_dims : bool

    Returns
    -------
    xr.DataArray
        Will have the same data as passed, but with coordinates and dimensions
    """
    # manage and transform copies
    if default_dims is None:
        default_dims = ["chain", "draw"]
    if "chain" in default_dims and "draw" in default_dims:
        ary = utils.two_de(ary)
        n_chains, n_samples, *_ = ary.shape
        if n_chains > n_samples:
            warnings.warn(
                "More chains ({n_chains}) than draws ({n_samples}). "
                "Passed array should have shape (chains, draws, *shape)".format(
                    n_chains=n_chains, n_samples=n_samples
                ),
                UserWarning,
            )
    else:
        ary = utils.one_de(ary)

    dims, coords = generate_dims_coords(
        ary.shape[len(default_dims) :],
        var_name,
        dims=dims,
        coords=coords,
        default_dims=default_dims,
        index_origin=index_origin,
        skip_event_dims=skip_event_dims,
    )

    # reversed order for default dims: 'chain', 'draw'
    if "draw" not in dims and "draw" in default_dims:
        dims = ["draw"] + dims
    if "chain" not in dims and "chain" in default_dims:
        dims = ["chain"] + dims

    index_origin = rcParams["data.index_origin"]
    if "chain" not in coords and "chain" in default_dims:
        coords["chain"] = np.arange(index_origin, n_chains + index_origin)
    if "draw" not in coords and "draw" in default_dims:
        coords["draw"] = np.arange(index_origin, n_samples + index_origin)

    # filter coords based on the dims
    coords = {key: xr.IndexVariable((key,), data=np.asarray(coords[key])) for key in dims}
    return xr.DataArray(ary, coords=coords, dims=dims)


def dict_to_dataset(
    data,
    *,
    attrs=None,
    library=None,
    coords=None,
    dims=None,
    default_dims=None,
    index_origin=None,
    skip_event_dims=None,
):
    """Convert a dictionary or pytree of numpy arrays to an xarray.Dataset.

    ArviZ itself supports conversion of flat dictionaries.
    Suport for pytrees requires ``dm-tree`` which is an optional dependency.
    See https://jax.readthedocs.io/en/latest/pytrees.html for what a pytree is, but
    this inclues at least dictionaries and tuple types.

    Parameters
    ----------
    data : dict of {str : array_like or dict} or pytree
        Data to convert. Keys are variable names.
    attrs : dict, optional
        Json serializable metadata to attach to the dataset, in addition to defaults.
    library : module, optional
        Library used for performing inference. Will be attached to the attrs metadata.
    coords : dict of {str : ndarray}, optional
        Coordinates for the dataset
    dims : dict of {str : list of str}, optional
        Dimensions of each variable. The keys are variable names, values are lists of
        coordinates.
    default_dims : list of str, optional
        Passed to :py:func:`numpy_to_data_array`
    index_origin : int, optional
        Passed to :py:func:`numpy_to_data_array`
    skip_event_dims : bool, optional
        If True, cut extra dims whenever present to match the shape of the data.
        Necessary for PPLs which have the same name in both observed data and log
        likelihood groups, to account for their different shapes when observations are
        multivariate.

    Returns
    -------
    xarray.Dataset
        In case of nested pytrees, the variable name will be a tuple of individual names.

    Notes
    -----
    This function is available through two aliases: ``dict_to_dataset`` or ``pytree_to_dataset``.

    Examples
    --------
    Convert a dictionary with two 2D variables to a Dataset.

    .. ipython::

        In [1]: import arviz as az
           ...: import numpy as np
           ...: az.dict_to_dataset({'x': np.random.randn(4, 100), 'y': np.random.rand(4, 100)})

    Note that unlike the :class:`xarray.Dataset` constructor, ArviZ has added extra
    information to the generated Dataset such as default dimension names for sampled
    dimensions and some attributes.

    The function is also general enough to work on pytrees such as nested dictionaries:

    .. ipython::

        In [1]: az.pytree_to_dataset({'top': {'second': 1.}, 'top2': 1.})

    which has two variables (as many as leafs) named ``('top', 'second')`` and ``top2``.

    Dimensions and co-ordinates can be defined as usual:

    .. ipython::

        In [1]: datadict = {
           ...:     "top": {"a": np.random.randn(100), "b": np.random.randn(1, 100, 10)},
           ...:     "d": np.random.randn(100),
           ...: }
           ...: az.dict_to_dataset(
           ...:     datadict,
           ...:     coords={"c": np.arange(10)},
           ...:     dims={("top", "b"): ["c"]}
           ...: )

    """
    if dims is None:
        dims = {}

    if tree is not None:
        try:
            data = {k[0] if len(k) == 1 else k: v for k, v in _flatten_with_path(data)}
        except TypeError:  # probably unsortable keys -- the function will still work if
            pass  # it is an honest dictionary.

    data_vars = {
        key: numpy_to_data_array(
            values,
            var_name=key,
            coords=coords,
            dims=dims.get(key),
            default_dims=default_dims,
            index_origin=index_origin,
            skip_event_dims=skip_event_dims,
        )
        for key, values in data.items()
    }
    return xr.Dataset(data_vars=data_vars, attrs=make_attrs(attrs=attrs, library=library))


pytree_to_dataset = dict_to_dataset


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
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "arviz_version": __version__,
    }
    if library is not None:
        library_name = library.__name__
        default_attrs["inference_library"] = library_name
        try:
            version = importlib.metadata.version(library_name)
            default_attrs["inference_library_version"] = version
        except importlib.metadata.PackageNotFoundError:
            if hasattr(library, "__version__"):
                version = library.__version__
                default_attrs["inference_library_version"] = version
    if attrs is not None:
        default_attrs.update(attrs)
    return default_attrs


def _extend_xr_method(func, doc=None, description="", examples="", see_also=""):
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

    {description}

    For more info see :meth:`xarray:xarray.Dataset.{method_name}`.
    In addition to the arguments available in the original method, the following
    ones are added by ArviZ to adapt the method to being called on an ``InferenceData`` object.
    """.format(
        description=description, method_name=func.__name__  # pylint: disable=no-member
    )
    params = """
    Other Parameters
    ----------------
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
    ret = {}
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


def infer_stan_dtypes(stan_code):
    """Infer Stan integer variables from generated quantities block."""
    # Remove old deprecated comments
    stan_code = "\n".join(
        line if "#" not in line else line[: line.find("#")] for line in stan_code.splitlines()
    )
    pattern_remove_comments = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"', re.DOTALL | re.MULTILINE
    )
    stan_code = re.sub(pattern_remove_comments, "", stan_code)

    # Check generated quantities
    if "generated quantities" not in stan_code:
        return {}

    # Extract generated quantities block
    gen_quantities_location = stan_code.index("generated quantities")
    block_start = gen_quantities_location + stan_code[gen_quantities_location:].index("{")

    curly_bracket_count = 0
    block_end = None
    for block_end, char in enumerate(stan_code[block_start:], block_start + 1):
        if char == "{":
            curly_bracket_count += 1
        elif char == "}":
            curly_bracket_count -= 1

            if curly_bracket_count == 0:
                break

    stan_code = stan_code[block_start:block_end]

    stan_integer = r"int"
    stan_limits = r"(?:\<[^\>]+\>)*"  # ignore group: 0 or more <....>
    stan_param = r"([^;=\s\[]+)"  # capture group: ends= ";", "=", "[" or whitespace
    stan_ws = r"\s*"  # 0 or more whitespace
    stan_ws_one = r"\s+"  # 1 or more whitespace
    pattern_int = re.compile(
        "".join((stan_integer, stan_ws_one, stan_limits, stan_ws, stan_param)), re.IGNORECASE
    )
    dtypes = {key.strip(): "int" for key in re.findall(pattern_int, stan_code)}
    return dtypes
