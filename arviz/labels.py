# pylint: disable=unused-argument
"""Utilities to generate labels from xarray objects."""
from typing import Union

__all__ = [
    "mix_labellers",
    "BaseLabeller",
    "DimCoordLabeller",
    "DimIdxLabeller",
    "MapLabeller",
    "NoVarLabeller",
    "NoModelLabeller",
]


def mix_labellers(labellers, class_name="MixtureLabeller"):
    """Combine Labeller classes dynamically.

    The Labeller class aims to split plot labeling in ArviZ into atomic tasks to maximize
    extensibility, and the few classes provided are designed with small deviations
    from the base class, in many cases only one method is modified by the child class.
    It is to be expected then to want to use multiple classes "at once".

    This functions helps combine classes dynamically.

    Parameters
    ----------
    labellers : iterable of types
        Iterable of Labeller types to combine
    class_name : str, optional
        The name of the generated class

    Returns
    -------
        type
            Mixture class object. *It is not initialized*

    Examples
    --------
    Combine the :class:`~arviz.labels.DimCoordLabeller` with the
    :class:`~arviz.labels.MapLabeller` to generate labels in the style of the
    ``DimCoordLabeller`` but using the mappings defined by ``MapLabeller``.
    Note that this works even though both modify the same methods because
    ``MapLabeller`` implements the mapping and then calls `super().method`.

    .. ipython::

        In [1]: from arviz.labels import mix_labellers, DimCoordLabeller, MapLabeller
           ...: l1 = DimCoordLabeller()
           ...: sel = {"dim1": "a", "dim2": "top"}
           ...: print(f"Output of DimCoordLabeller alone > {l1.sel_to_str(sel, sel)}")
           ...: l2 = MapLabeller(dim_map={"dim1": "$d_1$", "dim2": r"$d_2$"})
           ...: print(f"Output of MapLabeller alone > {l2.sel_to_str(sel, sel)}")
           ...: l3 = mix_labellers(
           ...:     (MapLabeller, DimCoordLabeller)
           ...: )(dim_map={"dim1": "$d_1$", "dim2": r"$d_2$"})
           ...: print(f"Output of mixture labeller > {l3.sel_to_str(sel, sel)}")

    We can see how the mappings are taken into account as well as the dim+coord style. However,
    he order in the ``labellers`` arg iterator is important! See for yourself:

    .. ipython:: python

        l4 = mix_labellers(
            (DimCoordLabeller, MapLabeller)
        )(dim_map={"dim1": "$d_1$", "dim2": r"$d_2$"})
        print(f"Output of inverted mixture labeller > {l4.sel_to_str(sel, sel)}")

    """
    return type(class_name, labellers, {})


class BaseLabeller:
    """WIP."""

    def dim_coord_to_str(self, dim, coord_val, coord_idx):
        """WIP."""
        return f"{coord_val}"

    def sel_to_str(self, sel: dict, isel: dict):
        """WIP."""
        if not sel:
            return ""
        return ", ".join(
            [
                self.dim_coord_to_str(dim, v, i)
                for (dim, v), (_, i) in zip(sel.items(), isel.items())
            ]
        )

    def var_name_to_str(self, var_name: Union[str, None]):
        """WIP."""
        return var_name

    def var_pp_to_str(self, var_name, pp_var_name):
        """WIP."""
        var_name_str = self.var_name_to_str(var_name)
        pp_var_name_str = self.var_name_to_str(pp_var_name)
        return f"{var_name_str} / {pp_var_name_str}"

    def model_name_to_str(self, model_name):
        """WIP."""
        return model_name

    def make_label_vert(self, var_name: Union[str, None], sel: dict, isel: dict):
        """WIP."""
        var_name_str = self.var_name_to_str(var_name)
        sel_str = self.sel_to_str(sel, isel)
        if not sel_str:
            return var_name_str
        if var_name_str is None:
            return sel_str
        return f"{var_name_str}\n{sel_str}"

    def make_label_flat(self, var_name: str, sel: dict, isel: dict):
        """WIP."""
        var_name_str = self.var_name_to_str(var_name)
        sel_str = self.sel_to_str(sel, isel)
        if not sel_str:
            return var_name_str
        if var_name_str is None:
            return sel_str
        return f"{var_name_str}[{sel_str}]"

    def make_pp_label(self, var_name, pp_var_name, sel, isel):
        """WIP."""
        names = self.var_pp_to_str(var_name, pp_var_name)
        return self.make_label_vert(names, sel, isel)

    def make_model_label(self, model_name, label):
        """WIP."""
        model_name_str = self.model_name_to_str(model_name)
        if model_name_str is None:
            return label
        return f"{model_name}: {label}"


class DimCoordLabeller(BaseLabeller):
    """WIP."""

    def dim_coord_to_str(self, dim, coord_val, coord_idx):
        """WIP."""
        return f"{dim}: {coord_val}"


class IdxLabeller(BaseLabeller):
    """WIP."""

    def dim_coord_to_str(self, dim, coord_val, coord_idx):
        """WIP."""
        return f"{coord_idx}"


class DimIdxLabeller(BaseLabeller):
    """WIP."""

    def dim_coord_to_str(self, dim, coord_val, coord_idx):
        """WIP."""
        return f"{dim}#{coord_idx}"


class MapLabeller(BaseLabeller):
    """WIP."""

    def __init__(self, var_name_map=None, dim_map=None, coord_map=None, model_name_map=None):
        """WIP."""
        self.var_name_map = {} if var_name_map is None else var_name_map
        self.dim_map = {} if dim_map is None else dim_map
        self.coord_map = {} if coord_map is None else coord_map
        self.model_name_map = {} if model_name_map is None else model_name_map

    def dim_coord_to_str(self, dim, coord_val, coord_idx):
        """WIP."""
        dim_str = self.dim_map.get(dim, dim)
        coord_str = self.coord_map.get(dim, {}).get(coord_val, coord_val)
        return super().dim_coord_to_str(dim_str, coord_str, coord_idx)

    def var_name_to_str(self, var_name):
        """WIP."""
        var_name_str = self.var_name_map.get(var_name, var_name)
        return super().var_name_to_str(var_name_str)

    def model_name_to_str(self, model_name):
        """WIP."""
        model_name_str = self.var_name_map.get(model_name, model_name)
        return super().model_name_to_str(model_name_str)


class NoVarLabeller(BaseLabeller):
    """WIP."""

    def var_name_to_str(self, var_name):
        """WIP."""
        return None


class NoModelLabeller(BaseLabeller):
    """WIP."""

    def make_model_label(self, model_name, label):
        """WIP."""
        return label
