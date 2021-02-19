# pylint: disable=unused-argument
"""Utilities to generate labels from xarray objects."""
from typing import Union

__all__ = [
    "BaseLabeller",
    "DimCoordLabeller",
    "DimIdxLabeller",
    "MapLabeller",
    "NoRepeatLabeller",
    "NoModelLabeller",
]


class BaseLabeller:
    def dim_coord_to_str(self, dim, coord_val, coord_idx):
        return f"{coord_val}"

    def sel_to_str(self, sel: dict, isel: dict):
        if not sel:
            return ""
        return ", ".join(
            [
                self.dim_coord_to_str(dim, v, i)
                for (dim, v), (_, i) in zip(sel.items(), isel.items())
            ]
        )

    def var_name_to_str(self, var_name: Union[str, None]):
        return var_name

    def make_label_vert(self, var_name: Union[str, None], sel: dict, isel: dict):
        var_name_str = self.var_name_to_str(var_name)
        sel_str = self.sel_to_str(sel, isel)
        if not sel_str:
            return var_name_str
        if var_name_str is None:
            return sel_str
        return f"{var_name_str}\n{sel_str}"

    def make_label_flat(self, var_name: str, sel: dict, isel: dict):
        var_name_str = self.var_name_to_str(var_name)
        sel_str = self.sel_to_str(sel, isel)
        if not sel_str:
            return var_name_str
        if var_name is None:
            return sel_str
        return f"{var_name_str}[{sel_str}]"

    def make_model_label(self, model_name, label):
        if model_name is None:
            return label
        return f"{model_name}: {label}"


class DimCoordLabeller(BaseLabeller):
    def dim_coord_to_str(self, dim, coord_val, coord_idx):
        return f"{dim}: {coord_val}"


class DimIdxLabeller(BaseLabeller):
    def dim_coord_to_str(self, dim, coord_val, coord_idx):
        return f"{dim}#{coord_idx}"


class MapLabeller(BaseLabeller):
    def __init__(self, var_name_map=None, dim_map=None, coord_map=None):
        self.var_name_map = {} if var_name_map is None else var_name_map
        self.dim_map = {} if dim_map is None else dim_map
        self.coord_map = {} if coord_map is None else coord_map

    def dim_coord_to_str(self, dim, coord_val, coord_idx):
        dim_str = self.dim_map.get(dim, dim)
        coord_str = self.coord_map.get(coord_val, coord_val)
        return super().dim_coord_to_str(dim_str, coord_str, coord_idx)

    def var_name_to_str(self, var_name):
        return self.var_name_map.get(var_name, var_name)


class NoRepeatLabeller(BaseLabeller):
    def __init__(self):
        self.current_var = None

    def var_name_to_str(self, var_name):
        current_var = getattr(self, "current_var", None)
        if var_name == current_var:
            return ""
        self.current_var = var_name
        return var_name


class NoModelLabeller(BaseLabeller):
    def make_model_label(self, model_name, label):
        return label
