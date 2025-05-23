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

    Allows dynamic creation of new labeller classes by combining multiple
    subclasses of BaseLabeller. Used to customize labeling behavior by stacking
    simple modifications.

    Parameters
    ----------
    labellers : iterable of types
        Iterable of Labeller types to combine.
    class_name : str, optional
        The name of the generated class.

    Returns
    -------
    type
        Dynamically created class combining provided labeller classes.

    Notes
    -----
    The returned class is *not* initialized.
    """
    return type(class_name, labellers, {})


class BaseLabeller:
    """
    Base class for generating plot labels from xarray-like selections.

    Provides methods for constructing human-readable labels for variables,
    dimensions, coordinates, and model components when plotting data
    stored in InferenceData objects.
    """

    def dim_coord_to_str(self, dim, coord_val, coord_idx):
        """
        Generate a label string for a single dimension/coordinate pair.

        Parameters
        ----------
        dim : str
            The name of the dimension.
        coord_val : any
            The value of the coordinate along that dimension.
        coord_idx : int
            The index of the coordinate value in the dimension.

        Returns
        -------
        str
            A string representation of the coordinate value.
        """
        return f"{coord_val}"

    def sel_to_str(self, sel: dict, isel: dict):
        """
        Convert selection dictionaries to a formatted string.

        Parameters
        ----------
        sel : dict
            Dictionary of dimension name to coordinate value.
        isel : dict
            Dictionary of dimension name to index value.

        Returns
        -------
        str
            String representation of the selection.
        """
        if sel:
            return ", ".join(
                [
                    self.dim_coord_to_str(dim, v, i)
                    for (dim, v), (_, i) in zip(sel.items(), isel.items())
                ]
            )
        return ""

    def var_name_to_str(self, var_name: Union[str, None]):
        """
        Convert a variable name to a display label.

        Parameters
        ----------
        var_name : str or None
            The name of the variable.

        Returns
        -------
        str or None
            The formatted variable name.
        """
        return var_name

    def var_pp_to_str(self, var_name, pp_var_name):
        """
        Convert a pair of variable names (e.g., prior and posterior) to a combined label.

        Parameters
        ----------
        var_name : str
            The name of the posterior variable.
        pp_var_name : str
            The name of the prior predictive variable.

        Returns
        -------
        str
            Combined label.
        """
        var_name_str = self.var_name_to_str(var_name)
        pp_var_name_str = self.var_name_to_str(pp_var_name)
        if var_name_str == pp_var_name_str:
            return f"{var_name_str}"
        return f"{var_name_str} / {pp_var_name_str}"

    def model_name_to_str(self, model_name):
        """
        Convert a model name to a display label.

        Parameters
        ----------
        model_name : str
            The name of the model.

        Returns
        -------
        str
            Display label for the model.
        """
        return model_name

    def make_label_vert(self, var_name: Union[str, None], sel: dict, isel: dict):
        """
        Create a multiline (vertical) label for a variable and its selection.

        Returns
        -------
        str
            Label with variable name and selection.
        """
        var_name_str = self.var_name_to_str(var_name)
        sel_str = self.sel_to_str(sel, isel)
        if not sel_str:
            return "" if var_name_str is None else var_name_str
        if var_name_str is None:
            return sel_str
        return f"{var_name_str}\n{sel_str}"

    def make_label_flat(self, var_name: str, sel: dict, isel: dict):
        """
        Create a flat (single-line) label with indexing format.

        Returns
        -------
        str
            Label in the format "var[dim:coord,...]".
        """
        var_name_str = self.var_name_to_str(var_name)
        sel_str = self.sel_to_str(sel, isel)
        if not sel_str:
            return "" if var_name_str is None else var_name_str
        if var_name_str is None:
            return sel_str
        return f"{var_name_str}[{sel_str}]"

    def make_pp_label(self, var_name, pp_var_name, sel, isel):
        """
        Create label for a prior-posterior pair.

        Returns
        -------
        str
            Multiline label showing both variable names and selection.
        """
        names = self.var_pp_to_str(var_name, pp_var_name)
        return self.make_label_vert(names, sel, isel)

    def make_model_label(self, model_name, label):
        """
        Create a model label combined with a component label.

        Returns
        -------
        str
            Combined model/component label.
        """
        model_name_str = self.model_name_to_str(model_name)
        if model_name_str is None:
            return label
        if label is None or label == "":
            return model_name_str
        return f"{model_name_str}: {label}"


class DimCoordLabeller(BaseLabeller):
    """
    Labeller that includes dimension names with coordinate values.
    """

    def dim_coord_to_str(self, dim, coord_val, coord_idx):
        return f"{dim}: {coord_val}"


class IdxLabeller(BaseLabeller):
    """
    Labeller that uses only coordinate indices.
    """

    def dim_coord_to_str(self, dim, coord_val, coord_idx):
        return f"{coord_idx}"


class DimIdxLabeller(BaseLabeller):
    """
    Labeller that combines dimension name with index.
    """

    def dim_coord_to_str(self, dim, coord_val, coord_idx):
        return f"{dim}#{coord_idx}"


class MapLabeller(BaseLabeller):
    """
    Labeller that maps names and values using user-provided dictionaries.
    """

    def __init__(self, var_name_map=None, dim_map=None, coord_map=None, model_name_map=None):
        self.var_name_map = {} if var_name_map is None else var_name_map
        self.dim_map = {} if dim_map is None else dim_map
        self.coord_map = {} if coord_map is None else coord_map
        self.model_name_map = {} if model_name_map is None else model_name_map

    def dim_coord_to_str(self, dim, coord_val, coord_idx):
        dim_str = self.dim_map.get(dim, dim)
        coord_str = self.coord_map.get(dim, {}).get(coord_val, coord_val)
        return super().dim_coord_to_str(dim_str, coord_str, coord_idx)

    def var_name_to_str(self, var_name):
        var_name_str = self.var_name_map.get(var_name, var_name)
        return super().var_name_to_str(var_name_str)

    def model_name_to_str(self, model_name):
        model_name_str = self.model_name_map.get(model_name, model_name)
        return super().model_name_to_str(model_name_str)


class NoVarLabeller(BaseLabeller):
    """
    Labeller that omits variable names.
    """

    def var_name_to_str(self, var_name):
        return None


class NoModelLabeller(BaseLabeller):
    """
    Labeller that omits model labels entirely.
    """

    def make_model_label(self, model_name, label):
        return label

