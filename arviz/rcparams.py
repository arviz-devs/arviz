"""ArviZ rcparams. Based on matplotlib's implementation."""
import locale
import logging
import os
import pprint
import re
import sys
import warnings
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Dict
from typing_extensions import Literal

NO_GET_ARGS: bool = False
try:
    from typing_extensions import get_args
except ImportError:
    NO_GET_ARGS = True


import numpy as np

_log = logging.getLogger(__name__)

ScaleKeyword = Literal["log", "negative_log", "deviance"]
ICKeyword = Literal["loo", "waic"]


def _make_validate_choice(accepted_values, allow_none=False, typeof=str):
    """Validate value is in accepted_values.

    Parameters
    ----------
    accepted_values : iterable
        Iterable containing all accepted_values.
    allow_none: boolean, optional
        Whether to accept ``None`` in addition to the values in ``accepted_values``.
    typeof: type, optional
        Type the values should be converted to.
    """
    # no blank lines allowed after function docstring by pydocstyle,
    # but black requires white line before function

    def validate_choice(value):
        if allow_none and (value is None or isinstance(value, str) and value.lower() == "none"):
            return None
        try:
            value = typeof(value)
        except (ValueError, TypeError) as err:
            raise ValueError(f"Could not convert to {typeof.__name__}") from err
        if isinstance(value, str):
            value = value.lower()

        if value in accepted_values:
            # Convert value to python boolean if string matches
            value = {"true": True, "false": False}.get(value, value)
            return value
        raise ValueError(
            "{} is not one of {}{}".format(
                value, accepted_values, " nor None" if allow_none else ""
            )
        )

    return validate_choice


def _make_validate_choice_regex(accepted_values, accepted_values_regex, allow_none=False):
    """Validate value is in accepted_values with regex.

    Parameters
    ----------
    accepted_values : iterable
        Iterable containing all accepted_values.
    accepted_values_regex : iterable
        Iterable containing all accepted_values with regex string.
    allow_none: boolean, optional
        Whether to accept ``None`` in addition to the values in ``accepted_values``.
    typeof: type, optional
        Type the values should be converted to.
    """
    # no blank lines allowed after function docstring by pydocstyle,
    # but black requires white line before function

    def validate_choice_regex(value):
        if allow_none and (value is None or isinstance(value, str) and value.lower() == "none"):
            return None
        value = str(value)
        if isinstance(value, str):
            value = value.lower()

        if value in accepted_values:
            # Convert value to python boolean if string matches
            value = {"true": True, "false": False}.get(value, value)
            return value
        elif any(re.match(pattern, value) for pattern in accepted_values_regex):
            return value
        raise ValueError(
            "{} is not one of {} or in regex {}{}".format(
                value, accepted_values, accepted_values_regex, " nor None" if allow_none else ""
            )
        )

    return validate_choice_regex


def _validate_positive_int(value):
    """Validate value is a natural number."""
    try:
        value = int(value)
    except ValueError as err:
        raise ValueError("Could not convert to int") from err
    if value > 0:
        return value
    else:
        raise ValueError("Only positive values are valid")


def _validate_float(value):
    """Validate value is a float."""
    try:
        value = float(value)
    except ValueError as err:
        raise ValueError("Could not convert to float") from err
    return value


def _validate_probability(value):
    """Validate a probability: a float between 0 and 1."""
    value = _validate_float(value)
    if (value < 0) or (value > 1):
        raise ValueError("Only values between 0 and 1 are valid.")
    return value


def _validate_boolean(value):
    """Validate value is a float."""
    if isinstance(value, str):
        value = value.lower()
    if value not in {True, False, "true", "false"}:
        raise ValueError("Only boolean values are valid.")
    return value is True or value == "true"


def _add_none_to_validator(base_validator):
    """Create a validator function that catches none and then calls base_fun."""
    # no blank lines allowed after function docstring by pydocstyle,
    # but black requires white line before function

    def validate_with_none(value):
        if value is None or isinstance(value, str) and value.lower() == "none":
            return None
        return base_validator(value)

    return validate_with_none


def _validate_bokeh_marker(value):
    """Validate the markers."""
    all_markers = (
        "Asterisk",
        "Circle",
        "CircleCross",
        "CircleX",
        "Cross",
        "Dash",
        "Diamond",
        "DiamondCross",
        "Hex",
        "InvertedTriangle",
        "Square",
        "SquareCross",
        "SquareX",
        "Triangle",
        "X",
    )
    if value not in all_markers:
        raise ValueError(f"{value} is not one of {all_markers}")
    return value


def _validate_dict_of_lists(values):
    if isinstance(values, dict):
        return {key: tuple(item) for key, item in values.items()}
    else:
        validated_dict = {}
        for value in values:
            tup = value.split(":", 1)
            if len(tup) != 2:
                raise ValueError(f"Could not interpret '{value}' as key: list or str")
            key, vals = tup
            key = key.strip(' "')
            vals = [val.strip(' "') for val in vals.strip(" [],").split(",")]
            if key in validated_dict:
                warnings.warn(f"Repeated key {key} when validating dict of lists")
            validated_dict[key] = tuple(vals)
        return validated_dict


def make_iterable_validator(scalar_validator, length=None, allow_none=False, allow_auto=False):
    """Validate value is an iterable datatype."""
    # based on matplotlib's _listify_validator function

    def validate_iterable(value):
        if allow_none and (value is None or isinstance(value, str) and value.lower() == "none"):
            return None
        if isinstance(value, str):
            if allow_auto and value.lower() == "auto":
                return "auto"
            value = tuple(v.strip("([ ])") for v in value.split(",") if v.strip())
        if np.iterable(value) and not isinstance(value, (set, frozenset)):
            val = tuple(scalar_validator(v) for v in value)
            if length is not None and len(val) != length:
                raise ValueError(f"Iterable must be of length: {length}")
            return val
        raise ValueError("Only ordered iterable values are valid")

    return validate_iterable


_validate_float_or_none = _add_none_to_validator(_validate_float)
_validate_positive_int_or_none = _add_none_to_validator(_validate_positive_int)
_validate_bokeh_bounds = make_iterable_validator(  # pylint: disable=invalid-name
    _validate_float_or_none, length=2, allow_none=True, allow_auto=True
)

METAGROUPS = {
    "posterior_groups": ["posterior", "posterior_predictive", "sample_stats", "log_likelihood"],
    "prior_groups": ["prior", "prior_predictive", "sample_stats_prior"],
    "posterior_groups_warmup": [
        "_warmup_posterior",
        "_warmup_posterior_predictive",
        "_warmup_sample_stats",
    ],
    "latent_vars": ["posterior", "prior"],
    "observed_vars": ["posterior_predictive", "observed_data", "prior_predictive"],
}

defaultParams = {  # pylint: disable=invalid-name
    "data.http_protocol": ("https", _make_validate_choice({"https", "http"})),
    "data.load": ("lazy", _make_validate_choice({"lazy", "eager"})),
    "data.metagroups": (METAGROUPS, _validate_dict_of_lists),
    "data.index_origin": (0, _make_validate_choice({0, 1}, typeof=int)),
    "data.log_likelihood": (True, _validate_boolean),
    "data.save_warmup": (False, _validate_boolean),
    "plot.backend": ("matplotlib", _make_validate_choice({"matplotlib", "bokeh"})),
    "plot.density_kind": ("kde", _make_validate_choice({"kde", "hist"})),
    "plot.max_subplots": (40, _validate_positive_int_or_none),
    "plot.point_estimate": (
        "mean",
        _make_validate_choice({"mean", "median", "mode"}, allow_none=True),
    ),
    "plot.bokeh.bounds_x_range": ("auto", _validate_bokeh_bounds),
    "plot.bokeh.bounds_y_range": ("auto", _validate_bokeh_bounds),
    "plot.bokeh.figure.dpi": (60, _validate_positive_int),
    "plot.bokeh.figure.height": (500, _validate_positive_int),
    "plot.bokeh.figure.width": (500, _validate_positive_int),
    "plot.bokeh.layout.order": (
        "default",
        _make_validate_choice_regex(
            {"default", r"column", r"row", "square", "square_trimmed"}, {r"\d*column", r"\d*row"}
        ),
    ),
    "plot.bokeh.layout.sizing_mode": (
        "fixed",
        _make_validate_choice(
            {
                "fixed",
                "stretch_width",
                "stretch_height",
                "stretch_both",
                "scale_width",
                "scale_height",
                "scale_both",
            }
        ),
    ),
    "plot.bokeh.layout.toolbar_location": (
        "above",
        _make_validate_choice({"above", "below", "left", "right"}, allow_none=True),
    ),
    "plot.bokeh.marker": ("Cross", _validate_bokeh_marker),
    "plot.bokeh.output_backend": ("webgl", _make_validate_choice({"canvas", "svg", "webgl"})),
    "plot.bokeh.show": (True, _validate_boolean),
    "plot.bokeh.tools": (
        "reset,pan,box_zoom,wheel_zoom,lasso_select,undo,save,hover",
        lambda x: x,
    ),
    "plot.matplotlib.show": (False, _validate_boolean),
    "stats.hdi_prob": (0.94, _validate_probability),
    "stats.information_criterion": (
        "loo",
        _make_validate_choice({"loo", "waic"} if NO_GET_ARGS else set(get_args(ICKeyword))),
    ),
    "stats.ic_pointwise": (True, _validate_boolean),
    "stats.ic_scale": (
        "log",
        _make_validate_choice(
            {"log", "negative_log", "deviance"} if NO_GET_ARGS else set(get_args(ScaleKeyword))
        ),
    ),
    "stats.ic_compare_method": (
        "stacking",
        _make_validate_choice({"stacking", "bb-pseudo-bma", "pseudo-bma"}),
    ),
}


class RcParams(MutableMapping):
    """Class to contain ArviZ default parameters.

    It is implemented as a dict with validation when setting items.
    """

    validate = {key: validate_fun for key, (_, validate_fun) in defaultParams.items()}

    # validate values on the way in
    def __init__(self, *args, **kwargs):
        self._underlying_storage: Dict[str, Any] = {}
        super().__init__()
        self.update(*args, **kwargs)

    def __setitem__(self, key, val):
        """Add validation to __setitem__ function."""
        try:
            try:
                cval = self.validate[key](val)
            except ValueError as verr:
                raise ValueError(f"Key {key}: {str(verr)}") from verr
            self._underlying_storage[key] = cval
        except KeyError as err:
            raise KeyError(
                "{} is not a valid rc parameter (see rcParams.keys() for "
                "a list of valid parameters)".format(key)
            ) from err

    def __getitem__(self, key):
        """Use underlying dict's getitem method."""
        return self._underlying_storage[key]

    def __delitem__(self, key):
        """Raise TypeError if someone ever tries to delete a key from RcParams."""
        raise TypeError("RcParams keys cannot be deleted")

    def clear(self):
        """Raise TypeError if someone ever tries to delete all keys from RcParams."""
        raise TypeError("RcParams keys cannot be deleted")

    def pop(self, key, default=None):
        """Raise TypeError if someone ever tries to delete a key from RcParams."""
        raise TypeError(
            "RcParams keys cannot be deleted. Use .get(key) of RcParams[key] to check values"
        )

    def popitem(self):
        """Raise TypeError if someone ever tries to delete a key from RcParams."""
        raise TypeError(
            "RcParams keys cannot be deleted. Use .get(key) of RcParams[key] to check values"
        )

    def setdefault(self, key, default=None):
        """Raise error when using setdefault, defaults are handled on initialization."""
        raise TypeError(
            "Defaults in RcParams are handled on object initialization during library"
            "import. Use arvizrc file instead."
            ""
        )

    def __repr__(self):
        """Customize repr of RcParams objects."""
        class_name = self.__class__.__name__
        indent = len(class_name) + 1
        repr_split = pprint.pformat(
            self._underlying_storage,
            indent=1,
            width=80 - indent,
        ).split("\n")
        repr_indented = ("\n" + " " * indent).join(repr_split)
        return f"{class_name}({repr_indented})"

    def __str__(self):
        """Customize str/print of RcParams objects."""
        return "\n".join(map("{0[0]:<22}: {0[1]}".format, sorted(self._underlying_storage.items())))

    def __iter__(self):
        """Yield sorted list of keys."""
        yield from sorted(self._underlying_storage.keys())

    def __len__(self):
        """Use underlying dict's len method."""
        return len(self._underlying_storage)

    def find_all(self, pattern):
        """
        Find keys that match a regex pattern.

        Return the subset of this RcParams dictionary whose keys match,
        using :func:`re.search`, the given ``pattern``.

        Notes
        -----
            Changes to the returned dictionary are *not* propagated to
            the parent RcParams dictionary.
        """
        pattern_re = re.compile(pattern)
        return RcParams((key, value) for key, value in self.items() if pattern_re.search(key))

    def copy(self):
        """Get a copy of the RcParams object."""
        return dict(self._underlying_storage)


def get_arviz_rcfile():
    """Get arvizrc file.

    The file location is determined in the following order:

    - ``$PWD/arvizrc``
    - ``$ARVIZ_DATA/arvizrc``
    - On Linux,
        - ``$XDG_CONFIG_HOME/arviz/arvizrc`` (if ``$XDG_CONFIG_HOME``
          is defined)
        - or ``$HOME/.config/arviz/arvizrc`` (if ``$XDG_CONFIG_HOME``
          is not defined)
    - On other platforms,
        - ``$HOME/.arviz/arvizrc`` if ``$HOME`` is defined

    Otherwise, the default defined in ``rcparams.py`` file will be used.
    """
    # no blank lines allowed after function docstring by pydocstyle,
    # but black requires white line before function

    def gen_candidates():
        yield os.path.join(os.getcwd(), "arvizrc")
        arviz_data_dir = os.environ.get("ARVIZ_DATA")
        if arviz_data_dir:
            yield os.path.join(arviz_data_dir, "arvizrc")
        xdg_base = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        if sys.platform.startswith(("linux", "freebsd")):
            configdir = str(Path(xdg_base, "arviz"))
        else:
            configdir = str(Path.home() / ".arviz")
        yield os.path.join(configdir, "arvizrc")

    for fname in gen_candidates():
        if os.path.exists(fname) and not os.path.isdir(fname):
            return fname

    return None


def read_rcfile(fname):
    """Return :class:`arviz.RcParams` from the contents of the given file.

    Unlike `rc_params_from_file`, the configuration class only contains the
    parameters specified in the file (i.e. default values are not filled in).
    """
    _error_details_fmt = 'line #%d\n\t"%s"\n\tin file "%s"'

    config = RcParams()
    with open(fname, "r", encoding="utf8") as rcfile:
        try:
            multiline = False
            for line_no, line in enumerate(rcfile, 1):
                strippedline = line.split("#", 1)[0].strip()
                if not strippedline:
                    continue
                if multiline:
                    if strippedline == "}":
                        multiline = False
                        val = aux_val
                    else:
                        aux_val.append(strippedline)
                        continue
                else:
                    tup = strippedline.split(":", 1)
                    if len(tup) != 2:
                        error_details = _error_details_fmt % (line_no, line, fname)
                        _log.warning("Illegal %s", error_details)
                        continue
                    key, val = tup
                    key = key.strip()
                    val = val.strip()
                    if key in config:
                        _log.warning("Duplicate key in file %r line #%d.", fname, line_no)
                    if key in {"data.metagroups"}:
                        aux_val = []
                        multiline = True
                        continue
                try:
                    config[key] = val
                except ValueError as verr:
                    error_details = _error_details_fmt % (line_no, line, fname)
                    raise ValueError(f"Bad val {val} on {error_details}\n\t{str(verr)}") from verr

        except UnicodeDecodeError:
            _log.warning(
                "Cannot decode configuration file %s with encoding "
                "%s, check LANG and LC_* variables.",
                fname,
                locale.getpreferredencoding(do_setlocale=False) or "utf-8 (default)",
            )
            raise

        return config


def rc_params(ignore_files=False):
    """Read and validate arvizrc file."""
    fname = None
    if not ignore_files:
        fname = get_arviz_rcfile()
    defaults = RcParams([(key, default) for key, (default, _) in defaultParams.items()])
    if fname is not None:
        file_defaults = read_rcfile(fname)
        defaults.update(file_defaults)
    return defaults


rcParams = rc_params()  # pylint: disable=invalid-name


class rc_context:  # pylint: disable=invalid-name
    """
    Return a context manager for managing rc settings.

    Parameters
    ----------
    rc : dict, optional
        Mapping containing the rcParams to modify temporally.
    fname : str, optional
        Filename of the file containing the rcParams to use inside the rc_context.

    Examples
    --------
    This allows one to do::

        with az.rc_context(fname='pystan.rc'):
            idata = az.load_arviz_data("radon")
            az.plot_posterior(idata, var_names=["gamma"])

    The plot would have settings from 'screen.rc'

    A dictionary can also be passed to the context manager::

        with az.rc_context(rc={'plot.max_subplots': None}, fname='pystan.rc'):
            idata = az.load_arviz_data("radon")
            az.plot_posterior(idata, var_names=["gamma"])

    The 'rc' dictionary takes precedence over the settings loaded from
    'fname'. Passing a dictionary only is also valid.
    """

    # Based on mpl.rc_context

    def __init__(self, rc=None, fname=None):
        self._orig = rcParams.copy()
        if fname:
            file_rcparams = read_rcfile(fname)
            rcParams.update(file_rcparams)
        if rc:
            rcParams.update(rc)

    def __enter__(self):
        """Define enter method of context manager."""
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Define exit method of context manager."""
        rcParams.update(self._orig)
