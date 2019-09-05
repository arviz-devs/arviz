"""ArviZ rcparams. Based on matplotlib's implementation."""
import sys
import os
from pathlib import Path
import re
import pprint
import logging
import locale
from collections import MutableMapping

_log = logging.getLogger(__name__)


def _make_validate_choice(accepted_values):
    """Validate value is in accepted_values."""
    # no blank lines allowed after function docstring by pydocstyle,
    # but black requires white line before function

    def validate_choice(value):
        if value.lower() in accepted_values:
            return value.lower()
        raise ValueError("{} is not one of {}".format(value, accepted_values))

    return validate_choice


def _validate_positive_int(value):
    """Validate value is a natural number."""
    try:
        value = int(value)
    except ValueError:
        raise ValueError("Could not convert to int")
    if value > 0:
        return value
    else:
        raise ValueError("Only positive values are valid")


def _validate_positive_int_or_none(value):
    """Validate value is a natural number or None."""
    if value is None or isinstance(value, str) and value.lower() == "none":
        return None
    else:
        return _validate_positive_int(value)


defaultParams = {  # pylint: disable=invalid-name
    "data.load": ("lazy", _make_validate_choice(("lazy", "eager"))),
    "plot.max_subplots": (40, _validate_positive_int_or_none),
    "stats.information_criterion": ("waic", _make_validate_choice(("waic", "loo"))),
}


class RcParams(MutableMapping, dict):  # pylint: disable=too-many-ancestors
    """Class to contain ArviZ default parameters.

    It is implemented as a dict with validation when setting items.
    """

    validate = {key: validate_fun for key, (_, validate_fun) in defaultParams.items()}

    # validate values on the way in
    def __init__(self, *args, **kwargs):  # pylint: disable=super-init-not-called
        self.update(*args, **kwargs)

    def __setitem__(self, key, val):
        """Add validation to __setitem__ function."""
        try:
            try:
                cval = self.validate[key](val)
            except ValueError as verr:
                raise ValueError("Key %s: %s" % (key, str(verr)))
            dict.__setitem__(self, key, cval)
        except KeyError:
            raise KeyError(
                "{} is not a valid rc parameter (see rcParams.keys() for "
                "a list of valid parameters)".format(key)
            )

    def __getitem__(self, key):
        """Use dict getitem method."""
        return dict.__getitem__(self, key)

    def __delitem__(self, key):
        """Raise TypeError if someone ever tries to delete a key from RcParams."""
        raise TypeError("RcParams keys cannot be deleted")

    def __repr__(self):
        """Customize repr of RcParams objects."""
        class_name = self.__class__.__name__
        indent = len(class_name) + 1
        repr_split = pprint.pformat(dict(self), indent=1, width=80 - indent).split("\n")
        repr_indented = ("\n" + " " * indent).join(repr_split)
        return "{}({})".format(class_name, repr_indented)

    def __str__(self):
        """Customize str/print of RcParams objects."""
        return "\n".join(map("{0[0]:<22}: {0[1]}".format, sorted(self.items())))

    def __iter__(self):
        """Yield sorted list of keys."""
        yield from sorted(dict.__iter__(self))

    def __len__(self):
        """Use dict len method."""
        return dict.__len__(self)

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
        return {k: dict.__getitem__(self, k) for k in self}


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
    with open(fname, "r") as rcfile:
        try:
            for line_no, line in enumerate(rcfile, 1):
                strippedline = line.split("#", 1)[0].strip()
                if not strippedline:
                    continue
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
                try:
                    config[key] = val
                except ValueError as verr:
                    error_details = _error_details_fmt % (line_no, line, fname)
                    raise ValueError("Bad val {} on {}\n\t{}".format(val, error_details, str(verr)))

        except UnicodeDecodeError:
            _log.warning(
                "Cannot decode configuration file %s with encoding "
                "%s, check LANG and LC_* variables.",
                fname,
                locale.getpreferredencoding(do_setlocale=False) or "utf-8 (default)",
            )
            raise

        return config


def rc_params():
    """Read and validate arvizrc file."""
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
        Filename of the file containig the rcParams to use inside the rc_context.

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
