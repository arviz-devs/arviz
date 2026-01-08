# src/arviz/_versioning.py
import logging

_log = logging.getLogger(__name__)


def import_arviz_subpackage(
    module_name: str,
    *,
    version_fallback: str | None = None,
):
    """
    Import an ArviZ subpackage and return its version string.

    Parameters
    ----------
    module_name : str
        Name of the subpackage (e.g. 'arviz_base').
    version_fallback : str, optional
        Version to use if the module does not expose __version__.
    """
    try:
        module = __import__(module_name)
    except ModuleNotFoundError as err:
        raise ImportError(
            f"arviz's dependency {module_name} is not installed",
            name="arviz",
        ) from err

    version = getattr(module, "__version__", version_fallback)

    _log.info(
        "%s %s available, exposing its functions as part of the `arviz` namespace",
        module_name,
        version,
    )

    return version
