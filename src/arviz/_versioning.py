import logging
import sys
from importlib import import_module

_log = logging.getLogger(__name__)


def import_arviz_subpackage(module_name: str, *, version_fallback: str | None = None):
    try:
        module = sys.modules.get(module_name)
        if module is None:
            module = import_module(module_name)
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
