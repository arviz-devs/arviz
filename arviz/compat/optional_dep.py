import importlib
import sys


class OptionalDep(object):
    """Wrapper for optional library dependencies.

    Note that since only __getattr__ is implemented, if this object implements
    methods, those will be used *before* the true library is called.

    For example

    class PyMC3(OptionalDep):
        def trace_to_dataframe(*args, **kwargs):
            ...

    pm = PyMC3()
    pm.trace_to_dataframe(trace)  # calls the OptionalDep method
    pm.Normal('x', 0, 1)  # calls pymc3.Normal
    """
    def __init__(self, name):
        self.__name = name
        self.__module = None

    def __getattr__(self, name):
        if self.__module is None:
            try:
                self.__module = importlib.import_module(self.__name)
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    'Failed to import optional module {}'.format(self.__name), sys.exc_info()[0])
        return getattr(self.__module, name)
