from functools import lru_cache, wraps


CACHE_ENABLED = True


def decorate_function(maxsize=128, typed=False):
    def inner_decorator(function):
        cache_extended_function = lru_cache(maxsize=128, typed=False)(function)

        @wraps(function)
        def smart_cache(*args, **kwargs):
            if CACHE_ENABLED:
                return cache_extended_function(*args, **kwargs)
            else:
                return function(*args, **kwargs)

        def cache_info():
            if CACHE_ENABLED:
                return cache_extended_function.cache_info()
            else:
                raise ValueError("Cache disabled and therefore not available.")
        smart_cache.cache_info = cache_info
        return smart_cache
    return inner_decorator


@decorate_function()
def f(a, b):
    return a + b


@decorate_function()
def g(a, b):
    return a * b

