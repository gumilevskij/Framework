#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 09:52:57 2021

@author: Alexei Goumilevski
"""

import warnings
import functools
import time

def debug(func):
    """Print the function signature and return value."""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]                      # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)           # 3
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")           # 4
        return value
    return wrapper_debug


def timer(func):
    """Print the runtime of the decorated function."""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.1f} secs")
        return value
    return wrapper_timer


def repeat(_func=None, *args, num_times=2):
    """Repeat decorator.
    
    If name has been called without arguments, the decorated function will be passed in as _func. 
    If it has been called with arguments, then _func will be None, and some of the keyword arguments 
    may have been changed from their default values. The * in the argument list means that the 
    remaining arguments can’t be called as positional arguments.
    

    Parameters:
        _func : function object, optional
            Repeats function executions. The default is None.
        args : Arguments
            Function arguments.
        num_times : int, optional
            Number of times to reprat fuction calls. The default is 2.

    Returns:
        decorator_repeat function

    """
    def decorator_repeat(func):
        @functools.wraps(func)
        def wrapper_repeat(*args, **kwargs):
            for _ in range(num_times):
                value = func(*args, **kwargs)
            return value
        return wrapper_repeat

    if _func is None:
        return decorator_repeat
    else:
        return decorator_repeat(_func)


def count_calls(func):
    """Decorator that counts the number of times a function is called."""
    @functools.wraps(func)
    def wrapper_count_calls(*args, **kwargs):
        wrapper_count_calls.num_calls += 1
        print(f"Call {wrapper_count_calls.num_calls} of {func.__name__!r}")
        return func(*args, **kwargs)
    wrapper_count_calls.num_calls = 0
    return wrapper_count_calls


def singleton(cls):
    """Make a class a Singleton class (only one instance)."""
    @functools.wraps(cls)
    def wrapper_singleton(*args, **kwargs):
        if not wrapper_singleton.instance:
            wrapper_singleton.instance = cls(*args, **kwargs)
        return wrapper_singleton.instance
    wrapper_singleton.instance = None
    return wrapper_singleton


def cache(func):
    """Keep a cache of previous function calls."""
    @functools.wraps(func)
    def wrapper_cache(*args, **kwargs):
        cache_key = args + tuple(kwargs.items())
        if cache_key not in wrapper_cache.cache:
            wrapper_cache.cache[cache_key] = func(*args, **kwargs)
        return wrapper_cache.cache[cache_key]
    wrapper_cache.cache = dict()
    return wrapper_cache


def deprecated(func):
    """This is a decorator which can be used to mark functions as deprecated."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        code = func.__code__
        warnings.warn_explicit(
            "Call to deprecated function {}.".format(func.__name__),
            category=Warning,
            filename=code.co_filename,
            lineno=code.co_firstlineno + 1,
        )
        return func(*args, **kwargs)
    return new_func