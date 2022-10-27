import numpy as np

def is_list(data):
    """checks if data is a list"""
    return isinstance(data, tuple)

def make_list(var, n=None):
    """
    Wraps the input into a list, and optionally repeats it to be size n

    Parameters
    ----------
    var : Any
        Variable to be wrapped in a list
    n : int
        How much the wrapped variable will be repeated

    Returns
    -------
    var_list : list
        List generated from var
    """
    var = var if is_list(var) else [var]
    if n is None:
        return var
    else:
        assert len(var) == 1 or len(var) == n, 'Wrong list length for make_list'
        return var * n if len(var) == 1 else var
