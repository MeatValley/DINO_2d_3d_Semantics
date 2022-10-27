import importlib
from myutils.lists import make_list

def load_class(method_name, paths, concat=True):
    """
    Look for a file in different locations and return its method with the same name
    Optionally, you can use concat to search in path.method_name instead

    Parameters
    ----------
    method_name : str
        Name of the method we are searching for
    paths : str or list of str
        Folders in which the file will be searched
    concat : bool
        Flag to concatenate method_name to each path during the search

    Returns
    -------
    method : Function
        Loaded method
    """
    for path in make_list(paths): # for each path in paths
        # Create full path
        full_path = '{}.{}'.format(path, method_name) if concat else path
        if importlib.util.find_spec(full_path):
            return getattr(importlib.import_module(full_path), method_name) # Return method with same name as the file
    raise ValueError('Unknown class {}'.format(method_name))

