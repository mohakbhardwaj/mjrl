import sys, importlib, os
def import_from_path(path, base_path=None) :
    """ Import module from an absolute/relative file path. """
    if base_path is not None:  # `path` is relative to the base_path
        path = os.path.join(base_path, path)
    pathname, filename = os.path.split(path)
    modulename = filename.split('.')[0]
    spec = importlib.util.spec_from_file_location(modulename, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modulename] = mod
    spec.loader.exec_module(mod)
    return mod

import copy
def parse_and_update_dict(base_dict, new_dict, token=':'):
    base_dict = copy.deepcopy(base_dict)
    for k,v in new_dict.items():
        keys = k.split(token)
        d = base_dict
        for i in range(len(keys)):
            ki = keys[i]
            if i==len(keys)-1:
                if ki in d:
                    assert not isinstance(d[ki], dict), "Inconsistent key in new_dict."
                d[ki] = v
            else:
                if ki not in d:
                    d[ki] = {}
                assert isinstance(d[ki], dict), "Inconsistent key in new_dict."
                d = d[ki]
    return base_dict