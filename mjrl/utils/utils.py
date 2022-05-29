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

    # sys.path.append(pathname)

    return mod