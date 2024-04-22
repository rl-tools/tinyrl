import sys
import importlib
import importlib.util
import importlib.abc

def load_module(module_name, path, force_reload=True):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, importlib.abc.Loader)
    spec.loader.exec_module(module)
    if module_name not in sys.modules:
        sys.modules[module_name] = module
    return module