import importlib
def load_module(module, path):
    spec = importlib.util.spec_from_file_location(module, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, importlib.abc.Loader)
    spec.loader.exec_module(module)
    return module