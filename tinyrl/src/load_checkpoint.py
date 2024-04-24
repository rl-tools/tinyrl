import os, hashlib
from .. import CACHE_PATH
from .load_module import load_module
from .compile import compile
from .accelerate import acceleration_flags

def load_checkpoint_from_path(checkpoint_path, interface_name="default", force_recompile=False, verbose=False):
    with open(checkpoint_path, "r") as f:
        checkpoint = f.read()
    return load_checkpoint(checkpoint, interface_name="default", force_recompile=force_recompile, verbose=verbose)

def load_checkpoint(checkpoint, interface_name="default", force_recompile=False, verbose=False):
    module_name = f"load_checkpoint_{interface_name}"
    output_directory = os.path.join(CACHE_PATH, "checkpoint", module_name)
    
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory, 'checkpoint.h')
    with open(output_path, "w") as f:
        f.write(checkpoint)

    module_flag = f'-DTINYRL_MODULE_NAME={module_name}'
    header_search_path_flag = f'-I{output_directory}'

    flags = [module_flag, header_search_path_flag]

    flags += acceleration_flags(module_name)

    absolute_path = os.path.dirname(os.path.abspath(__file__))
    source = os.path.join(absolute_path, '../interface/inference/inference.cpp')
    output_path = compile(source, module_name, flags, force_recompile=force_recompile, verbose=verbose)
    module = load_module(module_name, output_path)
    return module

