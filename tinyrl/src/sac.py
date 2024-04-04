import os, sys, importlib

from .compile import compile
from .link_accelerate import link_accelerate
from .link_mkl import link_mkl
from .load_module import load_module
from .render import render

from .. import CACHE_PATH


absolute_path = os.path.dirname(os.path.abspath(__file__))


def SAC(env_factory, enable_evaluation=True, **kwargs):
    module_name = 'tinyrl_sac'

    example_env = env_factory()
    observation_dim_flag = f'-DTINYRL_OBSERVATION_DIM={example_env.observation_space.shape[0]}'
    action_dim_flag = f'-DTINYRL_ACTION_DIM={example_env.action_space.shape[0]}'
    enable_evaluation_flag = '-DTINYRL_ENABLE_EVALUATION' if enable_evaluation else ''
    module_flag = f'-DTINYRL_MODULE_NAME={module_name}'

    flags = [observation_dim_flag, action_dim_flag, enable_evaluation_flag, module_flag]


    if sys.platform == "darwin":
        flags += link_accelerate()
    elif sys.platform == "linux":
        flags += link_mkl()


    source = os.path.join(absolute_path, '../interface/python_environment/python_environment.cpp')
    config_template = os.path.join(absolute_path, '../interface/python_environment/sac_default.h')

    render_output_directory = os.path.join(CACHE_PATH, 'template', module_name)
    
    os.makedirs(render_output_directory, exist_ok=True)
    render_output_path = os.path.join(render_output_directory, 'loop_core_config.h')
    new_config = render(config_template, render_output_path, **kwargs)
    if new_config:
        print('New SAC config detected, forcing recompilation...')
    
    loop_core_config_search_path_flag = f'-I{render_output_directory}'
    loop_core_config_flag = "-DTINYRL_USE_LOOP_CORE_CONFIG"

    flags += [loop_core_config_search_path_flag, loop_core_config_flag]

    output_path = compile(source, module_name, flags, force_recompile=new_config, **kwargs)

    module = load_module(module_name, output_path)
    module.set_environment_factory(env_factory)
    return module
