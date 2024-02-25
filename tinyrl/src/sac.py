import os, sys, importlib

from .compile import compile
from .link_accelerate import link_accelerate
from .link_mkl import link_mkl
from .load_module import load_module


absolute_path = os.path.dirname(os.path.abspath(__file__))


def SAC(env_factory, enable_evaluation=True, **kwargs):

    example_env = env_factory()
    observation_dim_flag = f'-DTINYRL_OBSERVATION_DIM={example_env.observation_space.shape[0]}'
    action_dim_flag = f'-DTINYRL_ACTION_DIM={example_env.action_space.shape[0]}'
    enable_evaluation_flag = '-DTINYRL_ENABLE_EVALUATION' if enable_evaluation else ''

    flags = [observation_dim_flag, action_dim_flag, enable_evaluation_flag]

    if sys.platform == "darwin":
        flags += link_accelerate()
    elif sys.platform == "linux":
        flags += link_mkl()


    module_name = 'tinyrl_sac'
    source = os.path.join(absolute_path, '../interface/python_environment/python_environment.cpp')

    output_path = compile(source, module_name, flags, **kwargs)

    module = load_module(module_name, output_path)
    module.set_environment_factory(env_factory)
    return module
