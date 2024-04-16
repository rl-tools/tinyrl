from .accelerate import acceleration_flags
from .load_module import load_module
from .compile import compile
import os
absolute_path = os.path.dirname(os.path.abspath(__file__))

def compile_training(module_name, env_factory, flags, verbose=False, force_recompile=False, enable_evaluation=True): #either gym env or custom env spec
    use_python_environment = type(env_factory) != dict
    if use_python_environment:
        example_env = env_factory()
        
    use_python_environment = example_env is not None
    custom_environment_header_search_path = None if use_python_environment else example_env["path"]
    custom_environment_flag = '-I' + custom_environment_header_search_path if custom_environment_header_search_path is not None else ''
    use_python_environment_flag = '-DTINYRL_USE_PYTHON_ENVIRONMENT' if use_python_environment else ''
    observation_dim_flag = f'-DTINYRL_OBSERVATION_DIM={example_env.observation_space.shape[0]}' if use_python_environment else ''
    action_dim_flag = f'-DTINYRL_ACTION_DIM={example_env.action_space.shape[0]}' if use_python_environment else ''
    enable_evaluation_flag = '-DTINYRL_ENABLE_EVALUATION' if enable_evaluation else ''
    module_flag = f'-DTINYRL_MODULE_NAME={module_name}'
    flags += [use_python_environment_flag, custom_environment_flag, observation_dim_flag, action_dim_flag, enable_evaluation_flag, module_flag]
    flags += acceleration_flags()
    source = os.path.join(absolute_path, '../interface/training/training.cpp')
    output_path = compile(source, module_name, flags, verbose=verbose, force_recompile=force_recompile) #, **compile_time_parameters)

    module = load_module(module_name, output_path)
    if use_python_environment:
        module.set_environment_factory(env_factory)
    return module