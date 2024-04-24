from .accelerate import acceleration_flags
from .load_module import load_module
from .compile import compile, compile_option
import os
absolute_path = os.path.dirname(os.path.abspath(__file__))


def get_time_limit(env):
    max_episode_steps = None
    try:
        max_episode_steps = env.spec.max_episode_steps
    except:
        pass
    return max_episode_steps


import re

def is_valid_identifier(s):
    pattern = r'^[_a-zA-Z][_a-zA-Z0-9]*$'
    return bool(re.match(pattern, s))

def compile_training(module_name, env_factory, flags, EPISODE_STEP_LIMIT=None, verbose=False, force_blas=False, enable_blas=True, force_recompile=False, enable_evaluation=False, evaluation_interval=None, num_evaluation_episodes=10, disable_tensorboard=True): #either gym env or custom env spec
    assert is_valid_identifier(module_name), "Module name is not a valid C identifier"
    assert enable_blas or not force_blas, "Cannot force BLAS acceleration without enabling BLAS"
    use_python_environment = type(env_factory) != dict
    if use_python_environment:
        example_env = env_factory()
        EPISODE_STEP_LIMIT = get_time_limit(example_env) if EPISODE_STEP_LIMIT is None else EPISODE_STEP_LIMIT
    
    if EPISODE_STEP_LIMIT is not None:
        flags += [compile_option("macro_definition", f'TINYRL_EPISODE_STEP_LIMIT={EPISODE_STEP_LIMIT}')]
        
    custom_environment_header_search_path = None if use_python_environment else env_factory["path"]
    custom_environment_flag = compile_option("header_search_path", custom_environment_header_search_path)
    use_python_environment_flag = compile_option("macro_definition", "TINYRL_USE_PYTHON_ENVIRONMENT") if use_python_environment else ''
    observation_dim_flag = compile_option("macro_definition", f'TINYRL_OBSERVATION_DIM={example_env.observation_space.shape[0]}') if use_python_environment else ''
    action_dim_flag = compile_option("macro_definition", f'TINYRL_ACTION_DIM={example_env.action_space.shape[0]}') if use_python_environment else ''
    evaluation_flags = [compile_option("macro_definition", 'TINYRL_ENABLE_EVALUATION') if enable_evaluation else '']
    evaluation_flags += [compile_option("macro_definition", f'TINYRL_EVALUATION_INTERVAL={evaluation_interval}') if evaluation_interval else '']
    evaluation_flags += [compile_option("macro_definition", f'TINYRL_NUM_EVALUATION_EPISODES={num_evaluation_episodes}') if num_evaluation_episodes else '']
    disable_tensorboard = compile_option("macro_definition", 'RLTOOLS_DISABLE_TENSORBOARD') if disable_tensorboard else ''
    module_flag = compile_option("macro_definition", f'TINYRL_MODULE_NAME={module_name}')
    flags += [use_python_environment_flag, custom_environment_flag, observation_dim_flag, action_dim_flag, *evaluation_flags, module_flag]
    if enable_blas:
        flags += acceleration_flags(module_name)
        flags += [compile_option("macro_definition", f'TINYRL_FORCE_BLAS={"true" if force_blas else "false"}') if force_blas else '']

    source = os.path.join(absolute_path, '../interface/training/training.cpp')
    output_path = compile(source, module_name, flags, verbose=verbose, force_recompile=force_recompile) #, **compile_time_parameters)

    module = load_module(module_name, output_path)
    if use_python_environment:
        module.set_environment_factory(env_factory)
    return module