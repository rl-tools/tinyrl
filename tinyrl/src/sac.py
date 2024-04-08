import os, sys, importlib

from .compile import compile
from .load_module import load_module
from .render import render, sanitize_values
from .accelerate import acceleration_flags

from .. import CACHE_PATH


absolute_path = os.path.dirname(os.path.abspath(__file__))


def SAC(env_factory, enable_evaluation=True,
    # Compile-time parameters:
    GAMMA = 0.99,
    ALPHA = 0.5,
    ACTOR_BATCH_SIZE = 100, #32,
    CRITIC_BATCH_SIZE = 100, #32,
    CRITIC_TRAINING_INTERVAL = 1,
    ACTOR_TRAINING_INTERVAL = 1,
    CRITIC_TARGET_UPDATE_INTERVAL = 1,
    ACTOR_POLYAK = 1.0 - 0.005,
    CRITIC_POLYAK = 1.0 - 0.005,
    IGNORE_TERMINATION = False,
    TARGET_ENTROPY = None,
    ADAPTIVE_ALPHA = True,
    N_WARMUP_STEPS = None,
    STEP_LIMIT = 10000,
    REPLAY_BUFFER_CAP = None,
    EPISODE_STEP_LIMIT = 200,
    ACTOR_HIDDEN_DIM = 64,
    ACTOR_NUM_LAYERS = 3,
    ACTOR_ACTIVATION_FUNCTION = "RELU",
    CRITIC_HIDDEN_DIM = 64,
    CRITIC_NUM_LAYERS = 3,
    CRITIC_ACTIVATION_FUNCTION = "RELU",
    COLLECT_EPISODE_STATS = True,
    EPISODE_STATS_BUFFER_SIZE = 1000
    ):

    use_python_environment = type(env_factory) == type(lambda: None)

    example_env = env_factory()
    ACTION_DIM = example_env.action_space.shape[0]
    TARGET_ENTROPY = TARGET_ENTROPY if TARGET_ENTROPY is not None else -ACTION_DIM
    REPLAY_BUFFER_CAP = REPLAY_BUFFER_CAP if REPLAY_BUFFER_CAP is not None else STEP_LIMIT
    N_WARMUP_STEPS = N_WARMUP_STEPS if N_WARMUP_STEPS is not None else max(ACTOR_BATCH_SIZE, CRITIC_BATCH_SIZE)

    compile_time_parameters = sanitize_values(locals())

    module_name = 'tinyrl_sac'

    use_python_environment_flag = '-DTINYRL_USE_PYTHON_ENVIRONMENT' if use_python_environment else ''
    observation_dim_flag = f'-DTINYRL_OBSERVATION_DIM={example_env.observation_space.shape[0]}'
    action_dim_flag = f'-DTINYRL_ACTION_DIM={ACTION_DIM}'
    enable_evaluation_flag = '-DTINYRL_ENABLE_EVALUATION' if enable_evaluation else ''
    module_flag = f'-DTINYRL_MODULE_NAME={module_name}'

    flags = [use_python_environment_flag, observation_dim_flag, action_dim_flag, enable_evaluation_flag, module_flag]


    flags += acceleration_flags()


    source = os.path.join(absolute_path, '../interface/training/training.cpp')
    config_template = os.path.join(absolute_path, '../interface/algorithms/sac/template.h')

    render_output_directory = os.path.join(CACHE_PATH, 'template', module_name)
    
    os.makedirs(render_output_directory, exist_ok=True)
    render_output_path = os.path.join(render_output_directory, 'loop_core_config.h')
    new_config = render(config_template, render_output_path, **compile_time_parameters)
    if new_config:
        print('New SAC config detected, forcing recompilation...')
    
    loop_core_config_search_path_flag = f'-I{render_output_directory}'
    loop_core_config_flag = "-DTINYRL_USE_LOOP_CORE_CONFIG"

    flags += [loop_core_config_search_path_flag, loop_core_config_flag]

    output_path = compile(source, module_name, flags, force_recompile=new_config, **compile_time_parameters)

    module = load_module(module_name, output_path)
    module.set_environment_factory(env_factory)
    return module
