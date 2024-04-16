import os
from .render import render, sanitize_values
from .training import compile_training

from .. import CACHE_PATH


absolute_path = os.path.dirname(os.path.abspath(__file__))


def get_time_limit(env):
    max_episode_steps = None
    try:
        max_episode_steps = env.spec.max_episode_steps
    except:
        pass
    return max_episode_steps



def SAC(env_factory, # can be either a lambda that creates a new Gym-like environment, or a dict with a specification of a C++ environment: {"path": "path/to/environment", "action_dim": xx, "observation_dim": yy}
    verbose=False,
    force_recompile=False,
    enable_evaluation=True,
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
    N_ENVIRONMENTS = 1,
    N_WARMUP_STEPS = None,
    STEP_LIMIT = 10000,
    REPLAY_BUFFER_CAP = None,
    EPISODE_STEP_LIMIT = None,
    ACTOR_HIDDEN_DIM = 64,
    ACTOR_NUM_LAYERS = 3,
    ACTOR_ACTIVATION_FUNCTION = "RELU",
    CRITIC_HIDDEN_DIM = 64,
    CRITIC_NUM_LAYERS = 3,
    CRITIC_ACTIVATION_FUNCTION = "RELU",
    COLLECT_EPISODE_STATS = True,
    EPISODE_STATS_BUFFER_SIZE = 1000,
    # optimizer
    OPTIMIZER_ALPHA=1e-3,
    OPTIMIZER_BETA_1=0.9,
    OPTIMIZER_BETA_2=0.999,
    OPTIMIZER_EPSILON=1e-7,
    ):


    use_python_environment = type(env_factory) != dict
    if use_python_environment:
        example_env = env_factory()
        ACTION_DIM = example_env.action_space.shape[0]
        EPISODE_STEP_LIMIT = get_time_limit(example_env) if EPISODE_STEP_LIMIT is None else EPISODE_STEP_LIMIT
        assert(EPISODE_STEP_LIMIT is not None)
    else:
        ACTION_DIM = env_factory["action_dim"]

    TARGET_ENTROPY = TARGET_ENTROPY if TARGET_ENTROPY is not None else -ACTION_DIM
    REPLAY_BUFFER_CAP = REPLAY_BUFFER_CAP if REPLAY_BUFFER_CAP is not None else STEP_LIMIT
    N_WARMUP_STEPS = N_WARMUP_STEPS if N_WARMUP_STEPS is not None else max(ACTOR_BATCH_SIZE, CRITIC_BATCH_SIZE)

    compile_time_parameters = sanitize_values(locals())

    module_name = 'tinyrl_sac'

    config_template = os.path.join(absolute_path, '../interface/algorithms/sac/template.h')

    print('TinyRL Cache Path: ', CACHE_PATH) if verbose else None
    render_output_directory = os.path.join(CACHE_PATH, 'template', module_name)
    
    os.makedirs(render_output_directory, exist_ok=True)
    render_output_path = os.path.join(render_output_directory, 'loop_core_config.h')
    new_config = render(config_template, render_output_path, **compile_time_parameters)
    if new_config:
        print('New SAC config detected, forcing recompilation...')
    
    loop_core_config_search_path_flag = f'-I{render_output_directory}'
    loop_core_config_flag = "-DTINYRL_USE_LOOP_CORE_CONFIG"
    flags = [loop_core_config_search_path_flag, loop_core_config_flag]
    return compile_training(module_name, env_factory, flags, verbose=verbose, force_recompile=(force_recompile or new_config), enable_evaluation=enable_evaluation)
