import os
from .render import render, sanitize_values
from .training import compile_training
from .compile import compile_option

from .. import CACHE_PATH


absolute_path = os.path.dirname(os.path.abspath(__file__))

def SAC(env_factory, # can be either a lambda that creates a new Gym-like environment, or a dict with a specification of a C++ environment: {"path": "path/to/environment", "action_dim": xx, "observation_dim": yy}
    verbose=False,
    force_recompile=False,
    enable_evaluation=True,
    evaluation_interval=1000,
    num_evaluation_episodes=10,
    interface_name="default", # this is the namespace used for the compilation of the TinyRL interface (in a temporary directory) and should be unique if run in parallel. We don't choose a random uuid because it would invalidate the cache and require a re-compilation every time
    # Compile-time parameters:
    # Same set of parameters as: rl::algorithms::td3::DefaultParameters
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
    ACTION_LOG_STD_LOWER_BOUND=-20,
    ACTION_LOG_STD_UPPER_BOUND=2,
    # Same set of parameters as rl::algorithms::sac::loop::core::DefaultParameters
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
    **kwargs
    ):
    verbose = verbose or "TINYRL_VERBOSE" in os.environ


    # The action dim is needed to set the default target entropy
    use_python_environment = type(env_factory) != dict
    if use_python_environment:
        example_env = env_factory()
        ACTION_DIM = example_env.action_space.shape[0]
    else:
        ACTION_DIM = env_factory["action_dim"]

    TARGET_ENTROPY = TARGET_ENTROPY if TARGET_ENTROPY is not None else -ACTION_DIM
    REPLAY_BUFFER_CAP = REPLAY_BUFFER_CAP if REPLAY_BUFFER_CAP is not None else STEP_LIMIT
    N_WARMUP_STEPS = N_WARMUP_STEPS if N_WARMUP_STEPS is not None else max(ACTOR_BATCH_SIZE, CRITIC_BATCH_SIZE)

    compile_time_parameters = sanitize_values(locals())

    module_name = f'tinyrl_sac_{interface_name}'

    config_template = os.path.join(absolute_path, '../interface/algorithms/sac/template.h')

    print('TinyRL Cache Path: ', CACHE_PATH, flush=True) if verbose else None
    render_output_directory = os.path.join(CACHE_PATH, 'template', module_name)
    
    os.makedirs(render_output_directory, exist_ok=True)
    render_output_path = os.path.join(render_output_directory, 'loop_core_config.h')
    new_config = render(config_template, render_output_path, **compile_time_parameters)
    if new_config:
        print('New SAC config detected, forcing recompilation...')
    
    loop_core_config_search_path_flag = compile_option("header_search_path", render_output_directory)
    loop_core_config_flag = compile_option("macro_definition", "TINYRL_USE_LOOP_CORE_CONFIG")
    flags = [loop_core_config_search_path_flag, loop_core_config_flag]
    return compile_training(module_name, env_factory, flags, verbose=verbose, force_recompile=(force_recompile or new_config), enable_evaluation=enable_evaluation, evaluation_interval=evaluation_interval, num_evaluation_episodes=num_evaluation_episodes, **kwargs)
