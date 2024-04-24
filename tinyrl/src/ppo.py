import os, math
import inspect
from .render import render, sanitize_values
from .training import compile_training
from .compile import compile_option

from .. import CACHE_PATH


absolute_path = os.path.dirname(os.path.abspath(__file__))

def PPO(env_factory, # can be either a lambda that creates a new Gym-like environment, or a dict with a specification of a C++ environment: {"path": "path/to/environment", "action_dim": xx, "observation_dim": yy}
    verbose=False,
    force_recompile=False,
    enable_evaluation=True,
    evaluation_interval=None,
    num_evaluation_episodes=10,
    interface_name="default", # this is the namespace used for the compilation of the TinyRL interface (in a temporary directory) and should be unique if run in parallel. We don't choose a random uuid because it would invalidate the cache and require a re-compilation every time
    # Compile-time parameters:
    # Same set of parameters as: rl::algorithms::ppo::DefaultParameters
    GAMMA = 0.99,
    LAMBDA = 0.95,
    EPSILON_CLIP = 0.2,
    INITIAL_ACTION_STD = 0.5,
    LEARN_ACTION_STD = True,
    ACTION_ENTROPY_COEFFICIENT = 0.01,
    ADVANTAGE_EPSILON = 1e-8,
    NORMALIZE_ADVANTAGE = True,
    ADAPTIVE_LEARNING_RATE = False,
    ADAPTIVE_LEARNING_RATE_POLICY_KL_THRESHOLD = 0.008,
    POLICY_KL_EPSILON = 1e-5,
    ADAPTIVE_LEARNING_RATE_DECAY = 1/1.5,
    ADAPTIVE_LEARNING_RATE_MIN = 1e-6,
    ADAPTIVE_LEARNING_RATE_MAX = 1e-2,
    NORMALIZE_OBSERVATIONS = False,
    N_WARMUP_STEPS_CRITIC = 0,
    N_WARMUP_STEPS_ACTOR = 0,
    N_EPOCHS = 10,
    IGNORE_TERMINATION = False, # ignoring the termination flag is useful for training on environments with negative rewards, where the agent would try to terminate the episode as soon as possible otherwise
    # Same set of parameters as rl::algorithms::ppo::loop::core::DefaultParameters
    STEP_LIMIT = None,
    TOTAL_STEP_LIMIT = 100000, # This is environment step limit => translated into the closets ppo step limit (smaller or equal in total environment steps)
    ACTOR_HIDDEN_DIM = 64,
    ACTOR_NUM_LAYERS = 3,
    ACTOR_ACTIVATION_FUNCTION = "RELU",
    CRITIC_HIDDEN_DIM = 64,
    CRITIC_NUM_LAYERS = 3,
    CRITIC_ACTIVATION_FUNCTION = "RELU",
    EPISODE_STEP_LIMIT = 1000,
    N_ENVIRONMENTS = 64,
    ON_POLICY_RUNNER_STEPS_PER_ENV = 64,
    BATCH_SIZE = 64,
    # optimizer
    OPTIMIZER_ALPHA=1e-3,
    OPTIMIZER_BETA_1=0.9,
    OPTIMIZER_BETA_2=0.999,
    OPTIMIZER_EPSILON=1e-7,
    **kwargs
    ):
    assert STEP_LIMIT is not None or TOTAL_STEP_LIMIT is not None, "Either STEP_LIMIT or TOTAL_STEP_LIMIT must be set"
    evaluation_interval = evaluation_interval if evaluation_interval is not None else 10
    verbose = verbose or "TINYRL_VERBOSE" in os.environ
    if STEP_LIMIT is None:
        STEP_LIMIT = math.floor(TOTAL_STEP_LIMIT/(ON_POLICY_RUNNER_STEPS_PER_ENV * N_ENVIRONMENTS))

    compile_time_parameters = sanitize_values({k:v for k, v in locals().items() if k in inspect.signature(PPO).parameters.keys()})

    module_name = f'tinyrl_ppo_{interface_name}'

    config_template = os.path.join(absolute_path, '../interface/algorithms/ppo/template.h')

    print('TinyRL Cache Path: ', CACHE_PATH) if verbose else None
    render_output_directory = os.path.join(CACHE_PATH, 'template', module_name)
    
    os.makedirs(render_output_directory, exist_ok=True)
    render_output_path = os.path.join(render_output_directory, 'loop_core_config.h')
    new_config = render(config_template, render_output_path, **compile_time_parameters)
    if new_config:
        print('New PPO config detected, forcing recompilation...')
    
    loop_core_config_search_path_flag = compile_option("header_search_path", render_output_directory)
    loop_core_config_flag = compile_option("macro_definition", "TINYRL_USE_LOOP_CORE_CONFIG")
    flags = [loop_core_config_search_path_flag, loop_core_config_flag]
    return compile_training(module_name, env_factory, flags, verbose=verbose, force_recompile=(force_recompile or new_config), enable_evaluation=enable_evaluation, evaluation_interval=evaluation_interval, num_evaluation_episodes=num_evaluation_episodes, **kwargs)
