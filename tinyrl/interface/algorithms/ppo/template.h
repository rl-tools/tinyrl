#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>

namespace rlt = rl_tools;

template<typename T, typename TI, typename ENVIRONMENT, TI T_EPISODE_STEP_LIMIT>
struct PPO_LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::Parameters<T, TI, ENVIRONMENT>{
    struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::Parameters<T, TI, ENVIRONMENT>{
        static constexpr T GAMMA = $GAMMA;
        static constexpr T LAMBDA = $LAMBDA;
        static constexpr T EPSILON_CLIP = $EPSILON_CLIP;
        static constexpr T INITIAL_ACTION_STD = $INITIAL_ACTION_STD;
        static constexpr bool LEARN_ACTION_STD = $LEARN_ACTION_STD;
        static constexpr T ACTION_ENTROPY_COEFFICIENT = $ACTION_ENTROPY_COEFFICIENT;
        static constexpr T ADVANTAGE_EPSILON = $ADVANTAGE_EPSILON;
        static constexpr bool NORMALIZE_ADVANTAGE = $NORMALIZE_ADVANTAGE;
        static constexpr bool ADAPTIVE_LEARNING_RATE = $ADAPTIVE_LEARNING_RATE;
        static constexpr T ADAPTIVE_LEARNING_RATE_POLICY_KL_THRESHOLD = $ADAPTIVE_LEARNING_RATE_POLICY_KL_THRESHOLD;
        static constexpr T POLICY_KL_EPSILON = $POLICY_KL_EPSILON;
        static constexpr T ADAPTIVE_LEARNING_RATE_DECAY = $ADAPTIVE_LEARNING_RATE_DECAY;
        static constexpr T ADAPTIVE_LEARNING_RATE_MIN = $ADAPTIVE_LEARNING_RATE_MIN;
        static constexpr T ADAPTIVE_LEARNING_RATE_MAX = $ADAPTIVE_LEARNING_RATE_MAX;
        static constexpr bool NORMALIZE_OBSERVATIONS = $NORMALIZE_OBSERVATIONS;
        static constexpr TI N_WARMUP_STEPS_CRITIC = $N_WARMUP_STEPS_CRITIC;
        static constexpr TI N_WARMUP_STEPS_ACTOR = $N_WARMUP_STEPS_ACTOR;
        static constexpr TI N_EPOCHS = $N_EPOCHS;
        static constexpr bool IGNORE_TERMINATION = $IGNORE_TERMINATION; // ignoring the termination flag is useful for training on environments with negative rewards, where the agent would try to terminate the episode as soon as possible otherwise
    };

    static constexpr TI STEP_LIMIT = $STEP_LIMIT;
    static constexpr TI ACTOR_HIDDEN_DIM = $ACTOR_HIDDEN_DIM;
    static constexpr TI ACTOR_NUM_LAYERS = $ACTOR_NUM_LAYERS;
    static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::$ACTOR_ACTIVATION_FUNCTION;
    static constexpr TI CRITIC_HIDDEN_DIM = $CRITIC_HIDDEN_DIM;
    static constexpr TI CRITIC_NUM_LAYERS = $CRITIC_NUM_LAYERS;
    static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::$CRITIC_ACTIVATION_FUNCTION;
    static constexpr TI EPISODE_STEP_LIMIT = T_EPISODE_STEP_LIMIT;
    static constexpr TI N_ENVIRONMENTS = $N_ENVIRONMENTS;
    static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = $ON_POLICY_RUNNER_STEPS_PER_ENV;
    static constexpr TI BATCH_SIZE = $BATCH_SIZE;

    struct OPTIMIZER_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<T>{
        static constexpr T ALPHA = $OPTIMIZER_ALPHA;
        static constexpr T BETA_1 = $OPTIMIZER_BETA_1;
        static constexpr T BETA_2 = $OPTIMIZER_BETA_2;
        static constexpr T EPSILON = $OPTIMIZER_EPSILON;
    };
};


template <typename T, typename TI, typename RNG, typename ENVIRONMENT, TI T_EPISODE_STEP_LIMIT>
using LOOP_CORE_CONFIG_FACTORY = rlt::rl::algorithms::ppo::loop::core::Config<T, TI, RNG, ENVIRONMENT, PPO_LOOP_CORE_PARAMETERS<T, TI, ENVIRONMENT, T_EPISODE_STEP_LIMIT>>;