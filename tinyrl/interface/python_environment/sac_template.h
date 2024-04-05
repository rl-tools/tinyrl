#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/algorithms/sac/loop/core/operations_cpu.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>


namespace rlt = rl_tools;

template <typename T, typename TI, typename ENVIRONMENT, TI T_EPISODE_STEP_LIMIT>
struct SAC_LOOP_CORE_PARAMETERS: rlt::rl::algorithms::sac::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
    struct SAC_PARAMETERS {
        static constexpr T GAMMA = $GAMMA;
        static constexpr T ALPHA = $ALPHA;
        static constexpr TI ACTOR_BATCH_SIZE = $ACTOR_BATCH_SIZE;
        static constexpr TI CRITIC_BATCH_SIZE = $CRITIC_BATCH_SIZE;
        static constexpr TI N_WARMUP_STEPS_CRITIC = 0;
        static constexpr TI N_WARMUP_STEPS_ACTOR = 0;
        static constexpr TI CRITIC_TRAINING_INTERVAL = $CRITIC_TRAINING_INTERVAL;
        static constexpr TI ACTOR_TRAINING_INTERVAL = $ACTOR_TRAINING_INTERVAL;
        static constexpr TI CRITIC_TARGET_UPDATE_INTERVAL = $CRITIC_TARGET_UPDATE_INTERVAL;
        static constexpr T ACTOR_POLYAK = $ACTOR_POLYAK;
        static constexpr T CRITIC_POLYAK = $CRITIC_POLYAK;
        static constexpr bool IGNORE_TERMINATION = $IGNORE_TERMINATION; 
        static constexpr T TARGET_ENTROPY = $TARGET_ENTROPY;
        static constexpr bool ADAPTIVE_ALPHA = $ADAPTIVE_ALPHA;
    };
    static constexpr TI N_WARMUP_STEPS = $N_WARMUP_STEPS;
    static constexpr TI STEP_LIMIT = $STEP_LIMIT;
    static constexpr TI REPLAY_BUFFER_CAP = $REPLAY_BUFFER_CAP;
    static constexpr TI EPISODE_STEP_LIMIT = $EPISODE_STEP_LIMIT;
    static constexpr TI ACTOR_HIDDEN_DIM = $ACTOR_HIDDEN_DIM;
    static constexpr TI ACTOR_NUM_LAYERS = $ACTOR_NUM_LAYERS;
    static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::$ACTOR_ACTIVATION_FUNCTION;
    static constexpr TI CRITIC_HIDDEN_DIM = $CRITIC_HIDDEN_DIM;
    static constexpr TI CRITIC_NUM_LAYERS = $CRITIC_NUM_LAYERS;
    static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::$CRITIC_ACTIVATION_FUNCTION;
    static constexpr bool COLLECT_EPISODE_STATS = $COLLECT_EPISODE_STATS;
    static constexpr TI EPISODE_STATS_BUFFER_SIZE = $EPISODE_STATS_BUFFER_SIZE;
};

template <typename T, typename TI, typename RNG, typename ENVIRONMENT, TI T_EPISODE_STEP_LIMIT>
using LOOP_CORE_CONFIG_TEMPLATE = rlt::rl::algorithms::sac::loop::core::Config<T, TI, RNG, ENVIRONMENT, SAC_LOOP_CORE_PARAMETERS<T, TI, ENVIRONMENT, T_EPISODE_STEP_LIMIT>>;