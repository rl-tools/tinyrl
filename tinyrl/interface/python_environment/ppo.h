#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>

namespace rlt = rl_tools;



template <typename T, typename TI, typename ENVIRONMENT, TI T_EPISODE_STEP_LIMIT>
struct PPO_LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::Parameters<T, TI, ENVIRONMENT>{
    struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<T, TI>{
        static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.0;
        static constexpr TI N_EPOCHS = 2;
        static constexpr T GAMMA = 0.9;
    };
    static constexpr TI BATCH_SIZE = 128;
    static constexpr TI ACTOR_HIDDEN_DIM = 16;
    static constexpr TI CRITIC_HIDDEN_DIM = 16;
    static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 256;
    static constexpr TI N_ENVIRONMENTS = 4;
    static constexpr TI TOTAL_STEP_LIMIT = 300000;
    static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT/(ON_POLICY_RUNNER_STEPS_PER_ENV * N_ENVIRONMENTS) + 1;
    static constexpr TI EPISODE_STEP_LIMIT = T_EPISODE_STEP_LIMIT;
};

template <typename T, typename TI, typename RNG, typename ENVIRONMENT, TI T_EPISODE_STEP_LIMIT>
using PPO_LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<T, TI, RNG, ENVIRONMENT, PPO_LOOP_CORE_PARAMETERS<T, TI, ENVIRONMENT, T_EPISODE_STEP_LIMIT>, rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsMLP>;