
#include <torch/extension.h>

namespace py = pybind11;

#ifndef TINYRL_DTYPE
#define TINYRL_DTYPE float
#endif

//#define RL_TOOLS_BACKEND_DISABLE_BLAS

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>

#include "operations_cpu.h"
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>


#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using TI = typename DEVICE::index_t;

using T = TINYRL_DTYPE;
static constexpr TI OBSERVATION_DIM = TINYRL_OBSERVATION_DIM;
static constexpr TI ACTION_DIM = TINYRL_ACTION_DIM;

using ENVIRONMENT_SPEC = PythonEnvironmentSpecification<T, TI, OBSERVATION_DIM, ACTION_DIM>;
using ENVIRONMENT = PythonEnvironment<ENVIRONMENT_SPEC>;

struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::Parameters<T, TI, ENVIRONMENT>{
    struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<T, TI>{
        static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.0;
        static constexpr TI N_EPOCHS = 2;
        static constexpr T GAMMA = 0.9;
    };
    static constexpr TI BATCH_SIZE = 256;
    static constexpr TI ACTOR_HIDDEN_DIM = 64;
    static constexpr TI CRITIC_HIDDEN_DIM = 64;
    static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 1024;
    static constexpr TI N_ENVIRONMENTS = 4;
    static constexpr TI TOTAL_STEP_LIMIT = 300000;
    static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT/(ON_POLICY_RUNNER_STEPS_PER_ENV * N_ENVIRONMENTS) + 1;
    static constexpr TI EPISODE_STEP_LIMIT = 200;
};
using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsMLP>;
template <typename NEXT>
struct LOOP_EVAL_PARAMETERS: rlt::rl::loop::steps::evaluation::Parameters<T, TI, NEXT>{
    static constexpr TI EVALUATION_INTERVAL = 1;
    static constexpr TI NUM_EVALUATION_EPISODES = 1000;
    static constexpr TI N_EVALUATIONS = NEXT::CORE_PARAMETERS::STEP_LIMIT / EVALUATION_INTERVAL;
};

DEVICE device;
using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CORE_CONFIG, LOOP_EVAL_PARAMETERS<LOOP_CORE_CONFIG>>;
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_EVAL_CONFIG>;
using LOOP_CONFIG = LOOP_TIMING_CONFIG;
using LOOP_STATE = typename LOOP_CONFIG::template State<LOOP_CONFIG>;

void set_environment_factory(std::function<py::object()> p_environment_factory) {
    environment_factory = p_environment_factory;
}

void init(LOOP_STATE& state, TI seed){
    rlt::malloc(device, state);
    rlt::init(device, state, seed);
    state.actor_optimizer.parameters.alpha = 1e-3;
    state.critic_optimizer.parameters.alpha = 1e-3;
}

bool step(LOOP_STATE& state){
    return rlt::step(device, state);
}

PYBIND11_MODULE(rl_tools, m) {
    m.doc() = "Python Environment Wrapper";

    m.def("set_environment_factory", &set_environment_factory, "Set the environment factory");
    py::class_<LOOP_STATE>(m, "LoopState")
            .def(py::init<>());
    m.def("init", &init, "Initialize the loop state");
    m.def("step", &step, "Step the loop");
}