
#include <torch/extension.h>

namespace py = pybind11;

#ifndef TINYRL_DTYPE
#define TINYRL_DTYPE float
#endif

#ifndef TINYRL_EPISODE_STEP_LIMIT
#define TINYRL_EPISODE_STEP_LIMIT 200
#endif

//#define RL_TOOLS_BACKEND_DISABLE_BLAS

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>

#include "operations_cpu.h"
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>


#ifdef TINYRL_USE_PPO
#include "ppo.h"
#else
#include "sac.h"
#endif

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using TI = typename DEVICE::index_t;

using T = TINYRL_DTYPE;
static constexpr TI OBSERVATION_DIM = TINYRL_OBSERVATION_DIM;
static constexpr TI ACTION_DIM = TINYRL_ACTION_DIM;
static constexpr TI EPISODE_STEP_LIMIT = TINYRL_EPISODE_STEP_LIMIT;


using ENVIRONMENT_SPEC = PythonEnvironmentSpecification<T, TI, OBSERVATION_DIM, ACTION_DIM>;
using ENVIRONMENT = PythonEnvironment<ENVIRONMENT_SPEC>;


#ifdef TINYRL_USE_PPO
using LOOP_CORE_CONFIG = PPO_LOOP_CORE_CONFIG<T, TI, RNG, ENVIRONMENT, EPISODE_STEP_LIMIT>;
#else
using LOOP_CORE_CONFIG = SAC_LOOP_CORE_CONFIG<T, TI, RNG, ENVIRONMENT, EPISODE_STEP_LIMIT>;
#endif


template <typename NEXT>
struct LOOP_EVAL_PARAMETERS: rlt::rl::loop::steps::evaluation::Parameters<T, TI, NEXT>{
    static constexpr TI EVALUATION_INTERVAL = 1000;
    static constexpr TI NUM_EVALUATION_EPISODES = 10;
    static constexpr TI N_EVALUATIONS = NEXT::CORE_PARAMETERS::STEP_LIMIT / EVALUATION_INTERVAL;
};

DEVICE device;

#ifdef TINYRL_ENABLE_EVALUATION
using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CORE_CONFIG, LOOP_EVAL_PARAMETERS<LOOP_CORE_CONFIG>>;
#else
using LOOP_EVAL_CONFIG = LOOP_CORE_CONFIG;
#endif
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_EVAL_CONFIG>;
using LOOP_CONFIG = LOOP_TIMING_CONFIG;
using LOOP_STATE = typename LOOP_CONFIG::template State<LOOP_CONFIG>;


struct State: LOOP_STATE{
    State(TI seed, std::function<py::object()> p_environment_factory){
        std::cout << "Environment Observation Dim: " << OBSERVATION_DIM << std::endl;
        std::cout << "Environment Action Dim: " << ACTION_DIM << std::endl;
        environment_factory = p_environment_factory;
        rlt::malloc(device, static_cast<LOOP_STATE&>(*this));
        rlt::init(device, static_cast<LOOP_STATE&>(*this), seed);
#ifdef TINYRL_USE_PPO
        state.actor_optimizer.parameters.alpha = 1e-3;
        state.critic_optimizer.parameters.alpha = 1e-3;
#endif
    }
    bool step(){
        return rlt::step(device, static_cast<LOOP_STATE&>(*this));
    }
    ~State(){
        rlt::free(device, static_cast<LOOP_STATE&>(*this));
        environment_factory = nullptr;
    }
};



#ifdef TINYRL_USE_PPO
PYBIND11_MODULE(tinyrl_ppo, m) {
#else
PYBIND11_MODULE(tinyrl_sac, m) {
#endif
    m.doc() = "Python Environment Wrapper";

    py::class_<State>(m, "State")
            .def(py::init<TI, std::function<py::object()>>())
            .def("step", &State::step, "Step the loop");
}