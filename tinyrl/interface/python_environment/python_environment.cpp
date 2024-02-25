
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

void set_environment_factory(std::function<pybind11::object()> p_environment_factory){
    environment_factory = p_environment_factory;
    auto python_atexit = pybind11::module_::import("atexit");
    python_atexit.attr("register")(pybind11::cpp_function([]() {
        environment_factory = nullptr;
    }));
}

struct State: LOOP_STATE{
    State(TI seed){
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
    pybind11::array_t<T> action(const pybind11::array_t<T>& observation){
        pybind11::buffer_info observation_info = observation.request();
        if (observation_info.format != pybind11::format_descriptor<T>::format() || observation_info.ndim != 1) {
            throw std::runtime_error("Incompatible buffer format. Check the floating point type of the observation returned by env.step() and the one configured when building the TinyRL interface");
        }
        auto observation_data_ptr = static_cast<T*>(observation_info.ptr);
        size_t num_elements = observation_info.shape[0];
        if(num_elements != OBSERVATION_DIM){
            throw std::runtime_error("Incompatible observation dimension. Check the dimension of the observation returned by env.step() and the one configured when building the TinyRL interface");
        }
        rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, OBSERVATION_DIM>> observation_rlt;
        rlt::malloc(device, observation_rlt);
        for(TI observation_i=0; observation_i<num_elements; observation_i++){
            rlt::set(observation_rlt, 0, observation_i, observation_data_ptr[observation_i]);
        }
        rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, 2*ACTION_DIM>> action_distribution; //2x for mean and std
        rlt::malloc(device, action_distribution);
        rlt::evaluate(device, this->actor_critic.actor, observation_rlt, action_distribution, this->actor_deterministic_evaluation_buffers);
        rlt::free(device, observation_rlt);

        auto action_rlt = rlt::view(device, action_distribution, rlt::matrix::ViewSpec<1, ACTION_DIM>{});

        std::vector<T> action(ACTION_DIM);

        for (TI action_i = 0; action_i < ACTION_DIM; action_i++) {
            action[action_i] = rlt::get(action_rlt, 0, action_i);
        }

        return pybind11::array_t<T>(ACTION_DIM, action.data());
    }
    ~State(){
        rlt::free(device, static_cast<LOOP_STATE&>(*this));
    }
};



#ifdef TINYRL_USE_PPO
PYBIND11_MODULE(tinyrl_ppo, m) {
#else
PYBIND11_MODULE(tinyrl_sac, m) {
#endif
    m.doc() = "TinyRL SAC Training Loop";
    pybind11::class_<State>(m, "State")
            .def(pybind11::init<TI>())
            .def("step", &State::step, "Step the loop")
            .def("action", &State::action, "Get the action for the given observation");
    m.def("set_environment_factory", &set_environment_factory, "Set the environment factory");
}