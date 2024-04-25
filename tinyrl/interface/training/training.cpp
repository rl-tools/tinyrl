#ifndef TINYRL_MODULE_NAME
#error "TINYRL_MODULE_NAME not defined"
#endif

#ifndef TINYRL_DTYPE
#define TINYRL_DTYPE float
#endif

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>

#ifdef TINYRL_USE_PYTHON_ENVIRONMENT
#include "../python_environment/operations_cpu.h"
#else
#include <environment.h>
#endif

#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn/parameters/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/mlp/persist_code.h>
// #include <rl_tools/nn_models/mlp/persist_code.h>

#include "loop_core_config.h"

namespace rlt = rl_tools;

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


namespace TINYRL_MODULE_NAME{
    using DEVICE = rlt::devices::DEVICE_FACTORY<>;
#ifdef TINYRL_FORCE_BLAS
    static_assert(DEVICE::DEVICE_ID == rlt::devices::DeviceId::CPU_MKL || DEVICE::DEVICE_ID == rlt::devices::DeviceId::CPU_ACCELERATE || DEVICE::DEVICE_ID == rlt::devices::DeviceId::CPU_OPENBLAS);
#endif
    using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
    using TI = typename DEVICE::index_t;

    using T = TINYRL_DTYPE;


    #ifdef TINYRL_USE_PYTHON_ENVIRONMENT
    constexpr TI OBSERVATION_DIM = TINYRL_OBSERVATION_DIM;
    constexpr TI ACTION_DIM = TINYRL_ACTION_DIM;
    using ENVIRONMENT_SPEC = PythonEnvironmentSpecification<T, TI, OBSERVATION_DIM, ACTION_DIM>;
    using ENVIRONMENT = PythonEnvironment<ENVIRONMENT_SPEC>;
    #else
    using ENVIRONMENT = ENVIRONMENT_FACTORY<T, TI>;
    #endif

    #ifdef TINYRL_EPISODE_STEP_LIMIT
    constexpr TI EPISODE_STEP_LIMIT = TINYRL_EPISODE_STEP_LIMIT;
    #else
    constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
    #endif



    // #ifdef TINYRL_USE_PPO
    // using LOOP_CORE_CONFIG = PPO_LOOP_CORE_CONFIG<T, TI, RNG, ENVIRONMENT, EPISODE_STEP_LIMIT>;
    // #else
    using LOOP_CORE_CONFIG = LOOP_CORE_CONFIG_FACTORY<T, TI, RNG, ENVIRONMENT, EPISODE_STEP_LIMIT>;
    // #endif



    DEVICE device;

    #ifdef TINYRL_ENABLE_EVALUATION
    constexpr bool ENABLE_EVALUATION = true;
    #ifndef TINYRL_EVALUATION_INTERVAL
    #error "TINYRL_EVALUATION_INTERVAL not defined"
    #else
    constexpr TI PARAMETER_EVALUATION_INTERVAL = TINYRL_EVALUATION_INTERVAL;
    constexpr TI PARAMETER_NUM_EVALUATION_EPISODES = TINYRL_NUM_EVALUATION_EPISODES;

    template <typename NEXT>
    struct LOOP_EVAL_PARAMETERS: rlt::rl::loop::steps::evaluation::Parameters<T, TI, NEXT>{
        static constexpr TI EVALUATION_INTERVAL = PARAMETER_EVALUATION_INTERVAL;
        static constexpr TI NUM_EVALUATION_EPISODES = PARAMETER_NUM_EVALUATION_EPISODES;
        static constexpr TI N_EVALUATIONS = NEXT::CORE_PARAMETERS::STEP_LIMIT / EVALUATION_INTERVAL;
    };
    using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CORE_CONFIG, LOOP_EVAL_PARAMETERS<LOOP_CORE_CONFIG>>;
    #endif
    #else
    constexpr bool ENABLE_EVALUATION = false;
    using LOOP_EVAL_CONFIG = LOOP_CORE_CONFIG;
    #endif
    using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_EVAL_CONFIG>;
    using LOOP_CONFIG = LOOP_TIMING_CONFIG;
    using LOOP_STATE = typename LOOP_CONFIG::template State<LOOP_CONFIG>;

    #ifdef TINYRL_USE_PYTHON_ENVIRONMENT
    void set_environment_factory(std::function<pybind11::object()> p_environment_factory){
        environment_factory = p_environment_factory;
        auto python_atexit = pybind11::module_::import("atexit");
        python_atexit.attr("register")(pybind11::cpp_function([]() {
            environment_factory = nullptr;
        }));
    }
    #endif
    struct State: LOOP_STATE{
        State(TI seed){
            rlt::malloc(device, static_cast<LOOP_STATE&>(*this));
            rlt::init(device, static_cast<LOOP_STATE&>(*this), seed);
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
            if(num_elements != ENVIRONMENT::OBSERVATION_DIM){
                throw std::runtime_error("Incompatible observation dimension. Check the dimension of the observation returned by env.step() and the one configured when building the TinyRL interface");
            }
            rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation_rlt;
            rlt::malloc(device, observation_rlt);
            for(TI observation_i=0; observation_i<num_elements; observation_i++){
                rlt::set(observation_rlt, 0, observation_i, observation_data_ptr[observation_i]);
            }
            using ACTOR_TYPE = rlt::utils::typing::remove_reference<decltype(rlt::get_actor(*this))>::type;
            rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, ACTOR_TYPE::OUTPUT_DIM>> action_distribution; //2x for mean and std
            rlt::malloc(device, action_distribution);
            bool rng = false;
            rlt::evaluate(device, rlt::get_actor(*this), observation_rlt, action_distribution, this->actor_deterministic_evaluation_buffers, rng);
            rlt::free(device, observation_rlt);

            auto action_rlt = rlt::view(device, action_distribution, rlt::matrix::ViewSpec<1, ENVIRONMENT::ACTION_DIM>{});

            std::vector<T> action(ENVIRONMENT::ACTION_DIM);

            for (TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                action[action_i] = rlt::get(action_rlt, 0, action_i);
            }

            return pybind11::array_t<T>(ENVIRONMENT::ACTION_DIM, action.data());
        }
        std::string export_policy(){
            return rlt::save_code(device, rlt::get_actor(*this), "policy");
        }
        ~State(){
            rlt::free(device, static_cast<LOOP_STATE&>(*this));
        }
    };
}




PYBIND11_MODULE(TINYRL_MODULE_NAME, m){
    m.doc() = "TinyRL Training Loop";
    pybind11::class_<TINYRL_MODULE_NAME::State>(m, "State")
            .def(pybind11::init<TINYRL_MODULE_NAME::TI>())
            .def("step", &TINYRL_MODULE_NAME::State::step, "Step the loop")
            .def("action", &TINYRL_MODULE_NAME::State::action, "Get the action for the given observation")
            .def("export_policy", &TINYRL_MODULE_NAME::State::export_policy, "Export the policy to a python file");
#ifdef TINYRL_USE_PYTHON_ENVIRONMENT
    m.def("set_environment_factory", &TINYRL_MODULE_NAME::set_environment_factory, "Set the environment factory");
#endif
}