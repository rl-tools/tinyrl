#include "python_environment.h"

#include <pybind11/functional.h>
#include <pybind11/numpy.h>

std::function<py::object()> environment_factory;

namespace rl_tools{
    template<typename DEVICE, typename SPEC>
    static void malloc(DEVICE& device, PythonEnvironment<SPEC>& env){
        env.environment = new py::object();
        *env.environment = environment_factory();
    }
    template<typename DEVICE, typename SPEC>
    static void init(DEVICE& device, PythonEnvironment<SPEC>& env){}
    template<typename DEVICE, typename SPEC>
    static void initial_state(DEVICE& device, const PythonEnvironment<SPEC>& env, typename PythonEnvironment<SPEC>::State& state){
        using T = typename SPEC::T;

        auto result = env.environment->attr("reset")();
        py::tuple result_tuple = py::cast<py::tuple>(result);
        auto first_element = result_tuple[0];
        py::array array = first_element.cast<py::array>();
        py::buffer_info info = array.request();
        if (info.format != py::format_descriptor<T>::format() || info.ndim != 1) {
            throw std::runtime_error("Incompatible buffer format. Check the floating point type of the observation returned by env.reset() and the one configured when building the TinyRL interface");
        }
        auto data_ptr = static_cast<T*>(info.ptr);
        size_t num_elements = info.shape[0];
        if(num_elements != SPEC::OBSERVATION_DIM){
            throw std::runtime_error("Incompatible observation dimension. Check the dimension of the observation returned by env.reset() and the one configured when building the TinyRL interface");
        }
        std::copy(data_ptr, data_ptr + num_elements, state.state.begin());
        state.terminated = false;
        state.reward = 0;
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_state(DEVICE& device, const PythonEnvironment<SPEC>& env, typename PythonEnvironment<SPEC>::State& state, RNG& rng){
        initial_state(device, env, state);
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    typename SPEC::T step(DEVICE& device, const PythonEnvironment<SPEC>& env, const typename PythonEnvironment<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, typename PythonEnvironment<SPEC>::State& next_state, RNG& rng) {
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == SPEC::ACTION_DIM);
        using T = typename SPEC::T;

        py::array_t<T> action_array(SPEC::ACTION_DIM);
        py::buffer_info action_info = action_array.request();
        auto action_data_ptr = static_cast<T*>(action_info.ptr);
        for(size_t i=0; i<SPEC::ACTION_DIM; i++){
            action_data_ptr[i] = get(action, 0, i);
        }
        auto result = env.environment->attr("step")(action_array);
        py::tuple result_tuple = py::cast<py::tuple>(result);
        auto observation = result_tuple[0];
        auto reward = result_tuple[1];
        auto terminated = result_tuple[2];
        auto truncated = result_tuple[3];

        py::array observation_array = observation.cast<py::array>();
        py::buffer_info observation_info = observation_array.request();
        if (observation_info.format != py::format_descriptor<T>::format() || observation_info.ndim != 1) {
            throw std::runtime_error("Incompatible buffer format. Check the floating point type of the observation returned by env.step() and the one configured when building the TinyRL interface");
        }

        auto observation_data_ptr = static_cast<T*>(observation_info.ptr);
        size_t num_elements = observation_info.shape[0];
        if(num_elements != SPEC::OBSERVATION_DIM){
            throw std::runtime_error("Incompatible observation dimension. Check the dimension of the observation returned by env.step() and the one configured when building the TinyRL interface");
        }
        std::copy(observation_data_ptr, observation_data_ptr + num_elements, next_state.state.begin());
        next_state.reward = reward.cast<T>();
        next_state.terminated = terminated.cast<bool>();
        return 0;
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    static typename SPEC::T reward(DEVICE& device, const PythonEnvironment<SPEC>& env, const typename PythonEnvironment<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, const typename PythonEnvironment<SPEC>::State& next_state, RNG& rng){
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == SPEC::ACTION_DIM);
        using T = typename SPEC::T;
        return next_state.reward;
    }

    template<typename DEVICE, typename SPEC, typename OBS_SPEC, typename RNG>
    static void observe(DEVICE& device, const PythonEnvironment<SPEC>& env, const typename PythonEnvironment<SPEC>::State& state, Matrix<OBS_SPEC>& observation, RNG& rng){
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == SPEC::OBSERVATION_DIM);
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        for(TI observation_i=0; observation_i<SPEC::OBSERVATION_DIM; observation_i++){
            set(observation, 0, observation_i, state.state[observation_i]);
        }
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    static bool terminated(DEVICE& device, const PythonEnvironment<SPEC>& env, const typename PythonEnvironment<SPEC>::State state, RNG& rng){
        return state.terminated;
    }
    template<typename DEVICE, typename SPEC>
    void free(DEVICE& device, PythonEnvironment<SPEC>& env){
        delete env.environment;
    }
}