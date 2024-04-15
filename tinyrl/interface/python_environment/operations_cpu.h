#include "python_environment.h"

std::function<pybind11::object()> environment_factory;

template <typename SPEC>
void observe_state(pybind11::array observation_array, typename PythonEnvironment<SPEC>::State& state){
    using T = typename SPEC::T;
    // pybind11::array observation_array = observation.template cast<pybind11::array>();
    pybind11::buffer_info observation_info = observation_array.request();
    bool observation_is_float = observation_info.format == pybind11::format_descriptor<float>::format();
    bool observation_is_double = observation_info.format == pybind11::format_descriptor<double>::format();
    if (!observation_is_float && !observation_is_double){
        throw std::runtime_error(std::string("Incompatible buffer format: [") + observation_info.format + std::string("]. Check the floating point type of the observation returned by env.step() and the one configured when building the TinyRL interface"));
    } 
    if(observation_info.ndim != 1){
        throw std::runtime_error("Incompatible buffer format. Check the dimension of the observation returned by env.step() and the one configured when building the TinyRL interface");
    }

    size_t num_elements = observation_info.shape[0];
    if(num_elements != SPEC::OBSERVATION_DIM){
        throw std::runtime_error("Incompatible observation dimension. Check the dimension of the observation returned by env.step() and the one configured when building the TinyRL interface");
    }

    if(observation_info.format == pybind11::format_descriptor<T>::format()){
        auto observation_data_ptr = static_cast<T*>(observation_info.ptr);
        std::copy(observation_data_ptr, observation_data_ptr + num_elements, state.state.begin());
    }
    else{
        auto observation_data_ptr = static_cast<double*>(observation_info.ptr);
        for(size_t i=0; i<SPEC::OBSERVATION_DIM; i++){
            state.state[i] = observation_data_ptr[i];
        }
    }
}

namespace rl_tools{
    template<typename DEVICE, typename SPEC>
    static void malloc(DEVICE& device, PythonEnvironment<SPEC>& env){
        using TI = typename DEVICE::index_t;
        using T = typename SPEC::T;
        env.environment = new pybind11::object();
        *env.environment = environment_factory();
        auto action_space = env.environment->attr("action_space");
        auto low = action_space.attr("low").template cast<pybind11::array_t<T>>();
        auto high = action_space.attr("high").template cast<pybind11::array_t<T>>();
        if(low.ndim() != 1 || high.ndim() != 1){
            throw std::runtime_error("Incompatible action space shape. Check the action space format returned by env.action_space (should be 1D and continuous) and the one configured when building the TinyRL interface");
        }
        if(low.shape(0) != SPEC::ACTION_DIM || high.shape(0) != SPEC::ACTION_DIM){
            throw std::runtime_error("Incompatible action space dimension. Limits: low " + std::to_string(low.shape(0)) + ", high " + std::to_string(high.shape(0)) + " (expected " + std::to_string(SPEC::ACTION_DIM) + "). Check the action space dimension returned by env.action_space and the one configured when building the TinyRL interface");
        }
        for(TI action_i=0; action_i<SPEC::ACTION_DIM; action_i++){
            T current_low = low.at(action_i);
            T current_high = high.at(action_i);
//             env.action_space_offset[action_i] = (current_high + current_low)/2;
//             env.action_space_range[action_i] = (current_high - current_low)/2;
// #ifdef TINYRL_VERBOSE
//             std::cout << "Action dim: " << action_i << " low: " << current_low << " high: " << current_high << std::endl;
// #endif
            if(math::abs(device.math, current_low - (-1.0)) > 1e-6 || math::abs(device.math, current_high - 1.0) > 1e-6){
                throw std::runtime_error("Incompatible action space limits. Limits: low " + std::to_string(current_low) + ", high " + std::to_string(current_high) + " (expected -1.0 and 1.0). You should use a RescaleActionV0(env, -1, 1) or equivalent.");
            };
        }
    }
    template<typename DEVICE, typename SPEC>
    static void init(DEVICE& device, PythonEnvironment<SPEC>& env){}
    template<typename DEVICE, typename SPEC>
    static void initial_state(DEVICE& device, const PythonEnvironment<SPEC>& env, typename PythonEnvironment<SPEC>::State& state){
        using T = typename SPEC::T;

        auto result = env.environment->attr("reset")();
        pybind11::tuple result_tuple = pybind11::cast<pybind11::tuple>(result);
        auto first_element = result_tuple[0];
        pybind11::array observation_array = first_element.cast<pybind11::array>();
        observe_state<SPEC>(observation_array, state);
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

        pybind11::array_t<T> action_array(SPEC::ACTION_DIM);
        pybind11::buffer_info action_info = action_array.request();
        auto action_data_ptr = static_cast<T*>(action_info.ptr);
        for(size_t i=0; i<SPEC::ACTION_DIM; i++){
            action_data_ptr[i] = get(action, 0, i); // * env.action_space_range[i] + env.action_space_offset[i];
        }
        auto result = env.environment->attr("step")(action_array);
        pybind11::tuple result_tuple = pybind11::cast<pybind11::tuple>(result);
        auto observation = result_tuple[0];
        auto reward = result_tuple[1];
        auto terminated = result_tuple[2];
        auto truncated = result_tuple[3];

        pybind11::array observation_array = observation.cast<pybind11::array>();
        observe_state<SPEC>(observation_array, next_state);

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