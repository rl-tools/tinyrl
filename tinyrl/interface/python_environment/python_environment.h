#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


template <typename T_T, typename T_TI, T_TI T_OBSERVATION_DIM, T_TI T_ACTION_DIM>
struct PythonEnvironmentSpecification{
    using T = T_T;
    using TI = T_TI;
    static constexpr TI OBSERVATION_DIM = T_OBSERVATION_DIM;
    static constexpr TI ACTION_DIM = T_ACTION_DIM;
};

template <typename T_T, typename T_TI, T_TI T_DIM>
struct PythonEnvironmentState{
    using T = T_T;
    using TI = T_TI;
    static constexpr TI DIM = T_DIM;
    std::array<T, DIM> state;
    T reward;
    bool terminated;
};

template <typename T_SPEC>
struct PythonEnvironment{
    using SPEC = T_SPEC;
    using T = typename SPEC::T;
    using TI = typename SPEC::TI;
    using State = PythonEnvironmentState<T, TI, SPEC::OBSERVATION_DIM>;
    static constexpr TI OBSERVATION_DIM = SPEC::OBSERVATION_DIM;
    static constexpr TI OBSERVATION_DIM_PRIVILEGED = SPEC::OBSERVATION_DIM;
    static constexpr TI ACTION_DIM = SPEC::ACTION_DIM;
    pybind11::object* environment;
    // T action_space_range[ACTION_DIM];
    // T action_space_offset[ACTION_DIM];
};