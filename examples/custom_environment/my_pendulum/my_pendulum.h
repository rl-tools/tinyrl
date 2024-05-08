#include <rl_tools/rl/environments/environments.h>
template <typename T>
struct MyPendulumParameters {
    constexpr static T G = 10;
    constexpr static T MAX_SPEED = 8;
    constexpr static T MAX_TORQUE = 2;
    constexpr static T DT = 0.05;
    constexpr static T M = 1;
    constexpr static T L = 1;
    constexpr static T INITIAL_STATE_MIN_ANGLE = -rl_tools::math::PI<T>;
    constexpr static T INITIAL_STATE_MAX_ANGLE = rl_tools::math::PI<T>;
    constexpr static T INITIAL_STATE_MIN_SPEED = -1;
    constexpr static T INITIAL_STATE_MAX_SPEED = 1;
};

template <typename T_T, typename T_TI, typename T_PARAMETERS = MyPendulumParameters<T_T>>
struct MyPendulumSpecification{
    using T = T_T;
    using TI = T_TI;
    using PARAMETERS = T_PARAMETERS;
};

template <typename T, typename TI>
struct MyPendulumState{
    static constexpr TI DIM = 2;
    T theta;
    T theta_dot;
};

template <typename T_SPEC>
struct MyPendulum: rl_tools::rl::environments::Environment{
    using SPEC = T_SPEC;
    using T = typename SPEC::T;
    using TI = typename SPEC::TI;
    using State = MyPendulumState<T, TI>;
    static constexpr TI OBSERVATION_DIM = 3;
    static constexpr TI OBSERVATION_DIM_PRIVILEGED = 0;
    static constexpr TI ACTION_DIM = 1;
    static constexpr TI EPISODE_STEP_LIMIT = 200;
};