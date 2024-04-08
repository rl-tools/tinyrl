#include <rl_tools/rl/environments/operations_generic.h>
// helper functions
template <typename T>
T clip(T x, T min, T max){
    x = x < min ? min : (x > max ? max : x);
    return x;
}
template <typename DEVICE, typename T>
T f_mod_python(const DEVICE& dev, T a, T b){
    return a - b * rl_tools::math::floor(dev, a / b);
}

template <typename DEVICE, typename T>
T angle_normalize(const DEVICE& dev, T x){
    return f_mod_python(dev, (x + rl_tools::math::PI<T>), (2 * rl_tools::math::PI<T>)) - rl_tools::math::PI<T>;
}


namespace rl_tools{
    template<typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_state(DEVICE& device, const MyPendulum<SPEC>& env, typename MyPendulum<SPEC>::State& state, RNG& rng){
        state.theta     = random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), SPEC::PARAMETERS::INITIAL_STATE_MIN_ANGLE, SPEC::PARAMETERS::INITIAL_STATE_MAX_ANGLE, rng);
        state.theta_dot = random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), SPEC::PARAMETERS::INITIAL_STATE_MIN_SPEED, SPEC::PARAMETERS::INITIAL_STATE_MAX_SPEED, rng);
    }
    template<typename DEVICE, typename SPEC>
    static void initial_state(DEVICE& device, const MyPendulum<SPEC>& env, typename MyPendulum<SPEC>::State& state){
        state.theta = -rl_tools::math::PI<typename SPEC::T>;
        state.theta_dot = 0;
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    typename SPEC::T step(DEVICE& device, const MyPendulum<SPEC>& env, const typename MyPendulum<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, typename MyPendulum<SPEC>::State& next_state, RNG& rng) {
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == 1);
        using T = typename SPEC::T;
        using PARAMS = typename SPEC::PARAMETERS;
        T u_normalised = get(action, 0, 0);
        T u = PARAMS::MAX_TORQUE * u_normalised;
        T g = PARAMS::G;
        T m = PARAMS::M;
        T l = PARAMS::L;
        T dt = PARAMS::DT;

        u = clip(u, -PARAMS::MAX_TORQUE, PARAMS::MAX_TORQUE);

        T newthdot = state.theta_dot + (3 * g / (2 * l) * rl_tools::math::sin(device.math, state.theta) + 3.0 / (m * l * l) * u) * dt;
        newthdot = clip(newthdot, -PARAMS::MAX_SPEED, PARAMS::MAX_SPEED);
        T newth = state.theta + newthdot * dt;

        next_state.theta = newth;
        next_state.theta_dot = newthdot;
        return SPEC::PARAMETERS::DT;
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    static typename SPEC::T reward(DEVICE& device, const MyPendulum<SPEC>& env, const typename MyPendulum<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, const typename MyPendulum<SPEC>::State& next_state, RNG& rng){
        using T = typename SPEC::T;
        T angle_norm = angle_normalize(device.math, state.theta);
        T u_normalised = get(action, 0, 0);
        T u = SPEC::PARAMETERS::MAX_TORQUE * u_normalised;
        T costs = angle_norm * angle_norm + 0.1 * state.theta_dot * state.theta_dot + 0.001 * (u * u);
        return -costs;
    }

    template<typename DEVICE, typename SPEC, typename OBS_SPEC, typename RNG>
    static void observe(DEVICE& device, const MyPendulum<SPEC>& env, const typename MyPendulum<SPEC>::State& state, Matrix<OBS_SPEC>& observation, RNG& rng){
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == 3);
        using T = typename SPEC::T;
        set(observation, 0, 0, rl_tools::math::cos(device.math, state.theta));
        set(observation, 0, 1, rl_tools::math::sin(device.math, state.theta));
        set(observation, 0, 2, state.theta_dot);
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    static bool terminated(DEVICE& device, const MyPendulum<SPEC>& env, const typename MyPendulum<SPEC>::State state, RNG& rng){
        using T = typename SPEC::T;
        return false;
    }
}