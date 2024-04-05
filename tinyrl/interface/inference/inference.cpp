#include <rl_tools/operations/cpu_mux.h>

#include <checkpoint.h>

#include <rl_tools/nn/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace rlt = rl_tools;


using DEVICE = rlt::devices::DEVICE_FACTORY<>;

DEVICE device;
using MODEL_TYPE = decltype(policy::model);
using T = typename MODEL_TYPE::T;
using TI = typename DEVICE::index_t;
typename MODEL_TYPE::template Buffer<1> buffer;
bool initialized = false;

void init(){
    if(!initialized){
        rlt::malloc(device, buffer);
    }
}

pybind11::array_t<T> evaluate(const pybind11::array_t<T>& observation){
    init();
    pybind11::buffer_info observation_info = observation.request();
    if (observation_info.format != pybind11::format_descriptor<T>::format() || observation_info.ndim != 1) {
        throw std::runtime_error("Incompatible buffer format. Check the floating point type of the observation returned by env.step() and the one configured when building the TinyRL interface");
    }
    auto observation_data_ptr = static_cast<T*>(observation_info.ptr);
    size_t num_elements = observation_info.shape[0];
    if(num_elements != MODEL_TYPE::INPUT_DIM){
        throw std::runtime_error("Incompatible observation dimension. Check the dimension of the observation returned by env.step() and the one configured when building the TinyRL interface");
    }
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, MODEL_TYPE::INPUT_DIM>> observation_rlt;
    rlt::malloc(device, observation_rlt);
    for(TI observation_i=0; observation_i<num_elements; observation_i++){
        rlt::set(observation_rlt, 0, observation_i, observation_data_ptr[observation_i]);
    }
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, MODEL_TYPE::OUTPUT_DIM>> action_rlt;
    rlt::malloc(device, action_rlt);
    rlt::evaluate(device, policy::model, observation_rlt, action_rlt, buffer);

    std::vector<T> action(MODEL_TYPE::OUTPUT_DIM);

    for (TI action_i = 0; action_i < MODEL_TYPE::OUTPUT_DIM; action_i++) {
        action[action_i] = rlt::get(action_rlt, 0, action_i);
    }

    rlt::free(device, observation_rlt);
    rlt::free(device, action_rlt);

    return pybind11::array_t<T>(MODEL_TYPE::OUTPUT_DIM, action.data());
}

PYBIND11_MODULE(TINYRL_MODULE_NAME, m){
    m.doc() = "TinyRL Policy Checkpoint";
    m.def("evaluate", &evaluate, "Evaluate the policy");
}