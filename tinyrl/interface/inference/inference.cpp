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

pybind11::array_t<T> evaluate(const pybind11::array_t<T>& input){
    init();
    pybind11::buffer_info input_info = input.request();
    if (input_info.format != pybind11::format_descriptor<T>::format() || input_info.ndim != 1) {
        throw std::runtime_error("Incompatible buffer format. Check the floating point type of the input and the one configured when building the TinyRL interface");
    }
    auto input_data_ptr = static_cast<T*>(input_info.ptr);
    size_t num_elements = input_info.shape[0];
    if(num_elements != MODEL_TYPE::INPUT_DIM){
        throw std::runtime_error("Incompatible input dimension. Check the dimension of the input and the one configured when building the TinyRL interface");
    }
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, MODEL_TYPE::INPUT_DIM>> input_rlt;
    rlt::malloc(device, input_rlt);
    for(TI input_i=0; input_i<num_elements; input_i++){
        rlt::set(input_rlt, 0, input_i, input_data_ptr[input_i]);
    }
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, MODEL_TYPE::OUTPUT_DIM>> output_rlt;
    rlt::malloc(device, output_rlt);
    bool rng = false;
    rlt::evaluate(device, policy::model, input_rlt, output_rlt, buffer, rng);

    std::vector<T> output(MODEL_TYPE::OUTPUT_DIM);

    for (TI output_i = 0; output_i < MODEL_TYPE::OUTPUT_DIM; output_i++) {
        output[output_i] = rlt::get(output_rlt, 0, output_i);
    }

    rlt::free(device, input_rlt);
    rlt::free(device, output_rlt);

    return pybind11::array_t<T>(MODEL_TYPE::OUTPUT_DIM, output.data());
}

PYBIND11_MODULE(TINYRL_MODULE_NAME, m){
    m.doc() = "TinyRL Inference";
    m.def("evaluate", &evaluate, "Evaluate the NN");
}