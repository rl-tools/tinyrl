from string import Template
import struct

file_template = Template("""
#include <rl_tools/containers.h>
#include <rl_tools/nn/layers/dense/layer.h>
#include <rl_tools/nn_models/sequential/model.h>
#include <rl_tools/utils/generic/typing.h>
$MODEL
""")


activation_functions = {
    "Relu": "RELU",
    "Identity": "IDENTITY"
}


def model_definition(N):
    # Type example: Module<layer_0::TYPE, Module<layer_1::TYPE, Module<layer_2::TYPE>>>
    # Instance example: {layer_0::layer, {layer_1::layer, {layer_2::layer}}}
    rendered_type = ""
    rendered_instance = ""
    for i in range(N):
        rendered_type += f"IF::Module<layer_{i}::TEMPLATE" + (", " if i < N - 1 else "")
        rendered_instance += f"{{layer_{i}::layer" + (", " if i < N - 1 else "")
    for i in range(N):
        rendered_type += ">"
        rendered_instance += "}"
    return rendered_type, rendered_instance


model_template = Template("""
namespace policy{
    $LAYERS
    namespace model_definition {
        using CAPABILITY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layer_capability::Forward; 
        using IF = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn_models::sequential::Interface<CAPABILITY>;
        using MODEL = $MODEL_TYPE;
    }
    using MODEL = model_definition::MODEL;
    const MODEL model = $MODEL_INSTANCE;
}
""")

layer_template = Template("""
namespace layer_$LAYER_ID{
    namespace weights{
        namespace parameters_memory {
            static_assert(sizeof(unsigned char) == 1);
            alignas($T) const unsigned char memory[] = {$WEIGHTS};
            using CONTAINER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::Specification<$T, $TI, $OUTPUT_DIM, $INPUT_DIM, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::layouts::RowMajorAlignment<$TI, 1>>;
            using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::MatrixDynamic<CONTAINER_SPEC>;
            const CONTAINER_TYPE container = {($T*)memory}; 
        }
    }
    namespace weights {
        using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::spec<parameters_memory::CONTAINER_TYPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Weights>;
        const rl_tools::nn::parameters::Plain::instance<PARAMETER_SPEC> parameters = {parameters_memory::container};
    }
    namespace biases {
        namespace parameters_memory {
            static_assert(sizeof(unsigned char) == 1);
            alignas($T) const unsigned char memory[] = {$BIASES};
            using CONTAINER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::Specification<$T, $TI, 1, $OUTPUT_DIM, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::layouts::RowMajorAlignment<$TI, 1>>;
            using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::MatrixDynamic<CONTAINER_SPEC>;
            const CONTAINER_TYPE container = {($T*)memory}; 
        }
    }
    namespace biases {
        using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::spec<parameters_memory::CONTAINER_TYPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Biases>;
        const rl_tools::nn::parameters::Plain::instance<PARAMETER_SPEC> parameters = {parameters_memory::container};
    }
    using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::Specification<$T, $TI, $INPUT_DIM, $OUTPUT_DIM, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::activation_functions::ActivationFunction::$ACTIVATION_FUNCTION, 1, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::MatrixDynamicTag, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::layouts::RowMajorAlignment<$TI, 1>>; 
    template <typename CAPABILITY>
    using TEMPLATE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::Layer<CAPABILITY, SPEC>;
    using CAPABILITY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layer_capability::Forward;
    using TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::Layer<CAPABILITY, SPEC>;
    const TYPE layer = {weights::parameters, biases::parameters};
}
""")


def convert_float_to_uint8(value):
    bytes_value = struct.pack('f', value)
    uint8_values = struct.unpack('4B', bytes_value)
    return uint8_values


def indent(text, spaces=4):
    return "\n".join([" " * spaces + line for line in text.split("\n")])


def render(model, dtype="float", itype="unsigned long"):
    layers = []
    for i, layer in enumerate(model):
        weights = ",".join([str(element) for tupl in map(convert_float_to_uint8, layer["weights"].flatten().tolist()) for element in tupl])
        biases = ",".join([str(element) for tupl in map(convert_float_to_uint8, layer["biases"].flatten().tolist()) for element in tupl])
        rendered_layer = layer_template.substitute(
            LAYER_ID=i,
            T=dtype,
            TI=itype,
            INPUT_DIM=layer["weights"].shape[1],
            OUTPUT_DIM=layer["weights"].shape[0],
            WEIGHTS=weights,
            BIASES=biases,
            ACTIVATION_FUNCTION=activation_functions[layer["activation"]]
        )
        layers.append(rendered_layer)
    rendered_layers = "\n".join(layers)
    model_type, model_instance = model_definition(len(model))
    rendered_model = model_template.substitute(
        LAYERS=indent(rendered_layers),
        MODEL_TYPE=model_type,
        MODEL_INSTANCE=model_instance
    )
    return file_template.substitute(MODEL=indent(rendered_model))