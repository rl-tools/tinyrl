import onnx
import numpy as np

activation_functions = {
    "Relu": lambda x: np.maximum(x, 0),
    "Identity": lambda x: x,
}

def load_mlp(path):
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)

    graph = onnx_model.graph
    initializers = {initializer.name:initializer for initializer in graph.initializer}
    current_layer = None
    sequential = []
    for node in graph.node:
        if node.op_type == "Gemm":
            assert(current_layer is None)
            attributes = {attr.name: attr for attr in node.attribute}
            assert(len(attributes) == 3)
            assert(attributes["alpha"].f == 1.0)
            assert(attributes["beta"].f == 1.0)
            assert(attributes["transB"].i == 1)
            _, weights_name, biases_name = node.input
            weights = onnx.numpy_helper.to_array(initializers[weights_name])
            biases = onnx.numpy_helper.to_array(initializers[biases_name])
            current_layer = {
                "weights": weights,
                "biases": biases
            }
        elif node.op_type in activation_functions:
            assert(current_layer is not None)
            current_layer["activation"] = node.op_type
            sequential.append(current_layer)
            current_layer = None
        else:
            raise NotImplementedError("Layer Type Not Implemented")
    assert(current_layer is not None)
    current_layer["activation"] = "Identity"
    sequential.append(current_layer)
    return sequential

def evaluate(model, input):
    for layer in model:
        input = np.dot(input, layer["weights"].T) + layer["biases"]
        input = activation_functions[layer["activation"]](input)
    return input