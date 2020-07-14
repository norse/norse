#include <torch/extension.h>


torch::Tensor identity(torch::Tensor x) {
    return x;
}

PYBIND11_MODULE(norse, m) {
    m.def("idenity", &identity, "welford mean variance");
}
