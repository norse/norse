#include <torch/custom_class.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <torch/extension.h>

#include "super.h"
#include "lif.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("superfun", superfun);
  m.def("lif_super_step", lif_super_step);
  m.def("lif_super_integral", lif_super_integral);
}