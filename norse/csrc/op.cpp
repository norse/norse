#include <torch/library.h>
#include <torch/torch.h>

#include "norse/integrators/rk4.hpp"
#include "norse/integrators/dopri5.hpp"


torch::Tensor heaviside(torch::Tensor input) {
  return (input > 0).type_as(input);
}

class SuperFunction : public torch::autograd::Function<SuperFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor input, torch::Tensor alpha) {
      ctx->save_for_backward({input, alpha});
      return heaviside(input);
    }
    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto alpha = saved[1];
        auto grad_output = grad_outputs[0];
        return {grad_output / (alpha * torch::abs(input) + 1.0).pow(
            2
        ), torch::Tensor()};
    }
};

torch::Tensor superfun(torch::Tensor input, torch::Tensor alpha) {
    return SuperFunction::apply(input, alpha);
}

torch::Tensor lif_step(torch::Tensor input) {

}







class LinearFunction : public torch::autograd::Function<LinearFunction> {
 public:
  // Note that both forward and backward are static functions

  // bias is an optional argument
  static torch::Tensor forward(
      torch::autograd::AutogradContext *ctx, torch::Tensor input, torch::Tensor weight, torch::Tensor bias = torch::Tensor()) {
    ctx->save_for_backward({input, weight, bias});
    auto output = input.mm(weight.t());
    if (bias.defined()) {
      output += bias.unsqueeze(0).expand_as(output);
    }
    return output;
  }

  static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto bias = saved[2];

    auto grad_output = grad_outputs[0];
    auto grad_input = grad_output.mm(weight);
    auto grad_weight = grad_output.t().mm(input);
    auto grad_bias = torch::Tensor();
    if (bias.defined()) {
      grad_bias = grad_output.sum(0);
    }

    return {grad_input, grad_weight, grad_bias};
  }
};


torch::Tensor mlinear(torch::Tensor input, torch::Tensor weight, torch::Tensor bias = torch::Tensor()) {
    return LinearFunction::apply(input, weight, bias);
}


torch::Tensor identity(torch::Tensor x) {
    return x;
}

TORCH_LIBRARY(norse_op, m) {
  m.def("superfun", superfun);
  m.def("identity", identity);
  m.def("linear", mlinear);
}