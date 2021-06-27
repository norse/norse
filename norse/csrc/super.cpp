#include <torch/custom_class.h>
#include <torch/torch.h>

#include "super.h"

torch::Tensor heaviside(torch::Tensor input)
{
  return (input > 0).type_as(input);
}

class SuperFunction : public torch::autograd::Function<SuperFunction>
{
public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor input, torch::Tensor alpha)
  {
    ctx->save_for_backward({input, alpha});
    return heaviside(input);
  }
  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs)
  {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto alpha = saved[1];
    auto grad_output = grad_outputs[0];
    return {grad_output / (alpha * torch::abs(input) + 1.0).pow(2),
            torch::Tensor()};
  }
};

torch::Tensor superfun(torch::Tensor input, torch::Tensor alpha)
{
  return SuperFunction::apply(input, alpha);
}