#include <torch/custom_class.h>
#include <torch/library.h>
#include <torch/torch.h>

torch::Tensor heaviside(torch::Tensor input) {
  return (input > 0).type_as(input);
}

class SuperFunction : public torch::autograd::Function<SuperFunction> {
public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor input, torch::Tensor alpha) {
    ctx->save_for_backward({input, alpha});
    return heaviside(input);
  }
  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto alpha = saved[1];
    auto grad_output = grad_outputs[0];
    return {grad_output / (alpha * torch::abs(input) + 1.0).pow(2),
            torch::Tensor()};
  }
};

torch::Tensor superfun(torch::Tensor input, torch::Tensor alpha) {
  return SuperFunction::apply(input, alpha);
}

template <torch::Tensor f(torch::Tensor, torch::Tensor)>
auto lif_step(torch::Tensor input_tensor,
              std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> s,
              torch::Tensor input_weights, torch::Tensor recurrent_weights,
              std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                         torch::Tensor, torch::Tensor, torch::Tensor>
                  p,
              double dt)
    -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> {
  auto [z, v, i] = s;
  auto [v_leak, v_th, v_reset, tau_mem_inv, tau_syn_inv, alpha] = p;
  auto dv = dt * tau_mem_inv * ((v_leak - v) + i);
  auto v_decayed = v + dv;

  // compute current updates
  auto di = -dt * tau_syn_inv * i;
  auto i_decayed = i + di;

  // compute new spikes
  auto z_new = f(v_decayed - v_th, alpha);
  // compute reset
  auto v_new = (1 - z_new) * v_decayed + z_new * v_reset;
  // compute current jumps
  auto i_new =
      (i_decayed + torch::nn::functional::linear(input_tensor, input_weights) +
       torch::nn::functional::linear(z, recurrent_weights));

  return {z_new, v_new, i_new};
}

auto lif_super_step = lif_step<superfun>;

template <torch::Tensor f(torch::Tensor, torch::Tensor)>
auto lif_integral(torch::Tensor input_tensor,
                  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> s,
                  torch::Tensor input_weights, torch::Tensor recurrent_weights,
                  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                             torch::Tensor, torch::Tensor, torch::Tensor>
                      p,
                  double dt)
    -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> {
  auto time_steps = input_tensor.size(0);

  std::vector<torch::Tensor> spikes;

  for (int64_t ts = 0; ts < time_steps; ts++) {
    auto input = input_tensor.index({ts});
    s = lif_step<f>(input, s, input_weights, recurrent_weights, p, dt);
    spikes.push_back(std::get<0>(s));
  }

  return {torch::stack(spikes), std::get<1>(s), std::get<2>(s)};
}

auto lif_super_integral = lif_integral<superfun>;

TORCH_LIBRARY(norse_op, m) {
  m.def("superfun", superfun);
  m.def("lif_super_step", lif_super_step);
  m.def("lif_super_integral", lif_super_integral);
}