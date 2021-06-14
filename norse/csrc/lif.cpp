#include <torch/torch.h>
#include "super.h"

template <torch::Tensor f(torch::Tensor, torch::Tensor)>
auto lif_feedforward_step(torch::Tensor input_tensor,
                          std::tuple<torch::Tensor, torch::Tensor> s,
                          std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                                     torch::Tensor, torch::Tensor, torch::Tensor>
                              p,
                          double dt)
    -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
{
  auto [v, i] = s;
  auto [tau_syn_inv, tau_mem_inv, v_leak, v_th, v_reset, alpha] = p;
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
  auto i_new = (i_decayed + input_tensor);

  return {z_new, v_new, i_new};
}

template <torch::Tensor f(torch::Tensor, torch::Tensor)>
auto lif_feedforward_step(torch::Tensor input_tensor,
                          std::tuple<torch::Tensor, torch::Tensor> s,
                          std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                                     torch::Tensor, torch::Tensor, torch::Tensor>
                              p,
                          double dt)
    -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;

auto lif_super_feedforward_step = lif_feedforward_step<superfun>;

template <torch::Tensor f(torch::Tensor, torch::Tensor)>
auto lif_feedforward_integral(torch::Tensor input_tensor,
                              std::tuple<torch::Tensor, torch::Tensor> s,
                              std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                                         torch::Tensor, torch::Tensor, torch::Tensor>
                                  p,
                              double dt)
    -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
{
  auto time_steps = input_tensor.size(0);

  std::vector<torch::Tensor> spikes;
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> new_s;

  for (int64_t ts = 0; ts < time_steps; ts++)
  {
    auto input = input_tensor.index({ts});
    new_s = lif_feedforward_step<f>(input, s, p, dt);
    spikes.push_back(std::get<0>(new_s));
  }

  return {torch::stack(spikes), std::get<1>(new_s), std::get<2>(new_s)};
}

auto lif_super_feedforward_integral = lif_feedforward_integral<superfun>;

template <torch::Tensor f(torch::Tensor, torch::Tensor)>
auto lif_step(torch::Tensor input_tensor,
              std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> s,
              torch::Tensor input_weights, torch::Tensor recurrent_weights,
              std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                         torch::Tensor, torch::Tensor, torch::Tensor>
                  p,
              double dt)
    -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
{
  auto [z, v, i] = s;
  auto [tau_syn_inv, tau_mem_inv, v_leak, v_th, v_reset, alpha] = p;
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
    -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
{
  auto time_steps = input_tensor.size(0);

  std::vector<torch::Tensor> spikes;

  for (int64_t ts = 0; ts < time_steps; ts++)
  {
    auto input = input_tensor.index({ts});
    s = lif_step<f>(input, s, input_weights, recurrent_weights, p, dt);
    spikes.push_back(std::get<0>(s));
  }

  return {torch::stack(spikes), std::get<1>(s), std::get<2>(s)};
}

auto lif_super_integral = lif_integral<superfun>;
