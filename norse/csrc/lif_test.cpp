#include <torch/torch.h>
#include <gtest/gtest.h>
#include "super.h"
#include "lif.h"

auto p = std::make_tuple(torch::full({1}, 1.0 / 5e-3), torch::full({1}, 1.0 / 1e-2),
                         torch::ones({1}), torch::ones({1}),
                         torch::zeros({1}), torch::full({1}, 100.0));
double dt = 0.001;

// LIF FF
TEST(LIF_FF_TEST, BasicAssertions)
{
  auto t = torch::ones({4, 1});
  auto s = std::make_tuple(torch::ones({1}), torch::ones({1}));
  auto out = norse::lif_feedforward_step<norse::superfun>(t, s, p, dt);
  auto [z, v, i] = out;
}
