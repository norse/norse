#include <gtest/gtest.h>
#include <torch/torch.h>
#include "super.h"
#include "lif.cpp"

auto p = std::make_tuple(torch::full({1}, 1.0 / 5e-3), torch::full({1}, 1.0 / 1e-2),
                         torch::zeros({1}), torch::ones({1}),
                         torch::zeros({1}), torch::full({1}, 100.0));
double dt = 0.001;

// LIF FF
TEST(LIF_TEST, LIF_FF)
{
  auto t = torch::ones({4, 1});
  auto s = std::make_tuple(torch::ones({1}), torch::ones({1}));
  auto [z, v, i] = lif_feedforward_step<superfun>(t, s, p, dt);
  EXPECT_TRUE(torch::allclose(torch::zeros({4, 1}), z));

  s = std::make_tuple(v, i);
  auto [z1, v1, i1] = lif_feedforward_step<superfun>(t, s, p, dt);

  EXPECT_TRUE(torch::allclose(torch::ones({4, 1}), z1));
}
