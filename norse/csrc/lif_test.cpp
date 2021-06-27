#include <gtest/gtest.h>
#include <torch/torch.h>
#include "super.h"
#include "lif.cpp"
#include <iostream>

auto p = std::make_tuple(torch::full({1}, 1.0 / 5e-3),
                         torch::full({1}, 1.0 / 1e-2),
                         torch::zeros({1}),
                         torch::ones({1}),
                         torch::zeros({1}), "", torch::full({1}, 100.0));
double dt = 0.001;

// LIF FF
TEST(LIF_TEST, LIF_FF)
{
  auto t = torch::ones({4, 1});
  auto s = std::make_tuple(torch::ones({4, 1}), torch::ones({4, 1}));
  auto [z, v, i] = lif_feed_forward_step<superfun>(t, s, p, dt);
  EXPECT_TRUE(torch::allclose(torch::zeros({4, 1}), z));

  s = std::make_tuple(v, i);
  auto [z1, v1, i1] = lif_feed_forward_step<superfun>(t, s, p, dt);

  EXPECT_TRUE(torch::allclose(torch::ones({4, 1}), z1));
}
// LIF FF Integral
TEST(LIF_TEST, LIF_FF_INTEGRAL)
{
  auto t = torch::ones({10, 4, 1});
  auto s = std::make_tuple(torch::zeros({4, 1}), torch::zeros({4, 1}));
  auto [z, v, i] = lif_feed_forward_integral<superfun>(t, s, p, dt);
  EXPECT_FALSE(torch::allclose(torch::zeros({10, 4, 1}), z));
}

// LIF recurrent
TEST(LIF_TEST, LIF_REC)
{
  auto t = torch::ones({2, 5});
  auto s = std::make_tuple(torch::zeros({2, 4}), torch::zeros({2, 4}), torch::zeros({2, 4}));
  auto input_weights = torch::linspace(0, 1, 20).view({4, 5});
  auto recurrent_weights = torch::linspace(0, 2, 16).view({4, 4});
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  s = lif_step<superfun>(t, s, input_weights, recurrent_weights, p, dt);
  EXPECT_TRUE(torch::allclose(torch::zeros({2, 4}), std::get<0>(s)));
  EXPECT_TRUE(torch::allclose(torch::zeros({2, 4}, options), std::get<1>(s)));
  EXPECT_FALSE(torch::allclose(torch::zeros({2, 4}, options), std::get<2>(s)));

  s = lif_step<superfun>(t, s, input_weights, recurrent_weights, p, dt);
  s = lif_step<superfun>(t, s, input_weights, recurrent_weights, p, dt);
  auto expected = torch::tensor({{0, 0, 0, 1}, {0, 0, 0, 1}}, options);
  EXPECT_TRUE(torch::allclose(expected, std::get<0>(s)));
  EXPECT_FALSE(torch::allclose(torch::zeros({2, 4}, options), std::get<1>(s)));
  EXPECT_FALSE(torch::allclose(torch::zeros({2, 4}, options), std::get<2>(s)));
}
// LIF recurrent Integral
TEST(LIF_TEST, LIF_REC_INTEGRAL)
{
  auto t = torch::ones({3, 2, 5});
  auto s = std::make_tuple(torch::zeros({2, 4}), torch::zeros({2, 4}), torch::zeros({2, 4}));
  auto input_weights = torch::linspace(0, 1, 20).view({4, 5});
  auto recurrent_weights = torch::linspace(0, 2, 16).view({4, 4});
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  s = lif_integral<superfun>(t, s, input_weights, recurrent_weights, p, dt);
  auto expected = torch::tensor({{{0, 0, 0, 0}, {0, 0, 0, 0}},
                                 {{0, 0, 0, 0}, {0, 0, 0, 0}},
                                 {{0, 0, 0, 1}, {0, 0, 0, 1}}},
                                options);
  EXPECT_TRUE(torch::allclose(expected, std::get<0>(s)));
  EXPECT_FALSE(torch::allclose(torch::zeros({2, 4}, options), std::get<1>(s)));
  EXPECT_FALSE(torch::allclose(torch::zeros({2, 4}, options), std::get<2>(s)));
}
