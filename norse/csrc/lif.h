#include <torch/torch.h>

namespace norse
{

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    lif_feedforward_step(torch::Tensor input_tensor,
                         std::tuple<torch::Tensor, torch::Tensor> s,
                         std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                                    torch::Tensor, torch::Tensor, torch::Tensor>
                             p,
                         double dt);

}