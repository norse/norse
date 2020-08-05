.. _page-hardware:

Hardware acceleration
-------------------

Norse is built on top of `PyTorch <https://pytorch.org>`_ which has excellent support for hardware acceleration. 
This article will cover how Norse can be accelerated with GPUs through the `CUDA platform <https://en.wikipedia.org/wiki/CUDA>`_.

Since Norse is using PyTorch primitives, it is worth familiarising yourself with the use of GPUs in PyTorch.
Here is a tutorial describing how to use GPUs with the ``to`` method in PyTorch: 
https://pytorch.org/tutorials/beginner/nn_tutorial.html#using-your-gpu

Accelerating Norse models with ``.to``
======================================

To accelerate neuron models in Norse, one simply has to call the ``.to`` method on models and data:


.. code:: python

    from norse.torch.module.lif import LIFCell
    LIFCell(10, 20).to('cuda')

.. code:: python

    import torch
    torch.randn(100, 20).to('cuda')

Accelerating Norse nested models
======================================

It might be necessary to sometimes build your own nested `torch.nn` modules. 
In that case it's recommended to propagate a ``device`` parameter to inner modules, to ensure that data is 
moved to the GPU as soon as possible.

Here is an example of a nested model that we will move to the GPU
(taken and simplified from our `MNIST task example <https://github.com/norse/norse/blob/master/norse/task/mnist.py#L60>`_):

.. code:: python

    class LIFConvNet(torch.nn.Module):
        def __init__(
            self,
            input_features,
            seq_length,
            model="super",
            device="cpu"
        ):
            super(LIFConvNet, self).__init__()
            self.constant_current_encoder = ConstantCurrentLIFEncoder(seq_length=seq_length)
            self.input_features = input_features
            self.rsnn = ConvNet4(method=model, device=device)
            self.seq_length = seq_length

        def forward(self, x):
            batch_size = x.shape[0]
            x = self.constant_current_encoder(
                x.view(-1, self.input_features)
            )
            x = x.reshape(self.seq_length, batch_size, 1, 28, 28)
            voltages = self.rsnn(x)
            m, _ = torch.max(voltages, 0)
            log_p_y = torch.nn.functional.log_softmax(m, dim=1)
            return log_p_y

We can now use this model in a ``main`` by initialising it with the correct device:

.. code:: python

    def main(args):
        device = ...
        input_features = ...
        seq_length = ...
        model = LIFConvNet(input_features, seq_length, device=device).to(device)
        ...