# Myelin

A library to do deep learning with spiking neural networks. 

## Getting Started

The primary dependencies of this project are torch, tensorflow and OpenAI gym.
A more comprehensive list of dependencies can be found in ```myelin.yml```.

## Examples

The directory [myelin/task](myelin/task) contains three example experiments.
You can execute them by invoking the run_*.py scripts from this directory.

- To train a MNIST classification network, invoke
    ```
    python run_mnist.py
    ```
- To train a CIFAR classification network, invoke
    ```
    python run_cifar.py
    ```
- To train the cartpole balancing task with Policy gradient, invoke
    ```
    python run_gym.py
    ```
The default choices of hyperparameters are meant as reasonable starting points,
they can be improved upon.

## License

LGPLv3. See [LICENSE](LICENSE) for license details.