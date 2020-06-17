# Example Tasks

These tasks serves as 1) a means to illustrate how Norse can be used and 2) what spiking neural networks (and Norse) can do in terms of performance. They are designed to be run on semi-beefy machines with dedicated GPUs. 

All tasks have CLI interfaces with elaborate help descriptions using the `--help` flag.

## MNIST

```bash
python -m norse.task.mnist
```

## CIFAR-10
```bash
python -m norse.task.cifar10
```

## Cartpole

```bash
python -m norse.task.cartpole --episodes=1000 --learning_rate=0.005 --device=cuda --weight_scale=1.0`
```

## Correlation experiment

```bash
python -m norse.task.correlation_experiment
```
