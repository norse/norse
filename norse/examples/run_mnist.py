#!/usr/bin/env python

"""
MNIST classification example
============================
"""

import norse.task.mnist as mnist

from absl import app
from absl import flags

flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to use by pytorch.")
flags.DEFINE_integer("epochs", 10, "Number of training episodes to do.")
flags.DEFINE_integer("seq_length", 200, "Number of timesteps to do.")
flags.DEFINE_integer("batch_size", 32, "Number of examples in one minibatch.")
flags.DEFINE_enum(
    "model",
    "super",
    ["super", "tanh", "circ", "logistic", "circ_dist"],
    "Model to use for training.",
)
flags.DEFINE_string("prefix", "", "Prefix to use for saving the results")
flags.DEFINE_enum(
    "optimizer", "adam", ["adam", "sgd"], "Optimizer to use for training."
)
flags.DEFINE_bool("clip_grad", False, "Clip gradient during backpropagation")
flags.DEFINE_float("grad_clip_value", 1.0, "Gradient to clip at.")
flags.DEFINE_float("learning_rate", 2e-3, "Learning rate to use.")
flags.DEFINE_integer(
    "log_interval", 10, "In which intervals to display learning progress."
)
flags.DEFINE_integer("model_save_interval", 50, "Save model every so many epochs.")
flags.DEFINE_boolean("save_model", True, "Save the model after training.")
flags.DEFINE_boolean("big_net", False, "Use bigger net...")
flags.DEFINE_boolean("only_output", False, "Train only the last layer...")
flags.DEFINE_boolean("do_plot", False, "Do intermediate plots")
flags.DEFINE_integer("random_seed", 1234, "Random seed to use")


def main():
    app.run(mnist.main)


if __name__ == "__main__":
    main()
