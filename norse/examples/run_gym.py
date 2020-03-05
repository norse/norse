#!/usr/bin/env python

"""
A cartpole learning example with OpenAI gym
===========================================

This example illustrates how to use
`OpenAI Gym <https://gym.openai.com/>`_ to
train a cartpole task.
"""

import norse.task.cartpole as cartpole
from absl import app
from absl import flags

flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"],
                  "Device to use by pytorch.")
flags.DEFINE_integer("episodes", 100, "Number of training trials.")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate to use.")
flags.DEFINE_float("gamma", 0.99, "discount factor to use")
flags.DEFINE_integer(
    "log_interval", 10, "In which intervals to display learning progress."
)
flags.DEFINE_enum("model", "super", ["super"], "Model to use for training.")
flags.DEFINE_enum("policy", "snn", ["snn", "lsnn", "ann"],
                  "Select policy to use.")
flags.DEFINE_boolean("render", False, "Render the environment")
flags.DEFINE_string("environment", "CartPole-v1", "Gym environment to use.")
flags.DEFINE_integer("random_seed", 1234, "Random seed to use")


def main():
    app.run(cartpole.main)


if __name__ == "__main__":
    main()
