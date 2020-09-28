import importlib
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_enum("task", "mnist", ["cifar", "gym", "mnist"], "Task to run.")


def main(argv):
    task_module = importlib.import_module(
        f"norse.examples.run_{FLAGS.task}", package="."
    )
    app.run(task_module.main)


if __name__ == "__main__":
    app.run(main)
