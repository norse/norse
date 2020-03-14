import subprocess
import sys


def main(argv):
    if argv[1] not in ["cifar", "gym", "mnist"]:
        sys.stderr.write("Error: Expected 'cifar', 'gym', or 'mnist'.\n")
        exit(-1)

    task_module = "norse/examples/run_" + argv[1] + ".py"
    subprocess.call([sys.executable, task_module] + argv[2:])


if __name__ == "__main__":
    main(sys.argv)
