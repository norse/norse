import argparse
import matplotlib.pyplot as plt
import pandas as pd
import sys


def plot_frames(frames):
    render_frames(frames)
    plt.show()


def render_frames(frames):
    ax = plt.gca()
    for frame in frames:
        print(frame.keys())
        frame.plot(y="duration_mean", x="input_features", ax=ax, label="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot a number of csv benchark files against each other"
    )
    parser.add_argument("--files", type=argparse.FileType("r"), nargs="+")
    args = parser.parse_args()

    files = args.files
    dfs = []
    for f in files:
        dfs.append(pd.read_csv(f))
    plot_frames(dfs)
