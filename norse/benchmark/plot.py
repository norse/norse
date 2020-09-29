import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd
import sys
import math


def plot_frames(frames, title):
    render_frames(frames, title)
    plt.show()


def save_frames(frames, title, filename):
    render_frames(frames, title)
    plt.savefig(filename)


def render_frames(frames, title):
    ax = plt.gca(yscale="log")
    for frame in frames:
        label = frame["label"][0].replace("_lif", "")
        plt.fill_between(
            frame["input_features"],
            frame["duration_mean"] - frame["duration_std"] * 2,
            frame["duration_mean"] + frame["duration_std"] * 2,
            alpha=0.2,
        )
        frame.plot(y="duration_mean", x="input_features", ax=ax, label=label)
        # Plot the crash, if any
        is_na = frame['duration_mean'].isnull()
        if is_na.any():
            last_index = is_na[is_na == True].index[0] - 1
            last_value = frame.loc[last_index]
            plt.scatter(x=[last_value['input_features']], y=[last_value['duration_mean']], color=ax.lines[-1].get_color(), s=60)
    
    xmin = frame['input_features'].min()
    xmax = frame['input_features'].max()
    
    ax.set_xlim(math.floor(xmin / 1000), math.ceil(xmax / 1000) * 1000)
    ax.set_title(title)
    ax.set_xlabel("No. of features")
    ax.set_ylabel("Running time in seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot a number of csv benchark files against each other"
    )
    parser.add_argument("files", type=argparse.FileType("r"), nargs="+")
    parser.add_argument(
        "--to",
        type=str,
        required=False,
        help="Save to given file instead of displaying",
    )
    parser.add_argument("--title", type=str, default="", help="Figure title")
    args = parser.parse_args()

    files = args.files
    dfs = []
    for f in files:
        dfs.append(pd.read_csv(f))
    if args.to:
        save_frames(dfs, args.title, args.to)
    else:
        plot_frames(dfs, args.title)
