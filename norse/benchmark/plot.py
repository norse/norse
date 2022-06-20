import argparse
import matplotlib.pyplot as plt

# pytype: disable=import-error
import pandas as pd

# pytype: enable=import-error
import numpy as np


def plot_frames(frames, title):
    render_frames(frames, title)
    plt.show()


def save_frames(frames, title, filename):
    render_frames(frames, title)
    plt.savefig(filename)


def render_frames(frames, title):
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.set_yscale("log")
    for group in frames:
        for label, frame in group.groupby("label"):
            label = label.replace("lif", "LIF")
            plt.fill_between(
                frame["input_features"],
                frame["duration_mean"] - frame["duration_std"] * 2,
                frame["duration_mean"] + frame["duration_std"] * 2,
                alpha=0.2,
            )
            frame.plot(y="duration_mean", x="input_features", ax=ax, label=label)
            # Plot the crash, if any
            is_na = frame["duration_mean"].isnull()
            if is_na.any():
                last_index = is_na[is_na == True].index[0] - 1
                last_value = frame.loc[last_index]
                plt.scatter(
                    x=[last_value["input_features"]],
                    y=[last_value["duration_mean"]],
                    color=ax.lines[-1].get_color(),
                    s=60,
                )

    xmin = frame["input_features"].min()
    xmax = frame["input_features"].max()

    ax.set_xlim(xmin, xmax)
    ax.set_xticks(np.arange(xmin, xmax + 1, 500))
    ax.set_title(title)
    ax.set_xlabel("Number of neurons")
    ax.set_ylabel("Running time in seconds")
    ax.legend(loc="upper left")
    return ax


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

    dfs = [pd.read_csv(f) for f in args.files]
    if args.to:
        save_frames(dfs, args.title, args.to)
    else:
        plot_frames(dfs, args.title)
