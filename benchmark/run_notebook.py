#!/usr/bin/env python

import nbformat, fire
from nbconvert.preprocessors import ExecutePreprocessor


def run_notebook(path):
    nb = nbformat.read(open(path), as_version=nbformat.NO_CONVERT)
    ExecutePreprocessor(timeout=600).preprocess(nb, {})
    print("Done")


if __name__ == "__main__":
    fire.Fire(run_notebook)
