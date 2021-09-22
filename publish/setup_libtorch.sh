#!/bin/env sh

set -euo pipefail # Quit on error

LIBTORCH_VERSION=$1
curl -o libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip
unzip libtorch
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/libtorch/