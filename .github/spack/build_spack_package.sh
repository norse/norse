#!/bin/bash

set -euxo pipefail
shopt -s inherit_errexit

WORKSPACE=${PWD}

TMP_DIR=$(mktemp -d)
cd "${TMP_DIR}"

mkdir ~/.spack
cp "$WORKSPACE/.github/spack/config.yaml" ~/.spack

# stick to latest stable spack release
SPACK_REPO=https://github.com/spack/spack
SPACK_BRANCH=releases/latest
git clone --depth 1 --branch "${SPACK_BRANCH}" "${SPACK_REPO}" spack

source spack/share/spack/setup-env.sh

spack repo create custom_repo
mkdir -p custom_repo/packages/py-norse
cp "$WORKSPACE/spack/package.py" custom_repo/packages/py-norse
spack repo add "${TMP_DIR}/custom_repo"

# we install a stripped down py-torch (no cuda, mpi, ...)
PACKAGE_PYTORCH="py-torch~cuda~cudnn~mkldnn~distributed~nccl"

# the ubuntu CI runner runs on multiple cpu archs; compile for an old one
ARCH="linux-ubuntu20.04-sandybridge"

spack spec -I py-norse@master ^${PACKAGE_PYTORCH} arch=${ARCH}

# drop staged builds anyways
spack clean --stage

if spack find py-norse; then
   spack uninstall --yes-to-all --all py-norse
fi
spack dev-build --source-path "${WORKSPACE}" py-norse@master ^${PACKAGE_PYTORCH} arch=${ARCH}
