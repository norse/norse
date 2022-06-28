#!/bin/bash

set -euxo pipefail
shopt -s inherit_errexit

WORKSPACE=${PWD}
BUILDCACHE_MIRROR=${WORKSPACE}/.spack-cache/mirror

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
PACKAGE_PYTORCH="py-torch@:1.10~cuda~mkldnn~rocm~distributed~onnx_ml~xnnpack~valgrind"

# the ubuntu CI runner runs on multiple cpu archs; compile for an old one
ARCH="linux-ubuntu20.04-x86_64"

# trigger spack's bootstrapping
spack spec zlib

echo "Compilers known to spack:"
spack compiler find /usr/bin
spack compilers

echo "spack spec of increasing specificity:"
spack spec ${PACKAGE_PYTORCH}
spack spec py-norse@master
spack spec py-norse@master ^${PACKAGE_PYTORCH}
spack spec -I py-norse@master ^${PACKAGE_PYTORCH} arch=${ARCH}

# enable buildcache (for faster CI)
spack mirror add spack_ci_cache "${BUILDCACHE_MIRROR}"
spack buildcache update-index -d "${BUILDCACHE_MIRROR}"

echo "Build cache contents:"
spack buildcache list

# drop staged builds anyways
spack clean --stage

if spack find py-norse; then
   spack uninstall --yes-to-all --all py-norse
fi

ret=0
spack dev-build --source-path "${WORKSPACE}" py-norse@master ^${PACKAGE_PYTORCH} arch=${ARCH} || ret=$?

echo "Installed spack packages:"
spack find --no-groups -L

# fill build cache
mkdir -p "${BUILDCACHE_MIRROR}"
for s in $(spack find --no-groups -L | cut -f 1 -d ' ' ); do
    spack buildcache create -d "${BUILDCACHE_MIRROR}" -a -u --only package "/$s"
done

echo "Build cache:"
spack buildcache list

# return exit code from spack call
exit $ret
