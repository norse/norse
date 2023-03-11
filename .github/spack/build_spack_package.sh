#!/bin/bash

set -euo pipefail
shopt -s inherit_errexit

WORKSPACE=${PWD}
BUILDCACHE_MIRROR=~/.spack-cache/mirror

TMP_DIR=$(mktemp -d)
cd "${TMP_DIR}"

mkdir ~/.spack
cp "$WORKSPACE/.github/spack/config.yaml" ~/.spack

# stick to latest stable spack release
SPACK_REPO=https://github.com/spack/spack
SPACK_BRANCH=releases/latest
git clone --depth 1 --branch "${SPACK_BRANCH}" "${SPACK_REPO}" spack

source spack/share/spack/setup-env.sh

# ignore existing installed or buildcache packages for concretization
spack config --scope defaults add concretizer:reuse:false

cat <<EOF > spack/etc/spack/packages.yaml
packages:
  all:
    compiler: [$COMPILER_VERSION]
EOF

spack repo create custom_repo
mkdir -p custom_repo/packages/py-norse
cp "$WORKSPACE/spack/package.py" custom_repo/packages/py-norse
spack repo add "${TMP_DIR}/custom_repo"

# we install a stripped down py-torch (no cuda, mpi, ...)
PACKAGE_PYTORCH_VARIANT="~cuda~mkldnn~rocm~distributed~onnx_ml~xnnpack~valgrind~breakpad"
PACKAGE_PYTORCH="py-torch${PACKAGE_PYTORCH_VARIANT}"

# pin specific py-torch version if provided by env var
if [ -n "${MATRIX_SPACK_PYTORCH}" ]; then
    PACKAGE_PYTORCH="${MATRIX_SPACK_PYTORCH}${PACKAGE_PYTORCH_VARIANT}"
fi

# pin specific python version if provided by env var
if [ -n "${MATRIX_SPACK_PYTHON}" ]; then
    PACKAGE_PYTORCH="${PACKAGE_PYTORCH} ^${MATRIX_SPACK_PYTHON}"
fi

# the ubuntu CI runner runs on multiple cpu archs; compile for an old one
ARCH="linux-ubuntu22.04-x86_64_v4"

# do some more cache cleaning
spack clean -fmp

# trigger spack's bootstrapping
spack spec zlib

echo "Compilers known to spack:"
spack compiler find /usr/bin
spack compilers

echo "spack spec of increasing specificity:"

echo "spack spec -t ${PACKAGE_PYTORCH}"
spack spec -t "${PACKAGE_PYTORCH}"
echo "spack spec -t py-norse@main"
spack spec -t py-norse@main
echo "spack spec -t py-norse@main ^${PACKAGE_PYTORCH}"
spack spec -t py-norse@main "^${PACKAGE_PYTORCH}"
echo "spack spec -t -I py-norse@main ^${PACKAGE_PYTORCH} arch=${ARCH}"
spack spec -t -I py-norse@main "^${PACKAGE_PYTORCH}" "arch=${ARCH}"

mkdir -p "${BUILDCACHE_MIRROR}"/build_cache

# enable buildcache (for faster CI)
spack mirror add spack_ci_cache "${BUILDCACHE_MIRROR}"

# drop py-norse CI builds from build cache
rm -rf "${BUILDCACHE_MIRROR}"/build_cache/*/*/py-norse-main
rm -rf "${BUILDCACHE_MIRROR}"/build_cache/*-py-norse-main-*.json

# (re)index the cache
spack buildcache update-index -d "${BUILDCACHE_MIRROR}"

echo "Build cache contents:"
spack buildcache list -aL

echo "Installed spack packages (pre-build):"
spack find -L

# drop staged builds anyways
spack clean --stage

if spack find py-norse; then
   spack uninstall --yes-to-all --all py-norse
fi

ret=0
spack dev-build --source-path "${WORKSPACE}" py-norse@main "^${PACKAGE_PYTORCH}" "arch=${ARCH}" || ret=$?

echo "Installed spack packages (post-build):"
spack find -L

# fill build cache
mkdir -p "${BUILDCACHE_MIRROR}"
for s in $(spack find --no-groups -L | cut -f 1 -d ' ' ); do
    spack buildcache create -d "${BUILDCACHE_MIRROR}" -a -u --only package "/$s"
done

echo "Build cache:"
spack buildcache list

# return exit code from spack call
exit $ret
