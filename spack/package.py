# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class PyNorse(PythonPackage):
    """A deep learning library for spiking neural networks."""

    homepage = "https://norse.github.io/norse/"
    git = "https://github.com/norse/norse.git"
    pypi = "norse/norse-0.0.7.post1.tar.gz"

    version("master", branch="master")
    version(
        "0.0.7.post1",
        sha256="aeea3bd08f47fcfe3b301f1190928dec482938956a1ab8ba568851deed94bda5",
    )

    depends_on("python@3.7.0:", type=("build", "run"))
    depends_on("py-numpy", type=("build", "run"))
    depends_on("py-torch@1.9.0:", type=("build", "run"))
    depends_on("py-torchvision@0.10.0:", type=("build", "run"))
    depends_on("py-setuptools", type=("build", "run"))
    depends_on("py-pybind11", type=("build", "link", "run"))

    def setup_build_environment(self, env):
        include = []
        library = []
        for dep in self.spec.dependencies(deptype="link"):
            query = self.spec[dep.name]
            include.extend(query.headers.directories)
            if "py-pybind11" in dep:
                # py-pybind11 does not provide any libraries for spack to
                # find, this raises an error -> fix on py-pybind11 side
                # at some point in the future
                continue
            library.extend(query.libs.directories)
