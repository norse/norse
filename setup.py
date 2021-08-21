from os import path

import setuptools
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CppExtension

pwd = path.abspath(path.dirname(__file__))

with open(path.join(pwd, "requirements.txt")) as fp:
    install_requires = fp.read()

with open(path.join(pwd, "README.md"), encoding="utf-8") as fp:
    readme_text = fp.read()

cpp_extension = CppExtension(
    name="norse_op",
    sources=["norse/csrc/op.cpp", "norse/csrc/super.cpp"],
    extra_compile_args=["-O3"],
)

setup(
    install_requires=install_requires,
    setup_requires=["setuptools", "wheel", "torch"],
    name="norse",
    version="0.0.7RC1",
    description="A library for deep learning with spiking neural networks",
    long_description=readme_text,
    long_description_content_type="text/markdown",
    url="http://github.com/norse/norse",
    author="Christian Pehle, Jens E. Pedersen",
    author_email="christian.pehle@gmail.com, jens@jepedersen.dk",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="machine learning spiking neural networks",
    ext_modules=[cpp_extension],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
