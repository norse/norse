import setuptools
from setuptools import setup

with open('requirements.txt') as fp:
    install_requires = fp.read()

setup(
    install_requires=install_requires,
    name="norse",
    description="A library for deep learning with spiking neural networks",
    url="http://github.com/electronicvisions/norse",
    author="Christian Pehle",
    author_email="christian.pehle@gmail.com",
    packages=setuptools.find_packages(),
)
