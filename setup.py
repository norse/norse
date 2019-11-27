import setuptools
from setuptools import setup

from os import path
pwd = path.abspath(path.dirname(__file__))

with open(path.join(pwd, 'requirements.txt')) as fp:
    install_requires = fp.read()

with open(path.join(pwd, 'Readme.md'), encoding='utf-8') as fp:
    readme_text = fp.read()

setup(
    install_requires=install_requires,
    name='norse',
    version = '0.0.1',
    description='A library for deep learning with spiking neural networks',
    long_description=readme_text,
    long_description_content_type='text/markdown',
    url='http://github.com/electronicvisions/norse',
    author='Christian Pehle',
    author_email='christian.pehle@gmail.com',
    packages=setuptools.find_packages(),
)
