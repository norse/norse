from setuptools import setup


with open('requirements.txt') as fp:
    install_requires = fp.read()
    print(install_requires)

setup(
    install_requires=install_requires,
    name="myelin",
    description="A library for deep learning with spiking neural networks",
    url="http://github.com/cpehle/myelin",
    author="Christian Pehle",
    author_email="christian.pehle@gmail.com",
    packages=["myelin"],
)
