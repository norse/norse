cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(norse)

# PyTorch requires C++14
set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
FIND_PACKAGE(Python3 REQUIRED COMPONENTS Development)

add_subdirectory(norse/csrc)
