cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(norse VERSION 0.0.6)

# PyTorch requires C++14
set(CMAKE_CXX_STANDARD 17)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

find_package(Torch REQUIRED)
FIND_PACKAGE(Python3 REQUIRED COMPONENTS Development)

add_subdirectory(norse/csrc)