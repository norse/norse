cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(norse)

find_package(Torch REQUIRED)
FIND_PACKAGE(PythonInterp 3 REQUIRED)
FIND_PACKAGE(PythonLibs 3 REQUIRED)

add_subdirectory(norse/csrc)