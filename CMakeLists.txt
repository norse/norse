cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(norse)

find_package(Torch REQUIRED)
FIND_PACKAGE(Python3 REQUIRED COMPONENTS Development)

add_subdirectory(norse/csrc)
