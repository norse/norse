cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(norse)

find_package(Torch REQUIRED)
find_package (Python3 REQUIRED COMPONENTS Development)

add_library(norse SHARED op.cpp)
target_compile_features(norse PRIVATE cxx_std_14)
target_include_directories(norse PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(norse ${Python3_LIBRARIES})
target_link_libraries(norse ${TORCH_LIBRARIES})