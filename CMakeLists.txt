cmake_minimum_required(VERSION 3.18)
project(apriltag-python-bindings)

option(BUILD_EXAMPLES "" OFF)
option(BUILD_PYTHON_WRAPPER "" OFF)

add_subdirectory(apriltag skbuild-lib)
