cmake_minimum_required (VERSION 2.8.11)
project(Project)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIR})

add_subdirectory(src)
