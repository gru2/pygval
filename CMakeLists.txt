project(pygval)
cmake_minimum_required(VERSION 3.1)

set(CMAKE_DEBUG_POSTFIX "d")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O2")
set (CMAKE_CXX_STANDARD 11)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(PkgConfig)
pkg_search_module(Eigen3 REQUIRED eigen3)

set(WORK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
link_directories(${WORK_DIR}/Usutf/build ${WORK_DIR}/GVal/build)
include_directories(. ${WORK_DIR}/Usutf ${WORK_DIR}/GVal ${PYTHON_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIRS})

add_library(pygval SHARED pygval.cpp)
target_link_libraries(pygval debug GVald optimized GVal debug Usutfd optimized Usutf)
set_target_properties(
    pygval
    PROPERTIES
        PREFIX ""
)
