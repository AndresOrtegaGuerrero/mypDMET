cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(cpp_libdmet)

# Set INCLUDE and LIB
LINK_DIRECTORIES(/user-environment/env/default/lib) # Adjust this path as needed
INCLUDE_DIRECTORIES(/user-environment/env/default/include) # Adjust this path as needed

# Find the LAWrap for mkl_blas/lapack, pybind11
find_package(LAWrap CONFIG QUIET)

if(NOT LAWrap_FOUND)
    message(STATUS "LAWrap not found in system, searching in local directory...")
    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/lawrap")
    find_package(LAWrap CONFIG REQUIRED)
endif()

message(STATUS "Found LAWrap: ${LAWrap_CONFIG}")

# Set python
if(DEFINED ENV{CONDA_PREFIX})
    set(Python3_ROOT_DIR "$ENV{CONDA_PREFIX}")
    message(STATUS "Using Conda Python from: $ENV{CONDA_PREFIX}")
elseif(DEFINED ENV{VIRTUAL_ENV})
    set(Python3_ROOT_DIR "$ENV{VIRTUAL_ENV}")
    message(STATUS "Using venv Python from: $ENV{VIRTUAL_ENV}")
endif()

set(Python3_FIND_VIRTUALENV FIRST)
set(Python3_FIND_STRATEGY LOCATION)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python executable: ${Python3_EXECUTABLE}")

# Get Python site-packages directory
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE Python3_SITELIB
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Python site-packages: ${Python3_SITELIB}")

# Look for pybind11 in site-packages
set(pybind11_DIR "${Python3_SITELIB}/pybind11/share/cmake/pybind11")
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Found pybind11: ${pybind11_CONFIG}")

# Openmp
include(FindOpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-depth=2048")

# Set pybind11 c++ standard
set(PYBIND11_CPP_STANDARD -std=c++11)

# Creates a python module named "module_name"
pybind11_add_module(libdmet MODULE libdmet.cpp)
if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    # Intel / AMD
    message(STATUS "Using Intel MKL")
    target_link_libraries(libdmet
        PUBLIC
        mkl_intel_lp64
        mkl_sequential
        mkl_core
    )
else()
    # ARM, Apple Silicon, etc.
    message(STATUS "Using OpenBLAS")
    target_link_libraries(libdmet
        PUBLIC
        /user-environment/env/default/lib/libopenblas.so
        gfortran
    )
endif()
