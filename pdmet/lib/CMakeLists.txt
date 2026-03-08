cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(cpp_libdmet)

# Set INCLUDE and LIB
#LINK_DIRECTORIES(/user-environment/env/default/lib) # Adjust this path as needed
#INCLUDE_DIRECTORIES(/user-environment/env/default/include) # Adjust this path as needed

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

if (LAWRAP_INCLUDE_DIR)
    message(STATUS "Found LAWrap in: ${LAWRAP_INCLUDE_DIR}")
    include_directories(${LAWRAP_INCLUDE_DIR})
else()
    message(STATUS "Using bundled LAWrap: ${CMAKE_CURRENT_SOURCE_DIR}/lawrap")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lawrap)
endif()

# Set python
if(DEFINED ENV{CONDA_PREFIX})
    set(PYTHON_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
    message(STATUS "Using Conda Python: ${PYTHON_EXECUTABLE}")
elseif(DEFINED ENV{VIRTUAL_ENV})
    set(PYTHON_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python")
    message(STATUS "Using venv Python: ${PYTHON_EXECUTABLE}")
else()
    # Fall back to system python
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    set(PYTHON_EXECUTABLE "${Python_EXECUTABLE}")
    message(STATUS "Using system Python: ${PYTHON_EXECUTABLE}")
endif()

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Python site-packages: ${PYTHON_SITE_PACKAGES}")

# Look for pybind11 in site-packages
set(PYBIND11_INCLUDE_DIR "${PYTHON_SITE_PACKAGES}/pybind11/include")
message(STATUS "PyBind11 headers: ${PYBIND11_INCLUDE_DIR}")

set(pybind11_DIR "${Python3_SITELIB}/pybind11/share/cmake/pybind11")
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Found pybind11: ${pybind11_DIR}")


if(APPLE)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(LIBOMP_PREFIX)
        message(STATUS "Found libomp at: ${LIBOMP_PREFIX}")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")

        # Explicitly set include and lib directories
        set(OpenMP_CXX_INCLUDE_DIRS "${LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_LIBRARIES "${LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()

# Openmp
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message(STATUS "OpenMP flags: ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not found - parallel support disabled")
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
        gfortran
    )
endif()
