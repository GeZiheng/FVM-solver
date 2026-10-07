cmake_minimum_required(VERSION 3.20)

# Python bindings pull pybind11 through the vcpkg "python" manifest feature,
# and replace the vcpkg python3 port with the empty overlay in
# python/overlay-ports (the extension targets the system interpreter via
# FindPython; vcpkg's own CPython is never needed). Both variables must be
# set before project() so the vcpkg toolchain picks them up.
if(DEFINED FVM_BUILD_PYTHON AND FVM_BUILD_PYTHON)
    set(VCPKG_MANIFEST_FEATURES "python" CACHE STRING "vcpkg manifest features" FORCE)
    set(VCPKG_OVERLAY_PORTS "${CMAKE_CURRENT_SOURCE_DIR}/python/overlay-ports"
        CACHE PATH "vcpkg overlay ports" FORCE)
endif()

project(FVM-solver VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find packages via vcpkg
find_package(Eigen3 CONFIG REQUIRED)
find_package(doctest CONFIG REQUIRED)

# Core library
add_library(fvm_core STATIC)

target_sources(fvm_core
    PRIVATE
        src/core/src/Mesh.cpp
        src/core/src/Field.cpp
        src/core/src/FluxField.cpp
        src/math/src/SparseMatrix.cpp
        src/math/src/LinearSolver.cpp
        src/io/src/VtkWriter.cpp
        src/numerical/src/Diffusion.cpp
        src/numerical/src/Convection.cpp
        src/numerical/src/GridOperators.cpp
        src/numerical/src/TransportEquation.cpp
        src/numerical/src/Simple.cpp
        src/numerical/src/Momentum.cpp
        src/numerical/src/Pressure.cpp
        src/numerical/src/Piso.cpp
)

target_include_directories(fvm_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/math/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/io/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/numerical/include
)

target_link_libraries(fvm_core
    PUBLIC
        Eigen3::Eigen
)

# Tests
enable_testing()
add_executable(fvm_tests)

target_sources(fvm_tests
    PRIVATE
        tests/test_mesh.cpp
        tests/test_linalg.cpp
        tests/test_field.cpp
        tests/test_flux.cpp
        tests/test_diffusion.cpp
        tests/test_convection.cpp
        tests/test_transport.cpp
        tests/test_simple.cpp
        tests/test_transient.cpp
        tests/test_piso.cpp
        tests/main.cpp
)

target_link_libraries(fvm_tests
    PRIVATE
        fvm_core
        doctest::doctest
)

add_test(NAME fvm_unit_tests COMMAND fvm_tests)

# Python bindings (optional): pybind11 extension module `pyfvm`.
# Configure with -DFVM_BUILD_PYTHON=ON; the .pyd lands in <build>/python.
option(FVM_BUILD_PYTHON "Build the pyfvm Python bindings (pybind11)" OFF)
if(FVM_BUILD_PYTHON)
    # The extension is version-locked to the interpreter it is built against
    # (a cp312 module does not import in 3.14). Honour an explicit
    # -DPython_EXECUTABLE=..., otherwise take the first python on PATH.
    if(NOT DEFINED Python_EXECUTABLE)
        find_program(Python_EXECUTABLE NAMES python3.14 python3 python)
    endif()
    set(PYBIND11_FINDPYTHON ON) # modern FindPython (honours Python_EXECUTABLE)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(pyfvm python/pyfvm.cpp)
    target_link_libraries(pyfvm PRIVATE fvm_core)
    set_target_properties(pyfvm PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python")
    if(MSVC)
        target_compile_options(pyfvm PRIVATE /bigobj)
        target_compile_definitions(pyfvm PRIVATE NOMINMAX)
    endif()
    # Run the binding tests with the interpreter the module was built
    # against (extension modules are version-locked).
    add_test(NAME pyfvm_tests
        COMMAND ${Python_EXECUTABLE} -m unittest discover
                -s ${CMAKE_CURRENT_SOURCE_DIR}/python/tests)
endif()
