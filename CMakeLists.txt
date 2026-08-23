cmake_minimum_required(VERSION 3.20)
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
        src/math/src/SparseMatrix.cpp
        src/math/src/LinearSolver.cpp
        src/io/src/VtkWriter.cpp
        src/numerical/src/Diffusion.cpp
        src/numerical/src/Convection.cpp
        src/numerical/src/TransportEquation.cpp
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

# Main executable
add_executable(fvm_solver src/app/main.cpp)
target_link_libraries(fvm_solver PRIVATE fvm_core)

# Tests
enable_testing()
add_executable(fvm_tests)

target_sources(fvm_tests
    PRIVATE
        tests/test_mesh.cpp
        tests/test_linalg.cpp
        tests/test_field.cpp
        tests/test_diffusion.cpp
        tests/test_convection.cpp
        tests/test_transport.cpp
        tests/main.cpp
)

target_link_libraries(fvm_tests
    PRIVATE
        fvm_core
        doctest::doctest
)

add_test(NAME fvm_unit_tests COMMAND fvm_tests)

# Install
install(TARGETS fvm_solver DESTINATION bin)
