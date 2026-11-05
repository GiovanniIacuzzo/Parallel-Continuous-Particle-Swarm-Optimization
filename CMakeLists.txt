cmake_minimum_required(VERSION 3.20)
project(cpso_serial VERSION 0.6.0 LANGUAGES CXX)

option(CPSO_BUILD_TESTS "Build the C++ test suite" ON)
option(CPSO_ENABLE_WARNINGS "Enable strict compiler warnings" ON)

find_package(Threads REQUIRED)

add_library(cpso_serial SHARED
    cpp/src/analytic_propagator.cpp
    cpp/src/objectives.cpp
    cpp/src/optimizer.cpp
    cpp/src/asynchronous_optimizer.cpp
    cpp/src/island_optimizer.cpp
    cpp/src/topology.cpp
    cpp/src/random_stream.cpp
    cpp/src/c_api.cpp
)

target_include_directories(cpso_serial
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)
target_compile_features(cpso_serial PUBLIC cxx_std_20)
target_link_libraries(cpso_serial PRIVATE Threads::Threads)
string(TOUPPER "${CMAKE_BUILD_TYPE}" CPSO_BUILD_TYPE_UPPER)
set(CPSO_EFFECTIVE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CPSO_BUILD_TYPE_UPPER}}")
string(REPLACE ";" " " CPSO_EFFECTIVE_CXX_FLAGS "${CPSO_EFFECTIVE_CXX_FLAGS}")
string(STRIP "${CPSO_EFFECTIVE_CXX_FLAGS}" CPSO_EFFECTIVE_CXX_FLAGS)
string(REPLACE "\\" "\\\\" CPSO_EFFECTIVE_CXX_FLAGS "${CPSO_EFFECTIVE_CXX_FLAGS}")
string(REPLACE "\"" "\\\"" CPSO_EFFECTIVE_CXX_FLAGS "${CPSO_EFFECTIVE_CXX_FLAGS}")
target_compile_definitions(cpso_serial PRIVATE
    CPSO_PROJECT_VERSION="${PROJECT_VERSION}"
    CPSO_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
    CPSO_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
    CPSO_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    CPSO_CXX_FLAGS="${CPSO_EFFECTIVE_CXX_FLAGS}"
    CPSO_TARGET_PROCESSOR="${CMAKE_SYSTEM_PROCESSOR}"
)
set_target_properties(cpso_serial PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(CPSO_ENABLE_WARNINGS)
    if(MSVC)
        target_compile_options(cpso_serial PRIVATE /W4 /permissive-)
    else()
        target_compile_options(cpso_serial PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()
endif()

if(CPSO_BUILD_TESTS)
    enable_testing()
    add_executable(cpso_cpp_tests cpp/tests/test_cpso.cpp)
    target_link_libraries(cpso_cpp_tests PRIVATE cpso_serial)
    target_compile_features(cpso_cpp_tests PRIVATE cxx_std_20)
    add_test(NAME cpso_cpp_tests COMMAND cpso_cpp_tests)
endif()
