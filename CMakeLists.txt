cmake_minimum_required(VERSION 3.20)
project(cpso_hpc VERSION 0.1.0 LANGUAGES CXX)

option(CPSO_ENABLE_OPENMP "Enable shared-memory parallel execution with OpenMP" ON)
option(CPSO_BUILD_TESTS "Build the C++ test executable" ON)

add_library(cpso_core SHARED
    cpp/src/cpso_engine.cpp
    cpp/src/c_api.cpp
)
target_include_directories(cpso_core PUBLIC cpp/include PRIVATE cpp/src)
target_compile_features(cpso_core PUBLIC cxx_std_17)
target_compile_options(cpso_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -Wall -Wextra -Wpedantic -Wconversion -Wshadow>
)
set_target_properties(cpso_core PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    OUTPUT_NAME cpso_core
)

if(CPSO_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(cpso_core PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(cpso_core PRIVATE CPSO_WITH_OPENMP=1)
        message(STATUS "CPSO: OpenMP enabled (${OpenMP_CXX_VERSION})")
    else()
        message(WARNING "OpenMP not found: building a serial fallback")
    endif()
endif()

if(CPSO_BUILD_TESTS)
    enable_testing()
    add_executable(cpso_cpp_tests cpp/tests/test_engine.cpp cpp/src/cpso_engine.cpp)
    target_include_directories(cpso_cpp_tests PRIVATE cpp/include cpp/src)
    target_compile_features(cpso_cpp_tests PRIVATE cxx_std_17)
    if(CPSO_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(cpso_cpp_tests PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(cpso_cpp_tests PRIVATE CPSO_WITH_OPENMP=1)
    endif()
    add_test(NAME cpso_cpp_tests COMMAND cpso_cpp_tests)
endif()

install(TARGETS cpso_core
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
)
install(DIRECTORY cpp/include/ DESTINATION include)
