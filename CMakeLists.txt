cmake_minimum_required(VERSION 3.15)
project(field_boundary_mapping LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(field_boundary_core src/bindings.cpp)
target_include_directories(field_boundary_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(field_boundary_core PRIVATE NOMINMAX)
target_compile_options(field_boundary_core PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall;-Wextra;-Wpedantic>
)
add_custom_command(TARGET field_boundary_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/lib"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:field_boundary_core>
        "${CMAKE_CURRENT_SOURCE_DIR}/lib/$<TARGET_FILE_NAME:field_boundary_core>"
    VERBATIM
)
enable_testing()
add_test(NAME field_boundary_core_tests
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_core.py"
)
set_tests_properties(field_boundary_core_tests PROPERTIES
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
)
