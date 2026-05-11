cmake_minimum_required(VERSION 3.16)

project(spinnv2_runtime C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_library(spkv2_runtime STATIC
    core/version.c
)

target_include_directories(spkv2_runtime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_testing()

add_executable(spkv2_runtime_smoke
    tests/test_runtime_smoke.c
)

target_link_libraries(spkv2_runtime_smoke PRIVATE spkv2_runtime)

add_test(NAME runtime_unit_smoke COMMAND spkv2_runtime_smoke)

