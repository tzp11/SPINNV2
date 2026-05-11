cmake_minimum_required(VERSION 3.16)

project(spinnv2_runtime C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(SPKV2_PLATFORM_SOURCE
    platform/platform.c
    CACHE STRING "Platform abstraction source file for SPINNV2 runtime"
)

add_library(spkv2_runtime STATIC
    core/version.c
    core/loader.c
    core/executor.c
    kernels/reference/kernels.c
    ${SPKV2_PLATFORM_SOURCE}
)

target_include_directories(spkv2_runtime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core
)

target_link_libraries(spkv2_runtime PUBLIC m)

enable_testing()

add_executable(spkv2_runtime_smoke
    tests/test_runtime_smoke.c
)

target_link_libraries(spkv2_runtime_smoke PRIVATE spkv2_runtime)

add_test(NAME runtime_unit_smoke COMMAND spkv2_runtime_smoke)

add_executable(spkv2_prepare_arena_test
    tests/test_prepare_arena.c
)

target_link_libraries(spkv2_prepare_arena_test PRIVATE spkv2_runtime)

add_test(NAME runtime_prepare_arena COMMAND spkv2_prepare_arena_test)

add_executable(spkv2_run
    tools/spkv2_run.c
)

target_link_libraries(spkv2_run PRIVATE spkv2_runtime m)
