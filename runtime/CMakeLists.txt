cmake_minimum_required(VERSION 3.16)

project(spinnv2_runtime C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Detect AVX2 + FMA support
include(CheckCCompilerFlag)
check_c_compiler_flag("-mavx2 -mfma" SPKV2_HAS_AVX2)

# OpenMP
find_package(OpenMP)

set(SPKV2_PLATFORM_SOURCE
    platform/platform.c
    CACHE STRING "Platform abstraction source file for SPINNV2 runtime"
)

file(GLOB_RECURSE SPKV2_REFERENCE_KERNEL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/reference/*.c"
)

file(GLOB_RECURSE SPKV2_OPTIMIZED_KERNEL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/optimized/*.c"
)

add_library(spkv2_runtime STATIC
    core/version.c
    core/loader.c
    core/executor.c
    ${SPKV2_REFERENCE_KERNEL_SOURCES}
    ${SPKV2_OPTIMIZED_KERNEL_SOURCES}
    ${SPKV2_PLATFORM_SOURCE}
)

if(SPKV2_HAS_AVX2)
    set(SIMD_FLAGS "-mavx2 -mfma")
    if(OpenMP_C_FOUND)
        set(SIMD_FLAGS "${SIMD_FLAGS} ${OpenMP_C_FLAGS}")
    endif()
    set_source_files_properties(
        ${SPKV2_OPTIMIZED_KERNEL_SOURCES}
        kernels/reference/kernels.c
        PROPERTIES COMPILE_FLAGS "${SIMD_FLAGS}"
    )
endif()

target_include_directories(spkv2_runtime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/reference
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/optimized
)

target_link_libraries(spkv2_runtime PUBLIC m)
if(OpenMP_C_FOUND)
    target_link_libraries(spkv2_runtime PUBLIC OpenMP::OpenMP_C)
endif()

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

add_executable(spkv2_prepare_conv_cache_test
    tests/test_prepare_conv_cache.c
)

target_include_directories(spkv2_prepare_conv_cache_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
)

target_link_libraries(spkv2_prepare_conv_cache_test PRIVATE spkv2_runtime)

add_test(NAME runtime_prepare_conv_cache COMMAND spkv2_prepare_conv_cache_test)

add_executable(spkv2_sgemm_epilogue_test
    tests/test_sgemm_epilogue.c
)

target_include_directories(spkv2_sgemm_epilogue_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/optimized
)

target_link_libraries(spkv2_sgemm_epilogue_test PRIVATE spkv2_runtime m)

add_test(NAME runtime_sgemm_epilogue COMMAND spkv2_sgemm_epilogue_test)

add_executable(spkv2_run
    tools/spkv2_run.c
)

target_link_libraries(spkv2_run PRIVATE spkv2_runtime m)

add_executable(spkv2_bench
    tools/spkv2_bench.c
)

target_link_libraries(spkv2_bench PRIVATE spkv2_runtime m)
