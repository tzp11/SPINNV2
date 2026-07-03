cmake_minimum_required(VERSION 3.16)

project(spinnv2_runtime C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Detect AVX2 + FMA support
include(CheckCCompilerFlag)
check_c_compiler_flag("-mavx2 -mfma" SPKV2_HAS_AVX2)
check_c_compiler_flag("-march=armv8-a+simd" SPKV2_HAS_NEON)

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

if(SPKV2_HAS_NEON AND NOT SPKV2_HAS_AVX2)
    set(SIMD_FLAGS "")
    if(OpenMP_C_FOUND)
        set(SIMD_FLAGS "${OpenMP_C_FLAGS}")
    endif()
    if(SIMD_FLAGS)
        set_source_files_properties(
            ${SPKV2_OPTIMIZED_KERNEL_SOURCES}
            kernels/reference/kernels.c
            PROPERTIES COMPILE_FLAGS "${SIMD_FLAGS}"
        )
    endif()
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
if(APPLE)
    target_link_libraries(spkv2_runtime PUBLIC "-framework Accelerate")
endif()
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

# Kernel unit tests
set(KERNEL_TESTS relu add softmax maxpool transpose)
foreach(name ${KERNEL_TESTS})
    add_executable(spkv2_test_kernel_${name}
        tests/test_kernel_${name}.c
    )
    target_include_directories(spkv2_test_kernel_${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/core
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/reference
    )
    target_link_libraries(spkv2_test_kernel_${name} PRIVATE spkv2_runtime)
    add_test(NAME runtime_kernel_${name} COMMAND spkv2_test_kernel_${name})
endforeach()

add_executable(spkv2_test_register_kernel
    tests/test_register_kernel.c
)
target_include_directories(spkv2_test_register_kernel PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/reference
)
target_link_libraries(spkv2_test_register_kernel PRIVATE spkv2_runtime)
add_test(NAME runtime_register_kernel COMMAND spkv2_test_register_kernel)

add_executable(spkv2_run
    tools/spkv2_run.c
)

target_link_libraries(spkv2_run PRIVATE spkv2_runtime m)

add_executable(spkv2_bench
    tools/spkv2_bench.c
)

target_link_libraries(spkv2_bench PRIVATE spkv2_runtime m)

add_executable(spkv2_bench_bnns
    tools/bench_bnns_int8.c
)

target_compile_options(spkv2_bench_bnns PRIVATE -Wno-deprecated-declarations)
target_link_libraries(spkv2_bench_bnns PRIVATE spkv2_runtime m)

# Shared library for ctypes / Python driver
add_library(spkv2_runtime_shared SHARED
    core/version.c
    core/loader.c
    core/executor.c
    ${SPKV2_REFERENCE_KERNEL_SOURCES}
    ${SPKV2_OPTIMIZED_KERNEL_SOURCES}
    ${SPKV2_PLATFORM_SOURCE}
)

set_target_properties(spkv2_runtime_shared PROPERTIES
    OUTPUT_NAME spkv2_runtime
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(spkv2_runtime_shared
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/reference
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/optimized
)

target_link_libraries(spkv2_runtime_shared PUBLIC m)
if(APPLE)
    target_link_libraries(spkv2_runtime_shared PUBLIC "-framework Accelerate")
endif()
if(OpenMP_C_FOUND)
    target_link_libraries(spkv2_runtime_shared PUBLIC OpenMP::OpenMP_C)
endif()

if(SPKV2_HAS_AVX2)
    get_target_property(_shared_sources spkv2_runtime_shared SOURCES)
    set(_shared_opt "")
    foreach(src ${_shared_sources})
        get_filename_component(_dir ${src} DIRECTORY)
        if("${_dir}" MATCHES "optimized" OR "${src}" MATCHES "kernels\\.c$")
            list(APPEND _shared_opt ${src})
        endif()
    endforeach()
    if(_shared_opt)
        set(_shared_flags "-mavx2 -mfma")
        if(OpenMP_C_FOUND)
            set(_shared_flags "${_shared_flags} ${OpenMP_C_FLAGS}")
        endif()
        set_source_files_properties(${_shared_opt}
            TARGET_DIRECTORY spkv2_runtime_shared
            PROPERTIES COMPILE_FLAGS "${_shared_flags}")
    endif()
endif()

# Reliability integration test
add_executable(spkv2_reliability_test
    tests/test_reliability.c
)
target_link_libraries(spkv2_reliability_test PRIVATE spkv2_runtime m)

# Fault injection CLI tool
add_executable(spkv2_fault_run
    tools/spkv2_fault_run.c
)
target_link_libraries(spkv2_fault_run PRIVATE spkv2_runtime m)

# Protection primitive benchmark
add_executable(spkv2_protection_bench
    tools/spkv2_protection_bench.c
)
target_link_libraries(spkv2_protection_bench PRIVATE m)
