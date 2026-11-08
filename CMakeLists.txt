cmake_minimum_required(VERSION 3.16)
project(gpu_sparse_ids VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------- SIMD / AVX2 support -------
# Enable AVX2 intrinsics for SIMD-vectorized pattern scanning.
# The guard __AVX2__ in source code provides a scalar fallback
# if the compiler/CPU does not support AVX2.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    add_compile_options(-mavx2)
    message(STATUS "AVX2 SIMD enabled (-mavx2)")
else()
    message(STATUS "AVX2 not supported — using scalar fallback")
endif()

# ------- Options -------
option(USE_CUDA "Enable CUDA/cuSPARSE GPU backend" OFF)

# ------- Dependencies -------
# OpenMP
find_package(OpenMP REQUIRED)

# libpcap
find_library(PCAP_LIBRARY pcap)
find_path(PCAP_INCLUDE_DIR pcap/pcap.h)
if(NOT PCAP_LIBRARY OR NOT PCAP_INCLUDE_DIR)
    message(FATAL_ERROR "libpcap not found. Install libpcap-dev.")
endif()
message(STATUS "libpcap: ${PCAP_LIBRARY}")

# ------- Core library -------
set(CORE_SOURCES
    src/ingestion/pcap_reader.cpp
    src/signatures/signature_db.cpp
    src/features/ngram_features.cpp
    src/sparse/csr_matrix.cpp
    src/compute/cpu_engine.cpp
    src/compute/openmp_engine.cpp
    src/output/alert_manager.cpp
)

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND CORE_SOURCES src/cuda/cuda_engine.cu)
    add_definitions(-DUSE_CUDA)
    message(STATUS "CUDA enabled: ${CUDAToolkit_VERSION}")
else()
    list(APPEND CORE_SOURCES src/compute/cuda_engine_stub.cpp)
    message(STATUS "CUDA disabled — using stub")
endif()

add_library(ids_core STATIC ${CORE_SOURCES})
target_include_directories(ids_core PUBLIC include ${PCAP_INCLUDE_DIR})
target_link_libraries(ids_core PUBLIC OpenMP::OpenMP_CXX ${PCAP_LIBRARY})

if(USE_CUDA)
    target_link_libraries(ids_core PUBLIC CUDA::cusparse CUDA::cudart)
    set_target_properties(ids_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# ------- Executables -------

# Main IDS
add_executable(sparse_ids src/main.cpp)
target_link_libraries(sparse_ids ids_core)

# Test data generator
add_executable(generate_test_data src/generate_test_data.cpp)
target_link_libraries(generate_test_data ids_core)

# Test suite
add_executable(test_ids tests/test_ids.cpp)
target_link_libraries(test_ids ids_core)

# Benchmark
add_executable(benchmark_ids src/benchmark.cpp)
target_link_libraries(benchmark_ids ids_core)

# ------- Install -------
install(TARGETS sparse_ids generate_test_data test_ids benchmark_ids
        RUNTIME DESTINATION bin)

