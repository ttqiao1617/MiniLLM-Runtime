cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(minillm_rt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(DEFAULT_ONNXRUNTIME_ROOT
  "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.26.0")
if(DEFINED ENV{ONNXRUNTIME_ROOT})
  set(DEFAULT_ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
endif()
set(ONNXRUNTIME_ROOT "${DEFAULT_ONNXRUNTIME_ROOT}"
  CACHE PATH "Path to ONNX Runtime C/C++ SDK")

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_ROOT}/lib")
set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_LIBRARY_DIR}/libonnxruntime.so")

set(DEFAULT_TENSORRT_ROOT "/usr/local/tensorrt")
if(DEFINED ENV{TENSORRT_ROOT})
  set(DEFAULT_TENSORRT_ROOT "$ENV{TENSORRT_ROOT}")
endif()
set(TENSORRT_ROOT "${DEFAULT_TENSORRT_ROOT}" CACHE PATH "Path to TensorRT SDK")

set(DEFAULT_CUDA_ROOT "/usr/local/cuda")
if(DEFINED ENV{CUDA_ROOT})
  set(DEFAULT_CUDA_ROOT "$ENV{CUDA_ROOT}")
endif()
set(CUDA_ROOT "${DEFAULT_CUDA_ROOT}" CACHE PATH "Path to CUDA toolkit")

set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
set(TENSORRT_LIBRARY_DIR "${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib")
set(TENSORRT_LIBRARY "${TENSORRT_LIBRARY_DIR}/libnvinfer.so")
set(CUDA_INCLUDE_DIR "${CUDA_ROOT}/include")
set(CUDA_LIBRARY_DIR "${CUDA_ROOT}/targets/x86_64-linux/lib")
set(CUDART_LIBRARY "${CUDA_LIBRARY_DIR}/libcudart.so")

if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "ONNX Runtime C++ header not found: ${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
endif()

if(NOT EXISTS "${ONNXRUNTIME_LIBRARY}")
  message(FATAL_ERROR "ONNX Runtime library not found: ${ONNXRUNTIME_LIBRARY}")
endif()

if(NOT EXISTS "${TENSORRT_INCLUDE_DIR}/NvInfer.h")
  message(FATAL_ERROR "TensorRT header not found: ${TENSORRT_INCLUDE_DIR}/NvInfer.h")
endif()

if(NOT EXISTS "${TENSORRT_LIBRARY}")
  message(FATAL_ERROR "TensorRT library not found: ${TENSORRT_LIBRARY}")
endif()

if(NOT EXISTS "${CUDA_INCLUDE_DIR}/cuda_runtime_api.h")
  message(FATAL_ERROR "CUDA runtime header not found: ${CUDA_INCLUDE_DIR}/cuda_runtime_api.h")
endif()

if(NOT EXISTS "${CUDART_LIBRARY}")
  message(FATAL_ERROR "CUDA runtime library not found: ${CUDART_LIBRARY}")
endif()

add_executable(minillm_rt
  cpp/src/main.cpp
  cpp/src/runtime.cpp
  cpp/src/runtime_session.cpp
  cpp/src/batch_plan.cpp
  cpp/src/model_spec.cpp
  cpp/src/execution_backend.cpp
  cpp/src/generation_backend.cpp
  cpp/src/device_argmax.cu
  cpp/src/kv_staging_kernels.cu
  cpp/src/ort_backend.cpp
  cpp/src/trt_backend.cpp
  cpp/src/kv_block_manager.cpp
  cpp/src/sampler.cpp
  cpp/src/profiler.cpp
  cpp/src/tensor_utils.cpp
)

target_include_directories(minillm_rt PRIVATE
  cpp/include
)

target_include_directories(minillm_rt SYSTEM PRIVATE
  "${ONNXRUNTIME_INCLUDE_DIR}"
  "${TENSORRT_INCLUDE_DIR}"
  "${CUDA_INCLUDE_DIR}"
)

target_link_libraries(minillm_rt PRIVATE
  "${ONNXRUNTIME_LIBRARY}"
  "${TENSORRT_LIBRARY}"
  "${CUDART_LIBRARY}"
)

set_target_properties(minillm_rt PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_LIBRARY_DIR};${TENSORRT_LIBRARY_DIR};${CUDA_LIBRARY_DIR}"
)

target_compile_options(minillm_rt PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wall -Wextra -Wpedantic>
)

add_executable(kv_block_manager_test
  cpp/tests/kv_block_manager_test.cpp
  cpp/src/kv_block_manager.cpp
)
target_include_directories(kv_block_manager_test PRIVATE cpp/include)
target_compile_options(kv_block_manager_test PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)
enable_testing()
add_test(NAME kv_block_manager_test COMMAND kv_block_manager_test)

add_executable(model_spec_test
  cpp/tests/model_spec_test.cpp
  cpp/src/model_spec.cpp
)
target_include_directories(model_spec_test PRIVATE cpp/include)
target_compile_options(model_spec_test PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)
add_test(NAME model_spec_test COMMAND model_spec_test)

add_executable(runtime_session_test
  cpp/tests/runtime_session_test.cpp
  cpp/src/runtime_session.cpp
  cpp/src/batch_plan.cpp
  cpp/src/sampler.cpp
  cpp/src/model_spec.cpp
)
target_include_directories(runtime_session_test PRIVATE cpp/include)
target_compile_options(runtime_session_test PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)
add_test(NAME runtime_session_test COMMAND runtime_session_test)

add_executable(batch_plan_test
  cpp/tests/batch_plan_test.cpp
  cpp/src/batch_plan.cpp
)
target_include_directories(batch_plan_test PRIVATE cpp/include)
target_compile_options(batch_plan_test PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)
add_test(NAME batch_plan_test COMMAND batch_plan_test)

add_executable(device_argmax_test
  cpp/tests/device_argmax_test.cu
  cpp/src/device_argmax.cu
)
target_include_directories(device_argmax_test PRIVATE cpp/include)
add_test(NAME device_argmax_test COMMAND device_argmax_test)

add_executable(kv_staging_kernels_test
  cpp/tests/kv_staging_kernels_test.cu
  cpp/src/kv_staging_kernels.cu
)
target_include_directories(kv_staging_kernels_test PRIVATE cpp/include)
add_test(NAME kv_staging_kernels_test COMMAND kv_staging_kernels_test)
