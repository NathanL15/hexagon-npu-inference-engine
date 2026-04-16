cmake_minimum_required(VERSION 3.15)
project(npu_inference_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(npu_inference_engine
    src/main.cpp
    src/QnnManager.cpp
)

# Link Qualcomm SDK libraries here once available.
# target_link_libraries(npu_inference_engine PRIVATE qnn_sdk)
