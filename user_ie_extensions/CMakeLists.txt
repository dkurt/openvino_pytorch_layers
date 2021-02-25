# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:extension]
set(TARGET_NAME "user_cpu_extension")

option(USE_MKL "Use oneMKL library for FFT implementation" OFF)

find_package(ngraph REQUIRED)
find_package(InferenceEngine REQUIRED)
if (USE_MKL)
  find_package(LAPACK REQUIRED)
  add_definitions(-DHAVE_MKL=1)
else()
  find_package(OpenCV REQUIRED)
  find_package(TBB REQUIRED tbb tbbmalloc)
  add_definitions(-DHAVE_OPENCV=1)
endif()

file(GLOB_RECURSE SRC *.cpp)

add_library(${TARGET_NAME} SHARED ${SRC})

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_EXTENSION_API)
target_link_libraries(${TARGET_NAME} PRIVATE ${InferenceEngine_LIBRARIES}
                                             ${NGRAPH_LIBRARIES})
if (NOT USE_MKL)
  target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCV_LIBRARIES}
                                               ${TBB_IMPORTED_TARGETS})
endif()

# [cmake:extension]
