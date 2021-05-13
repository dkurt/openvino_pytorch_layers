# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:extension]
set(CMAKE_CXX_STANDARD 11)

set(TARGET_NAME "user_cpu_extension")

if (NOT NGRAPH_LIBRARIES)
  find_package(ngraph REQUIRED)
endif()

if (NOT InferenceEngine_LIBRARIES)
  find_package(InferenceEngine REQUIRED)
endif()

if (NOT TBB_IMPORTED_TARGETS)
  find_package(TBB REQUIRED tbb tbbmalloc)
endif()

find_package(OpenCV)

if (OpenCV_FOUND)
  add_definitions(-DHAVE_OPENCV=1)
endif()

file(GLOB_RECURSE SRC *.cpp)

add_library(${TARGET_NAME} MODULE ${SRC})

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_EXTENSION_API)
target_link_libraries(${TARGET_NAME} PRIVATE ${InferenceEngine_LIBRARIES}
                                             ${NGRAPH_LIBRARIES}
                                             ${OpenCV_LIBRARIES}
                                             ${TBB_IMPORTED_TARGETS})
# [cmake:extension]
