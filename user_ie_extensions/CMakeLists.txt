# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:extension]
set(TARGET_NAME "user_cpu_extension")

# find_package(ngraph REQUIRED)
# find_package(InferenceEngine REQUIRED)
# find_package(TBB REQUIRED tbb tbbmalloc)
find_package(OpenCV)

if (OpenCV_FOUND)
 add_definitions(-DHAVE_OPENCV=1)
endif()

file(GLOB_RECURSE SRC *.cpp)

add_library(${TARGET_NAME} SHARED ${SRC})

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_EXTENSION_API)
target_include_directories(${TARGET_NAME} PRIVATE ${InferenceEngine_INCLUDE_DIRS}
                                                  ${NGRAPH_INCLUDE_DIRS}
                                                  ${TBB_INCLUDE_DIR})
target_link_libraries(${TARGET_NAME} PRIVATE ${InferenceEngine_LIBRARIES}
                                             ${NGRAPH_LIBRARIES}
                                             ${OpenCV_LIBRARIES}
                                             ${TBB_IMPORTED_TARGETS})
# [cmake:extension]
