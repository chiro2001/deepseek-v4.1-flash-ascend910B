cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(v41_merge LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
set(ACL_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")

add_library(v41merge_ops SHARED
    op_kernel/v41_merge.asc
    op_host/v41_merge_host.cpp
)

target_include_directories(v41merge_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${ACL_INCLUDE_DIR}
)

target_compile_options(v41merge_ops PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

target_link_libraries(v41merge_ops PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
    ascendcl
    ascendc_runtime
)

target_link_directories(v41merge_ops PRIVATE ${ACL_LIB_DIR})
