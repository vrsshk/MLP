cmake_minimum_required(VERSION 3.10)
project(MLP)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES
    source/source.cpp
    source/func.cpp
    source/matrix.cpp
    source/network.cpp
)

set(INCLUDE_DIRS
    include
)

set(DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data)
add_definitions(-DDATA_DIR="${CMAKE_CURRENT_BINARY_DIR}/data/")

include_directories(${INCLUDE_DIRS})

add_executable(MLP ${SOURCE_FILES})

file(COPY ${DATA_DIR}/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data)

set_target_properties(MLP PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)