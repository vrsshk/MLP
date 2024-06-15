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

add_library(mlp_lib ${SOURCE_FILES})
add_executable(MLP source/source.cpp)
target_link_libraries(MLP mlp_lib)

file(COPY ${DATA_DIR}/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data)

set_target_properties(MLP PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(test source/test.cpp)
target_link_libraries(test mlp_lib)
target_include_directories(test PUBLIC ${INCLUDE_DIRS})