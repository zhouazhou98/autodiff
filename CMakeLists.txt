cmake_minimum_required(VERSION 3.10)

project(AutoDiff VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${PROJECT_SOURCE_DIR}/src)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_library(autodiff 
                src/autodiff/autodiff.cc
                src/autodiff/node.cc
            )

# 添加源文件
add_executable(test_autodiff tests/test_autodiff.cc)
target_link_libraries(test_autodiff autodiff)

