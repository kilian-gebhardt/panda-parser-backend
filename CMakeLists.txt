cmake_minimum_required(VERSION 3.6)
project(STermParser)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -O3 -msse2 -ffast-math")

include_directories(/usr/include/eigen3)

set(SOURCE_FILES main.cpp HybridTree.h SDCP.h SDCP_Parser.h util.h Trace.h SplitMergeUtil.h eigen_test.cpp EigenUtil.h)

add_executable(STermParser main.cpp)