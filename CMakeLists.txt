cmake_minimum_required(VERSION 3.6)
project(sterm_parser)

set(CMAKE_CXX_STANDARD 14)


set(SOURCE_FILES
        HybridTree.h
        main.cpp
        SDCP.h
        SDCP_Parser.h
        SplitMergeUtil.h
        Trace.h
        util.h LCFRS.h)

set(MANAGER_SOURCES
        manager.cpp Manager.h Hypergraph.h)

set(LCFRS_SOURCES
        LCFRS.h
        main_lcfrs.cpp LCFRS_Parser.h LCFRS_util.h
        Manager.h Hypergraph.h)




set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wextra")

add_executable(sterm_parser ${SOURCE_FILES})
add_executable(lcfrs_parser ${LCFRS_SOURCES})
add_executable(manager ${MANAGER_SOURCES})
