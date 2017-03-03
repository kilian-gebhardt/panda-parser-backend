cmake_minimum_required(VERSION 3.6)
project(STermParser)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wextra -lpthread")

include_directories(/usr/include/eigen3)

set(SOURCE_FILES main.cpp HybridTree.h SDCP.h SDCP_Parser.h util.h Trace.h SplitMergeUtil.h EigenUtil.h GrammarInfo.h)

set(EIGEN_SOURCES HybridTree.h SDCP.h SDCP_Parser.h util.h Trace.h SplitMergeUtil.h eigen_test.cpp EigenUtil.h)

set(TRACEMANAGER_SOURCES main_tracemanager.cpp HybridTree.h SDCP.h SDCP_Parser.h util.h Trace.h SplitMergeUtil.h EigenUtil.h TraceManager.h StorageManager.h SplitMergeTrainer.h EMTrainerLA.h TrainingCommon.h Names.h EMTrainer.h TrainerFactory.h)

set(MANAGER_SOURCES
        manager.cpp Manager.h Hypergraph.h GrammarInfo.h)

set(LCFRS_SOURCES
        LCFRS.h
        main_lcfrs.cpp LCFRS_Parser.h LCFRS_util.h
        Manager.h Hypergraph.h)



add_executable(STermParser ${SOURCE_FILES})
add_executable(EigenTest ${EIGEN_SOURCES})
add_executable(TraceManager ${TRACEMANAGER_SOURCES})
add_executable(lcfrs_parser ${LCFRS_SOURCES})
add_executable(manager ${MANAGER_SOURCES})
