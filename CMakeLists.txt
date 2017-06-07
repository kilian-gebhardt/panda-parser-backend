cmake_minimum_required(VERSION 3.6)
project(STermParser)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wextra -lpthread -fopenmp")

include_directories(/usr/include/eigen3)

set(SOURCE_FILES Legacy/main.cpp DCP/HybridTree.h DCP/SDCP.h DCP/SDCP_Parser.h util.h Legacy/Trace.h Legacy/SplitMergeUtil.h Legacy/EigenUtil.h Trainer/GrammarInfo.h Trainer/LatentAnnotation.h)

set(EIGEN_SOURCES DCP/HybridTree.h DCP/SDCP.h DCP/SDCP_Parser.h util.h Legacy/Trace.h Legacy/SplitMergeUtil.h Legacy/eigen_test.cpp Legacy/EigenUtil.h)

set(TRACEMANAGER_SOURCES main_tracemanager.cpp DCP/HybridTree.h DCP/SDCP.h DCP/SDCP_Parser.h util.h Legacy/Trace.h Legacy/SplitMergeUtil.h Legacy/EigenUtil.h Trainer/TraceManager.h Trainer/StorageManager.h Trainer/SplitMergeTrainer.h Trainer/EMTrainerLA.h Trainer/TrainingCommon.h Names.h Trainer/EMTrainer.h Trainer/TrainerBuilder.h DCP/util.h Trainer/MergePreparator.h Trainer/Smoother.h Trainer/HypergraphRanker.h Trainer/Validation.h)

set(MANAGER_SOURCES
        manager.cpp Manage/Manager.h Manage/Hypergraph.h Trainer/GrammarInfo.h Manage/Manager_util.h Trainer/AnnotationProjection.h)

set(LCFRS_SOURCES
        LCFR/LCFRS.h
        LCFR/main_lcfrs.cpp LCFR/LCFRS_Parser.h LCFR/LCFRS_util.h
        Manage/Manager.h Manage/Hypergraph.h LCFR/manager_util.h)



add_executable(STermParser ${SOURCE_FILES})
add_executable(EigenTest ${EIGEN_SOURCES})
add_executable(TraceManager ${TRACEMANAGER_SOURCES})
add_executable(lcfrs_parser ${LCFRS_SOURCES})
add_executable(manager ${MANAGER_SOURCES})
