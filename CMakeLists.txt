cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
PROJECT(calib_laser)

FIND_PACKAGE(PCL 1.7 REQUIRED COMPONENTS common io visualization segmentation sample_consensus kdtree features filters surface rangeimage)
#FIND_PACKAGE(PCL 1.5 REQUIRED)

find_package(Boost 1.40.0 COMPONENTS system filesystem thread date_time)
include_directories(${Boost_INCLUDE_DIRS})

INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})

FIND_PACKAGE( OpenCV REQUIRED )
#include_directories(EIGEN3_INCLUDE_DIR)

INCLUDE_DIRECTORIES(${OPENCV_INCLUDE} )

FIND_PACKAGE(Eigen3 )
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})



link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


set(SRCS calib_laser.cpp Lu.cpp  Utilities.cpp )


add_executable(calib_laser  ${SRCS})
target_link_libraries(calib_laser ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES}
 ${PCL_VISUALIZATION_LIBRARIES} ${PCL_SEGMENTATION_LIBRARIES} 
${PCL_KDTREE_LIBRARIES}${OpenCV_LIBS}  ${PCL_FILTERS_LIBRARIES} 
${PCL_SAMPLE_CONSENSUS_LIBRARIES} ${PCL_SURFACE_LIBRARIES} ${PCL_RANGEIMAGE_LIBRARIES})

