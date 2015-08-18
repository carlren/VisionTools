cmake_minimum_required(VERSION 2.8)
project(PWP)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories("/home/carl/Work/3rdparty")
# include_directories("/home/carl/Work/3rdparty/Eigen3")

find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} )

add_executable(PWP
pwp/Histogram.hpp
pwp/PWPTracker.hpp
pwp/PWPTracker.cpp
main.cpp)

target_link_libraries(PWP ${OpenCV_LIBS})


