cmake_minimum_required(VERSION 2.8)
project(absolute_orientation)

#FIND_PACKAGE(LAPACK)

IF(MSVC_IDE)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
ELSE(MSVC_IDE)
  set(CFLAGS_WARN "-Wall -Wextra -Wno-unused-parameter -Wno-strict-aliasing")
  set(CMAKE_CXX_FLAGS "-fPIC -O3 -march=native ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
ENDIF(MSVC_IDE)

INCLUDE_DIRECTORIES("/home/carl/Work/3rdparty")
INCLUDE_DIRECTORIES("/home/carl/Work/3rdparty/Eigen3")
#INCLUDE_DIRECTORIES("/Users/carlren/Work/VisLib/VisLib/Frameworks/gtsam/gtsam/3rdparty/Eigen")
#INCLUDE_DIRECTORIES("/Users/carlren/Work/3rdparty")

add_executable(test_absolute_orientation test_absolute_orientation.cpp absolute_orientation_horn.h)
target_link_libraries(test_absolute_orientation)

#add_executable(test_relative_pose test_relative_pose.cpp relative_pose_solver.h CalibStereo.h CalibStereo.cpp)
add_executable(test_relative_pose test_relative_pose.cpp relative_pose_solver.h )
target_link_libraries(test_relative_pose ${LAPACK_LIBRARIES})

add_executable(SRP solve_relative_pose.cpp relative_pose_solver.h )
target_link_libraries(SRP)
