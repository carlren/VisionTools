cmake_minimum_required(VERSION 2.8)
project(MotionDetection)

#FIND_PACKAGE(LAPACK)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS chrono filesystem program_options system thread)

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

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})


add_executable(MotionDetection
prob_model.h
params.h
MotionDetector.h
MotionDetector.cpp
main.cpp)

target_link_libraries(MotionDetection ${OpenCV_LIBS} ${Boost_LIBRARIES})


