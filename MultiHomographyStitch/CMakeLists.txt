cmake_minimum_required(VERSION 2.8)
project(MotionDetection)

find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})


add_executable(MultiHomographyStitch
MultiHomographyStitch.h
MultiHomographyStitch.cpp
main.cpp)

target_link_libraries(MultiHomographyStitch ${OpenCV_LIBS})


